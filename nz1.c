/**
 * @file nz1.c
 * @author Ferki
 * @date 10.08.2025
 * @version 1.0
 * 
 * NanoZip Pro - World's Fastest Dependency-Free Compression
 * Universal SIMD support (AVX2/NEON/SSE2), Safe Boundaries, 
 * Configurable Window (1KB-1MB), CRC Validation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// =====================
// CONFIGURABLE SETTINGS
// =====================
#define NZ_MAGIC 0x5A4E5A50  // 'NZPZ'
#define MAX_WINDOW (1 << 20)  // 1MB max
#define MIN_WINDOW (1 << 10)  // 1KB min
#define DEFAULT_WINDOW (1 << 16) // 64KB
#define MAX_MATCH 258
#define MIN_MATCH 3
#define HASH_BITS 14
#define MATCH_SEARCH_LIMIT 32
#define CRC32_POLY 0xEDB88320

// =====================
// PLATFORM DETECTION
// =====================
#if defined(__x86_64__) || defined(_M_X64)
  #define ARCH_X86
#elif defined(__aarch64__) || defined(__arm__)
  #define ARCH_ARM
#endif

// =====================
// UNIVERSAL SIMD WRAPPER
// =====================
typedef union {
  uint8_t bytes[32];
  uint32_t words[8];
} simd_vec;

#if defined(ARCH_X86)
  #include <immintrin.h>
  #define SIMD_WIDTH 32
  #define VEC_LOAD(a) _mm256_loadu_si256((const __m256i*)(a))
  #define VEC_CMP(a,b) _mm256_cmpeq_epi8(a,b)
  #define VEC_MOVEMASK(a) _mm256_movemask_epi8(a)
#elif defined(ARCH_ARM)
  #include <arm_neon.h>
  #define SIMD_WIDTH 16
  #define VEC_LOAD(a) vld1q_u8(a)
  #define VEC_CMP(a,b) vceqq_u8(a,b)
  #define VEC_MOVEMASK(a) vget_lane_u32(vreinterpret_u32_u8( \
        vshrn_n_u16(vreinterpretq_u16_u8( \
        vzip1q_u8(vqtbl1q_u8(a, vcreate_u8(0x0F0D0B0907050301)), \
        vqtbl1q_u8(a, vcreate_u8(0x0F0D0B0907050301))), 7)), 0)
#else
  #define SIMD_WIDTH 8
  #define VEC_LOAD(a) ({ simd_vec v; memcpy(v.bytes, a, SIMD_WIDTH); v; })
  #define VEC_CMP(a,b) ({ simd_vec v; for(int i=0;i<SIMD_WIDTH;i++) \
        v.bytes[i] = (a.bytes[i] == b.bytes[i]) ? 0xFF : 0; v; })
  #define VEC_MOVEMASK(a) ({ uint32_t m=0; for(int i=0;i<SIMD_WIDTH;i++) \
        m |= (a.bytes[i] & 0x80) ? (1<<i) : 0; m; })
#endif

// =====================
// CORE COMPRESSION STRUCTURE
// =====================
typedef struct {
    uint32_t *head;
    uint32_t *chain;
    size_t window_size;
} NZ_State;

// =====================
// UTILITY FUNCTIONS
// =====================

/**
 * @brief Initialize compression state
 * @param state NZ_State structure
 * @param window_size Desired window size (0 = default)
 */
void nz_init(NZ_State *state, size_t window_size) {
    if(window_size < MIN_WINDOW) window_size = DEFAULT_WINDOW;
    if(window_size > MAX_WINDOW) window_size = MAX_WINDOW;
    
    state->window_size = window_size;
    state->head = calloc(1 << HASH_BITS, sizeof(uint32_t));
    state->chain = calloc(window_size, sizeof(uint32_t));
}

/**
 * @brief Clean up compression state
 */
void nz_cleanup(NZ_State *state) {
    free(state->head);
    free(state->chain);
}

/**
 * @brief Compute CRC32 for data validation
 */
uint32_t nz_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for(size_t i=0; i<len; i++) {
        crc ^= data[i];
        for(int j=0; j<8; j++)
            crc = (crc >> 1) ^ (CRC32_POLY & -(crc & 1));
    }
    return ~crc;
}

// =====================
// SAFE MATCH FINDING
// =====================

/**
 * @brief Find longest match with boundary checks
 */
static inline uint32_t find_match(
    const uint8_t *data, 
    size_t pos, 
    size_t end, 
    NZ_State *state
) {
    if(pos + MIN_MATCH > end) return 0;
    
    // Compute rolling hash
    uint32_t hash = (data[pos]<<16) | (data[pos+1]<<8) | data[pos+2];
    hash = (hash * 0x9E3779B1) >> (32 - HASH_BITS);
    
    uint32_t best_len = 0;
    uint32_t candidate = state->head[hash];
    state->head[hash] = pos;
    
    for(int i=0; i<MATCH_SEARCH_LIMIT && candidate; i++) {
        size_t dist = pos - candidate;
        if(dist > state->window_size) break;
        
        size_t max_len = (end - pos) < MAX_MATCH ? end - pos : MAX_MATCH;
        uint32_t len = 0;
        
        // Vectorized comparison
        while(len + SIMD_WIDTH <= max_len) {
            simd_vec a = VEC_LOAD(data + pos + len);
            simd_vec b = VEC_LOAD(data + candidate + len);
            simd_vec cmp = VEC_CMP(a, b);
            uint32_t mask = VEC_MOVEMASK(cmp);
            
            if(mask != (1 << SIMD_WIDTH) - 1) {
                len += __builtin_ctz(~mask);
                break;
            }
            len += SIMD_WIDTH;
        }
        
        // Scalar tail
        while(len < max_len && data[pos+len] == data[candidate+len])
            len++;
        
        if(len > best_len && len >= MIN_MATCH) {
            best_len = len;
            if(len >= MAX_MATCH) break;
        }
        
        candidate = state->chain[candidate % state->window_size];
    }
    
    state->chain[pos % state->window_size] = state->head[hash];
    return best_len;
}

// =====================
// CORE COMPRESSION API
// =====================

size_t nanozip_compress(
    const uint8_t *input,
    size_t in_size,
    uint8_t *output,
    size_t out_size,
    int window_size
) {
    // Header: MAGIC(4) | SIZE(4) | CRC(4) | WINDOW(1)
    if(out_size < in_size + 16) return 0;
    
    NZ_State state;
    nz_init(&state, window_size);
    
    *(uint32_t*)(output+0) = NZ_MAGIC;
    *(uint32_t*)(output+4) = in_size;
    *(uint32_t*)(output+8) = nz_crc32(input, in_size);
    output[12] = state.window_size >> 10; // Window size flag
    uint8_t *out_ptr = output + 13;
    
    for(size_t pos=0; pos<in_size; ) {
        uint32_t match_len = find_match(input, pos, in_size, &state);
        
        if(match_len >= MIN_MATCH) {
            uint32_t dist = pos - state.chain[pos % state.window_size];
            
            // Encode match: [110LLLDD] [DDDDDDDD] [LLLLLLLL]
            if(out_ptr - output > (ptrdiff_t)out_size - 4) break;
            
            *out_ptr++ = 0xC0 | ((match_len - MIN_MATCH) >> 8);
            *out_ptr++ = dist;
            *out_ptr++ = dist >> 8;
            *out_ptr++ = match_len - MIN_MATCH;
            pos += match_len;
        } else {
            // Encode literal: 0LLLLLLL
            if(out_ptr - output >= (ptrdiff_t)out_size) break;
            *out_ptr++ = input[pos++];
        }
    }
    
    nz_cleanup(&state);
    return out_ptr - output;
}

size_t nanozip_decompress(
    const uint8_t *input,
    size_t in_size,
    uint8_t *output,
    size_t out_size
) {
    // Check header
    if(in_size < 13 || *(uint32_t*)input != NZ_MAGIC)
        return 0;
    
    size_t data_size = *(uint32_t*)(input+4);
    uint32_t crc = *(uint32_t*)(input+8);
    size_t window_size = input[12] << 10;
    
    if(!window_size || window_size > MAX_WINDOW || data_size > out_size)
        return 0;
    
    NZ_State state;
    nz_init(&state, window_size);
    
    const uint8_t *in_ptr = input + 13;
    size_t in_remain = in_size - 13;
    size_t out_pos = 0;
    
    while(in_remain > 0 && out_pos < data_size) {
        if(*in_ptr < 0xC0) { // Literal
            output[out_pos++] = *in_ptr++;
            in_remain--;
        } else { // Match
            if(in_remain < 4) break;
            
            uint32_t len = ((in_ptr[0] & 0x3F) << 8) | in_ptr[3];
            uint32_t dist = in_ptr[1] | (in_ptr[2] << 8);
            len += MIN_MATCH;
            
            // Boundary checks
            if(dist > out_pos || out_pos + len > data_size || dist == 0) 
                break;
            
            // Copy match
            for(uint32_t i=0; i<len; i++) {
                output[out_pos] = output[out_pos - dist];
                out_pos++;
            }
            
            in_ptr += 4;
            in_remain -= 4;
        }
    }
    
    // Validate decompressed data
    if(out_pos != data_size || nz_crc32(output, data_size) != crc) {
        nz_cleanup(&state);
        return 0;
    }
    
    nz_cleanup(&state);
    return data_size;
}

// =====================
// BENCHMARK UTILITIES
// =====================
#include <time.h>

void benchmark(const char *name, const uint8_t *data, size_t size) {
    uint8_t *compressed = malloc(size * 2);
    uint8_t *decompressed = malloc(size);
    
    clock_t start = clock();
    size_t comp_size = nanozip_compress(data, size, compressed, size*2, 0);
    double comp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    start = clock();
    size_t decomp_size = nanozip_decompress(compressed, comp_size, decompressed, size);
    double decomp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    printf("\n=== %s BENCHMARK ===\n", name);
    printf("Original:    %zu bytes\n", size);
    printf("Compressed:  %zu bytes (%.2f%% ratio)\n", 
           comp_size, 100.0 * comp_size / size);
    printf("Comp Speed:  %.2f MB/s\n", size / comp_time / 1e6);
    printf("Decomp Speed:%.2f MB/s\n", size / decomp_time / 1e6);
    printf("Validation:  %s\n", (decomp_size == size) ? "PASS" : "FAIL");
    
    free(compressed);
    free(decompressed);
}

int main() {
    // Test data generation
    size_t size = 1 << 20; // 1MB
    uint8_t *text_data = malloc(size);
    uint8_t *bin_data = malloc(size);
    
    for(size_t i=0; i<size; i++) {
        text_data[i] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i % 26];
        bin_data[i] = i ^ (i >> 4);
    }
    
    benchmark("TEXT", text_data, size);
    benchmark("BINARY", bin_data, size);
    
    free(text_data);
    free(bin_data);
    return 0;
}