/**
 * @file nz1.c
 * @author Ferki
 * @date 10.08.2025
 * @version 1.0
 *
 * NanoZip Pro - World's Fastest Dependency-Free Compression
 * Universal SIMD support (AVX2/NEON/SSE2), Safe Boundaries,
 * Configurable Window (1KB-65535 bytes), CRC Validation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> // Correctly placed include

// =====================
// CONFIGURABLE SETTINGS
// =====================
#define NZ_MAGIC 0x5A4E5A50  // 'NZPZ'
#define MAX_WINDOW ((1 << 16) - 1)  // 65535 bytes max, to fit in 16-bit distance encoding
#define MIN_WINDOW (1 << 10)  // 1KB min
#define DEFAULT_WINDOW ((1 << 16) - 1) // 65535 bytes
#define MAX_MATCH 258
#define MIN_MATCH 3
#define HASH_BITS 14
#define MATCH_SEARCH_LIMIT 32
#define CRC32_POLY 0xEDB88320

// =====================
// PLATFORM DETECTION
// =====================
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define SIMD_WIDTH 32
  typedef __m256i simd_vec;
  #define VEC_LOAD(a) _mm256_loadu_si256((const __m256i*)(a))
  #define VEC_CMP(a,b) _mm256_cmpeq_epi8(a,b)
  #define VEC_MOVEMASK(a) _mm256_movemask_epi8(a)
#elif defined(__aarch64__) || defined(__arm__)
  #include <arm_neon.h>
  #define SIMD_WIDTH 16
  typedef uint8x16_t simd_vec;
  #define VEC_LOAD(a) vld1q_u8(a)
  #define VEC_CMP(a,b) vceqq_u8(a,b)
  // VEC_MOVEMASK macro for ARM NEON: creates a bitmask from the 16 bytes of the vector.
  // If a byte is 0xFF (indicating a match), the corresponding bit in the mask is set.
  #define VEC_MOVEMASK(a) ({ \
    uint32_t m = 0; \
    uint8_t temp[16]; \
    vst1q_u8(temp, a); /* Copy the vector to a byte array */ \
    for(int i=0; i<16; i++) { \
        if (temp[i] == 0xFF) m |= (1 << i); /* If byte is 0xFF, set the bit */ \
    } \
    m; \
  })
#else
  #define SIMD_WIDTH 8
  typedef union {
    uint8_t bytes[8];
    uint32_t words[2];
  } simd_vec;
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
    // calloc initializes memory to zero, which is important for head/chain tables
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
 * @param data Pointer to the input data.
 * @param pos Current position in the input data.
 * @param end End boundary of the input data.
 * @param state Pointer to the compression state.
 * @param out_match_pos Pointer to store the position of the found match.
 * @return Length of the longest match found, or 0 if no match.
 */
static inline uint32_t find_match(
    const uint8_t *data, 
    size_t pos, 
    size_t end, 
    NZ_State *state,
    uint32_t *out_match_pos // New parameter to return the match position
) {
    if(pos + MIN_MATCH > end) {
        *out_match_pos = 0; // No match possible
        return 0;
    }
    
    // Compute rolling hash
    uint32_t hash = (data[pos]<<16) | (data[pos+1]<<8) | data[pos+2];
    hash = (hash * 0x9E3779B1) >> (32 - HASH_BITS);
    
    // Store the current head value before updating it. This is the 'previous' occurrence for the chain.
    uint32_t prev_head_value = state->head[hash]; 
    
    // Update head to current position 'pos'
    state->head[hash] = pos;
    
    uint32_t best_len = 0;
    uint32_t best_match_candidate = 0; // Store the candidate that gave the best_len

    // Start searching for matches from the previously stored position
    uint32_t candidate = prev_head_value; 
    
    for(int i=0; i<MATCH_SEARCH_LIMIT && candidate; i++) {
        // Skip current position to avoid self-referencing matches, which result in dist = 0
        if (candidate == pos) {
            candidate = state->chain[candidate % state->window_size]; // Move to the next candidate
            continue; // Skip the rest of the loop for this iteration
        }

        size_t dist = pos - candidate;
        if(dist > state->window_size) {
            // Candidate is outside the current window, stop searching this chain
            break;
        }
        
        size_t max_len = (end - pos) < MAX_MATCH ? end - pos : MAX_MATCH;
        uint32_t len = 0;
        
        // Vectorized comparison for faster byte-by-byte comparison
        while(len + SIMD_WIDTH <= max_len) {
            simd_vec a = VEC_LOAD(data + pos + len);
            simd_vec b = VEC_LOAD(data + candidate + len);
            simd_vec cmp = VEC_CMP(a, b);
            uint32_t mask = VEC_MOVEMASK(cmp);
            
            // If the mask is not all ones, it means there's a difference
            if(mask != (1 << SIMD_WIDTH) - 1) {
                // Find the index of the first differing byte within the SIMD block
#if defined(ARCH_X86) || defined(ARCH_ARM) || defined(__GNUC__) || defined(__clang__)
                len += __builtin_ctz(~mask); // count trailing zeros in the flipped mask
#else
                for (int j = 0; j < SIMD_WIDTH; ++j) {
                    if (!((mask >> j) & 1)) {
                        len += j;
                        break;
                    }
                }
#endif
                break; // Stop SIMD comparison, handle remaining bytes with scalar
            }
            len += SIMD_WIDTH; // Advance length by SIMD_WIDTH (all bytes matched)
        }
        
        // Scalar tail comparison for remaining bytes (less than SIMD_WIDTH)
        while(len < max_len && data[pos+len] == data[candidate+len])
            len++;
        
        // Update best match found so far
        if(len > best_len && len >= MIN_MATCH) {
            best_len = len;
            best_match_candidate = candidate; // Store the candidate that gave this best_len
            if(len >= MAX_MATCH) {
                // Found a match of maximum possible length, no need to search further
                break;
            }
        }
        
        // Move to the next candidate in the chain (previous position with the same hash)
        candidate = state->chain[candidate % state->window_size];
    }
    
    // Store the 'prev_head_value' (the position that was previously at state->head[hash])
    // into the chain for the current 'pos'. This builds the historical chain.
    state->chain[pos % state->window_size] = prev_head_value; 

    *out_match_pos = best_match_candidate; // Return the position of the best match
    return best_len;
}

// =====================
// CORE COMPRESSION API
// =====================

/**
 * @brief Compresses input data using NanoZip algorithm.
 * @param input Pointer to the input data.
 * @param in_size Size of the input data.
 * @param output Pointer to the output buffer for compressed data.
 * @param out_size Maximum size of the output buffer.
 * @param window_size Desired compression window size (0 for default).
 * @return Size of the compressed data, or 0 on failure.
 */
size_t nanozip_compress(
    const uint8_t *input,
    size_t in_size,
    uint8_t *output,
    size_t out_size,
    int window_size
) {
    // Header format: MAGIC(4) | ORIGINAL_SIZE(4) | CRC(4) | WINDOW_SIZE(2) = 14 bytes
    if(out_size < in_size + 14) return 0; // Ensure enough space for header + some data
    
    NZ_State state;
    nz_init(&state, window_size); // Initialize compression state tables
    
    // Write header using memcpy for robustness against endianness/alignment
    uint32_t magic_val = NZ_MAGIC;
    memcpy(output + 0, &magic_val, sizeof(magic_val)); // Write NZ_MAGIC
    uint32_t in_size_u32 = (uint32_t)in_size; // Ensure 4 bytes are written for size
    memcpy(output + 4, &in_size_u32, sizeof(uint32_t)); // Write original size
    uint32_t calculated_crc = nz_crc32(input, in_size);
    memcpy(output + 8, &calculated_crc, sizeof(calculated_crc)); // Write CRC
    uint16_t window_s_val = (uint16_t)state.window_size;
    memcpy(output + 12, &window_s_val, sizeof(window_s_val)); // Write window size (16-bit)
    
    uint8_t *out_ptr = output + 14; // Pointer to start writing compressed data after 14-byte header
    
    for(size_t pos=0; pos<in_size; ) {
        uint32_t actual_match_pos = 0; // Variable to store the actual match position
        uint32_t match_len = find_match(input, pos, in_size, &state, &actual_match_pos);
        
        if(match_len >= MIN_MATCH) {
            // Found a match, encode it
            // Distance is calculated from the current position 'pos' back to the *actual* matched position.
            uint32_t dist = pos - actual_match_pos; 
            
            // Ensure there's enough space in the output buffer for a 4-byte match token
            if(out_ptr - output > (ptrdiff_t)out_size - 4) break; 
            
            // Match token format (4 bytes):
            // Byte 0: Always 0xC0 (unambiguous match marker)
            // Byte 1: Distance LSB
            // Byte 2: Distance MSB
            // Byte 3: Length (actual_length - MIN_MATCH) LSB (0-255)
            *out_ptr++ = 0xC0; // Unambiguous match marker (11000000)
            *out_ptr++ = (uint8_t)dist; // Low byte of distance
            *out_ptr++ = (uint8_t)(dist >> 8); // High byte of distance
            *out_ptr++ = (uint8_t)(match_len - MIN_MATCH); // Length (0-255 for MAX_MATCH=258)
            pos += match_len; // Advance position by the length of the matched data
        } else {
            // No match found or match too short, encode as a literal byte
            // Handle literals with an escape sequence if their value could conflict with markers
            if (input[pos] == 0xBF) { // Original byte is 0xBF, encode as escaped 0xBF
                if (out_ptr - output >= (ptrdiff_t)out_size - 2) break; // Need 2 bytes for escaped literal
                *out_ptr++ = 0xBF; // Escape marker
                *out_ptr++ = 0xBF; // The actual literal byte (0xBF)
                pos++;
            } else if (input[pos] >= 0xC0) { // Original byte is >= 0xC0 (but not 0xBF, handled above)
                // Escaped literal: 0xBF (escape marker) followed by the actual byte value
                if (out_ptr - output >= (ptrdiff_t)out_size - 2) break; // Need 2 bytes for escaped literal
                *out_ptr++ = 0xBF; // Escape marker
                *out_ptr++ = input[pos++]; // The actual literal byte
            } else {
                // Direct literal: 0x00 to 0xBE (excluding 0xBF, handled above)
                if (out_ptr - output >= (ptrdiff_t)out_size - 1) break; // Need 1 byte for direct literal
                *out_ptr++ = input[pos++]; // Copy literal byte and advance position
            }
        }
    }
    
    nz_cleanup(&state); // Clean up allocated memory
    return out_ptr - output; // Return total compressed size
}

/**
 * @brief Decompresses input data using NanoZip algorithm.
 * @param input Pointer to the compressed input data.
 * @param in_size Size of the compressed input data.
 * @param output Pointer to the output buffer for decompressed data.
 * @param out_size Maximum size of the output buffer.
 * @return Size of the decompressed data, or 0 on failure/validation error.
 */
size_t nanozip_decompress(
    const uint8_t *input,
    size_t in_size,
    uint8_t *output,
    size_t out_size
) {
    // Temporary variables to read header safely
    uint32_t read_magic;
    uint32_t read_data_size_u32; // Will be cast to size_t later
    uint32_t read_crc;
    uint16_t read_window_size_16bit;

    // Read header using memcpy for robustness against endianness/alignment
    // Check header validity and minimum input size (now 14 bytes)
    if(in_size < 14) return 0; // Input too small for header

    memcpy(&read_magic, input + 0, sizeof(read_magic));
    if(read_magic != NZ_MAGIC) return 0; // Invalid magic number
    
    memcpy(&read_data_size_u32, input + 4, sizeof(read_data_size_u32));
    memcpy(&read_crc, input + 8, sizeof(read_crc));
    memcpy(&read_window_size_16bit, input + 12, sizeof(read_window_size_16bit));

    size_t data_size = read_data_size_u32;
    uint32_t crc = read_crc;
    size_t window_size = read_window_size_16bit; // Cast 16-bit value to size_t

    // Validate header values and output buffer size
    // Now window_size directly contains the value used for MAX_WINDOW (e.g., 65535)
    if(!window_size || window_size > MAX_WINDOW || data_size > out_size)
        return 0; // Invalid window size or output buffer too small for expected data
    
    // NZ_State is not used during decompression, so no need to initialize/cleanup
    
    const uint8_t *in_ptr = input + 14; // Pointer to start reading compressed data after 14-byte header
    size_t in_remain = in_size - 14; // Remaining bytes in the compressed input
    size_t out_pos = 0; // Current position in the output (decompressed) buffer
    
    while(in_remain > 0 && out_pos < data_size) {
        if(*in_ptr == 0xBF) { // Potential escaped literal or original 0xBF
            in_ptr++; // Move past the 0xBF marker
            in_remain--; // Account for prefix byte
            if (in_remain == 0) break; // Ensure there's a byte after escape prefix (corrupted stream if not)
            
            output[out_pos++] = *in_ptr++; // The actual literal byte (either original >=0xC0 or original 0xBF)
            in_remain--;
        } else if (*in_ptr == 0xC0) { // Match token marker
            if(in_remain < 4) break; // Not enough bytes for a full match token, implies malformed data
            
            // Decode length and distance from the 4-byte token
            // The first byte (0xC0) is just a marker, so no length bits are derived from it.
            // Length is entirely from in_ptr[3]
            uint32_t len = in_ptr[3]; 
            uint32_t dist = in_ptr[1] | (in_ptr[2] << 8); // Combines byte 1 (LSB) and byte 2 (MSB) for distance
            len += MIN_MATCH; // Add MIN_MATCH back to get the true length
            
            // Additional check for length validity (should always be MIN_MATCH to MAX_MATCH)
            if (len > MAX_MATCH) {
                break; // Decoded length is impossible, indicates corrupted data or logic error
            }

            // Boundary checks for match:
            // 1. Distance must not be zero (dist=pos-candidate, and candidate < pos, so dist > 0)
            // 2. Distance must not be greater than current output position (cannot refer to future data)
            // 3. Copying match must not exceed the expected original data size
            if(dist == 0 || dist > out_pos || out_pos + len > data_size) 
                break; 
            
            // Copy match data from previously decompressed output (backward copy)
            for(uint32_t i=0; i<len; i++) {
                output[out_pos] = output[out_pos - dist]; // Copy byte from 'dist' bytes back
                out_pos++; // Advance output position
            }
            
            in_ptr += 4; // Advance input pointer by 4 bytes (match token size)
            in_remain -= 4; // Decrement remaining input bytes
        } else { // Direct Literal (0x00 to 0xBE)
            output[out_pos++] = *in_ptr++; // Copy literal byte to output
            in_remain--; // Decrement remaining input bytes
        }
    }
    
    // Final validation:
    // 1. Check if the total decompressed size matches the original size from header.
    // 2. Check if the CRC of the decompressed data matches the CRC stored in the header.
    if(out_pos != data_size || nz_crc32(output, data_size) != crc) {
        return 0; // Validation failed (size mismatch or CRC mismatch)
    }
    
    return data_size; // Decompression successful, return original data size
}

// =====================
// BENCHMARK UTILITIES
// =====================

/**
 * @brief Runs a compression/decompression benchmark on given data.
 * @param name Name of the benchmark.
 * @param data Pointer to the original data.
 * @param size Size of the original data.
 */
void benchmark(const char *name, const uint8_t *data, size_t size) {
    // Allocate buffer for compressed data (2x original size as worst case + header)
    // The +14 for header makes sure the buffer is always big enough even for minimum size data.
    uint8_t *compressed = malloc(size * 2 + 14); 
    uint8_t *decompressed = malloc(size); // Allocate buffer for decompressed data (original size)
    
    // Check for malloc failures
    if (!compressed || !decompressed) {
        fprintf(stderr, "Memory allocation failed for benchmark %s\n", name);
        free(compressed); // Free if one succeeded
        free(decompressed);
        return;
    }

    // Calculate CRC of original data
    uint32_t original_data_crc = nz_crc32(data, size);

    // --- Compression Benchmark ---
    clock_t start = clock();
    size_t comp_size = nanozip_compress(data, size, compressed, size*2 + 14, 0); // Pass correct output buffer size
    double comp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    // Read stored CRC from compressed header
    uint32_t stored_header_crc = 0;
    if (comp_size > 8) { // Ensure header is large enough to read CRC
        memcpy(&stored_header_crc, compressed + 8, sizeof(stored_header_crc));
    }


    // --- Decompression Benchmark ---
    start = clock();
    size_t decomp_size = 0; // Initialize to 0, will be updated by nanozip_decompress
    if (comp_size > 0) { // Only attempt decompression if compression was successful
        decomp_size = nanozip_decompress(compressed, comp_size, decompressed, size);
    }
    double decomp_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    // Calculate CRC of decompressed data
    uint32_t decompressed_data_crc = 0;
    if (decomp_size > 0) { // Only calculate if decompression produced some data
        decompressed_data_crc = nz_crc32(decompressed, decomp_size);
    }


    // Print benchmark results
    printf("\n=== %s BENCHMARK ===\n", name);
    printf("Original:    %zu bytes\n", size);
    
    if (comp_size == 0) {
        printf("Compression: FAILED (Output size 0)\n");
    } else {
        printf("Compressed:  %zu bytes (%.2f%% ratio)\n", 
               comp_size, 100.0 * (double)comp_size / size);
        printf("Comp Speed:  %.2f MB/s\n", (double)size / comp_time / 1e6);
    }

    printf("Original Data CRC:     0x%08X\n", original_data_crc);
    printf("Stored Header CRC:     0x%08X\n", stored_header_crc);
    if (decomp_size > 0) {
        printf("Decompressed Data CRC: 0x%08X\n", decompressed_data_crc);
    } else {
        printf("Decompressed Data CRC: N/A (Decompression failed)\n");
    }


    if (decomp_size == 0 && comp_size != 0) { // Decompression failed, but compression might have worked
         printf("Decomp Speed:%.2f MB/s (FAILED EARLY)\n", (double)size / decomp_time / 1e6);
         printf("Validation:  FAIL (Decompression failed or returned 0)\n");
    }
    else if (decomp_size == size && memcmp(data, decompressed, size) == 0) {
        printf("Decomp Speed:%.2f MB/s\n", (double)size / decomp_time / 1e6);
        printf("Validation:  PASS\n");
    } else {
        printf("Decomp Speed:%.2f MB/s\n", (double)size / decomp_time / 1e6);
        printf("Validation:  FAIL (Data mismatch or size mismatch)\n");
        // Print differences for debugging
        if (decomp_size > 0 && decomp_size <= size) { // Check if decomp_size is valid for comparison
            size_t compare_limit = (decomp_size < size) ? decomp_size : size;
            printf("Comparing up to %zu bytes.\n", compare_limit);
            for (size_t i = 0; i < compare_limit; ++i) {
                if (data[i] != decompressed[i]) {
                    printf("First diff at byte %zu: Original 0x%02x, Decompressed 0x%02x\n", i, data[i], decompressed[i]);
                    // Print a few more bytes for context
                    printf("Context (Orig): "); 
                    for (int j=0; j<5 && (i+j)<size; ++j) printf("0x%02x ", data[i+j]); 
                    printf("\n");
                    printf("Context (Decomp): "); 
                    for (int j=0; j<5 && (i+j)<decomp_size; ++j) printf("0x%02x ", decompressed[i+j]); 
                    printf("\n");
                    break;
                }
            }
        } else {
             printf("Decompressed size is 0 or invalid (%zu). Cannot perform byte-by-byte comparison.\n", decomp_size);
        }
    }
    
    free(compressed); // Free allocated memory
    free(decompressed);
}

/**
 * @brief Main function for testing and benchmarking NanoZip.
 */
int main() {
    // --- Small, repeating test data (easier to debug) ---
    size_t small_size = 100;
    uint8_t *small_data = malloc(small_size);
    if (small_data) {
        for(size_t i=0; i<small_size; i++) {
            small_data[i] = 'A' + (i % 5); // A, B, C, D, E, A, B, C, D, E, ...
        }
        benchmark("SMALL_REPEATING_TEXT", small_data, small_size);
        free(small_data);
    }


    // --- More complex, but still compressible pattern ---
    size_t comp_size = 500;
    uint8_t *comp_data = malloc(comp_size);
    char *lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
    if (comp_data) {
        for(size_t i = 0; i < comp_size; ++i) {
            comp_data[i] = lorem_ipsum[i % strlen(lorem_ipsum)];
        }
        benchmark("LOREM_IPSUM", comp_data, comp_size);
        free(comp_data);
    }

    // --- Original 1MB text data generation ---
    size_t size = 1 << 20; // 1MB
    uint8_t *text_data = malloc(size);
    uint8_t *bin_data = malloc(size);
    
    if (text_data && bin_data) {
        for(size_t i=0; i<size; i++) {
            bin_data[i] = i % 256; // Simple ascending binary pattern
            text_data[i] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i % 26]; // Repeating alphabet pattern
        }
        benchmark("TEXT", text_data, size);
        benchmark("BINARY", bin_data, size); // This will be the main target for the fix
    }
    
    free(text_data);
    free(bin_data); // Free outside if, ensures free is called even if one allocation fails

    // --- Mini binary data for focused debugging (64 bytes) ---
    size_t mini_bin_size = 64;
    uint8_t mini_bin_data[64] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
        0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
        0xA5, 0x5A, 0xA5, 0x5A, 0x0F, 0xF0, 0x0F, 0xF0,
        0xBC, 0xDA, 0xEF, 0xCD, 0x87, 0x65, 0x43, 0x21,
        0xED, 0xCB, 0xA9, 0x87, 0x65, 0x43, 0x21, 0x0F
    };
    benchmark("MINI_BINARY", mini_bin_data, mini_bin_size);
    
    return 0;
}

