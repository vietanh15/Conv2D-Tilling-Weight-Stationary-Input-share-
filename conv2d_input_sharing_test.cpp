#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --- CẤU HÌNH BÀI TOÁN ---
#define INPUT_H 112
#define INPUT_W 112
#define INPUT_C 32
#define KERNEL_H 3
#define KERNEL_W 3
#define OUTPUT_F 1 
#define OUTPUT_H 112
#define OUTPUT_W 112
#define STRIDE 1
#define PADDING 1

// --- CẤU HÌNH PHẦN CỨNG (HW SPEC) ---
#define NUM_PE 48               
#define MACS_PER_PE 3           
#define BUFFER_SIZE_BYTES 144   // 48 PE * 3 inputs * 1 byte
#define PARALLEL_CHANNELS 16    // Số channel xử lý song song (sẽ được thay thế bởi C_SUB)

// --- CẤU HÌNH TILING (INPUT SHARING) ---
#define TILE_W 4 // Số output pixel tính toán song song theo chiều rộng
#define C_SUB 4  // Số channel xử lý trong một lần tính toán của PE array
// Kiểm tra cấu hình: TILE_W * C_SUB * KERNEL_H * KERNEL_W phải bằng NUM_PE * MACS_PER_PE
// 4 * 4 * 3 * 3 = 144, khớp với 48 * 3 = 144

// --- CẤU HÌNH HIỆU NĂNG (PERFORMANCE METRICS) ---
#define SYSTEM_FREQ_MHZ 100.0   // Tần số hoạt động: 100 MHz
#define DRAM_BUS_WIDTH_BYTES 8  // Bus 64-bit (8 bytes/cycle)
#define PE_COMPUTE_CYCLES 1     // Số cycle để PE array hoàn thành tính toán (Pipelined)

// Biến toàn cục để lưu thống kê
unsigned long long total_dma_cycles = 0;
unsigned long long total_compute_cycles = 0;
unsigned long long total_cycles = 0;

// 1. MÔ PHỎNG DRAM
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

void dram_init() {
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C * sizeof(int8_t));
    // Load IFM
    FILE* f_ifm = fopen("c:\\Code c\\Learning\\ifm.txt", "r");
    if(f_ifm) {
        char line[64];
        int idx = 0;
        while(fgets(line, 64, f_ifm)) ifm_dram[idx++] = (int8_t)atoi(line);
        fclose(f_ifm);
    } else {
        printf("Cannot open ifm.txt. Initializing with 1s.\n");
        memset(ifm_dram, 1, INPUT_H * INPUT_W * INPUT_C); 
    }

    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, sizeof(int8_t));
    // Load Weights
    FILE* f_w = fopen("c:\\Code c\\Learning\\weights.txt", "r");
    if(f_w) {
        char line[64];
        //WEIGHTS: C->W->H->F
        for(int f=0; f<OUTPUT_F; f++)
            for(int h=0; h<KERNEL_H; h++)
                for(int w=0; w<KERNEL_W; w++)
                    for(int c=0; c<INPUT_C; c++)
                        if(fgets(line, 64, f_w)) {
                             int val = atoi(line);
                             if (val > 0x7F) val -= 0x100;
                             int idx = h*(KERNEL_W*INPUT_C*OUTPUT_F) + w*(INPUT_C*OUTPUT_F) + c*OUTPUT_F + f;
                             weight_dram[idx] = (int8_t)val;
                        }
        fclose(f_w);
    } else {
        printf("Cannot open weights.txt. Initializing with 0s.\n");
    }

    ofm_dram = (int32_t*)malloc(OUTPUT_H * OUTPUT_W * OUTPUT_F * sizeof(int32_t));
}

// 2. MÔ PHỎNG BUFFER & DMA (INPUT SHARING)
int8_t buffer_ifm[BUFFER_SIZE_BYTES];
int8_t buffer_weight[BUFFER_SIZE_BYTES];

int dma_load_buffers_input_sharing(int ho, int wo_start, int pass_idx, int actual_tile_w) {
    memset(buffer_ifm, 0, BUFFER_SIZE_BYTES);
    memset(buffer_weight, 0, BUFFER_SIZE_BYTES);

    int channel_start = pass_idx * C_SUB;
    
    // --- Data Preparation for PE array ---
    // Sắp xếp dữ liệu vào buffer để PE array có thể tính toán
    for (int tile_w = 0; tile_w < actual_tile_w; tile_w++) {
        for (int c_s = 0; c_s < C_SUB; c_s++) {
            int current_c = channel_start + c_s;
            if (current_c >= INPUT_C) continue;

            for (int kh = 0; kh < KERNEL_H; kh++) {
                for (int kw = 0; kw < KERNEL_W; kw++) {
                    int mac_idx = kw + kh * KERNEL_W + c_s * (KERNEL_H * KERNEL_W) + tile_w * (C_SUB * KERNEL_H * KERNEL_W);
                    
                    // Fetch IFM
                    int wo = wo_start + tile_w;
                    int hi = ho * STRIDE + kh - PADDING;
                    int wi = wo * STRIDE + kw - PADDING;
                    int8_t val_ifm = 0;
                    if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                        int dram_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c;
                        val_ifm = ifm_dram[dram_idx];
                    }
                    buffer_ifm[mac_idx] = val_ifm;

                    // Fetch Weight
                    int w_dram_idx = kh * (KERNEL_W * INPUT_C * OUTPUT_F) +
                                     kw * (INPUT_C * OUTPUT_F) +
                                     current_c * OUTPUT_F + 0; 
                    int8_t val_w = weight_dram[w_dram_idx];
                    buffer_weight[mac_idx] = val_w;
                }
            }
        }
    }

    // --- TÍNH TOÁN LATENCY DMA ---
    // Tính toán số byte duy nhất cần tải từ DRAM
    int ifm_tile_width_needed = (actual_tile_w - 1) * STRIDE + KERNEL_W;
    int unique_ifm_bytes = 0;
    int unique_weight_bytes = 0;

    for (int c_s = 0; c_s < C_SUB; c_s++) {
        if (channel_start + c_s >= INPUT_C) break;
        unique_ifm_bytes += KERNEL_H * ifm_tile_width_needed;
        unique_weight_bytes += KERNEL_H * KERNEL_W;
    }
    
    int total_bytes = unique_ifm_bytes + unique_weight_bytes; 
    int cycles = (total_bytes + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    
    return cycles;
}

// 3. MÔ PHỎNG COMPUTE ENGINE (INPUT SHARING)
void run_pe_array_input_sharing(int* cycles_taken, int32_t* partial_sums, int actual_tile_w) {
    for(int i=0; i<TILE_W; i++) partial_sums[i] = 0;

    // Logic tính toán chức năng (Functional)
    for (int mac_idx = 0; mac_idx < NUM_PE * MACS_PER_PE; mac_idx++) {
        int tile_w = mac_idx / (C_SUB * KERNEL_H * KERNEL_W);
        
        if (tile_w >= actual_tile_w) continue;

        int8_t a = buffer_ifm[mac_idx];
        int8_t b = buffer_weight[mac_idx];
        partial_sums[tile_w] += (int32_t)a * (int32_t)b;
    }

    // --- TÍNH TOÁN LATENCY ---
    *cycles_taken = PE_COMPUTE_CYCLES;
}

// 4. CONTROLLER & REPORT
void run_accelerator() {
    printf("--- STARTING SIMULATION (INPUT SHARING) ---\n");
    printf("Specs:\n");
    printf("  - Frequency: %.1f MHz\n", SYSTEM_FREQ_MHZ);
    printf("  - DMA Bandwidth: %d Bytes/cycle\n", DRAM_BUS_WIDTH_BYTES);
    printf("  - PE Array: %d PEs, TILE_W=%d, C_SUB=%d\n", NUM_PE, TILE_W, C_SUB);
    printf("---------------------------------------------\n");

    int num_passes = (INPUT_C + C_SUB - 1) / C_SUB;
    
    total_dma_cycles = 0;
    total_compute_cycles = 0;

    // Main Loop
    for (int ho = 0; ho < OUTPUT_H; ho++) {
        for (int wo = 0; wo < OUTPUT_W; wo += TILE_W) {
            
            int32_t final_accumulators[TILE_W] = {0};
            int actual_tile_w = (wo + TILE_W > OUTPUT_W) ? (OUTPUT_W - wo) : TILE_W;

            for (int p = 0; p < num_passes; p++) {
                
                // 1. DMA Load
                int dma_c = dma_load_buffers_input_sharing(ho, wo, p, actual_tile_w);
                total_dma_cycles += dma_c;

                // 2. Compute
                int comp_c = 0;
                int32_t pass_results[TILE_W] = {0};
                run_pe_array_input_sharing(&comp_c, pass_results, actual_tile_w);
                total_compute_cycles += comp_c;
                
                for(int i=0; i<actual_tile_w; i++) {
                    final_accumulators[i] += pass_results[i];
                }
            }

            for(int i=0; i<actual_tile_w; i++) {
                int out_idx = ho * OUTPUT_W + (wo + i);
                ofm_dram[out_idx] = final_accumulators[i];
            }
        }
    }
    
    total_cycles = total_dma_cycles + total_compute_cycles;

    // --- REPORT KẾT QUẢ ---
    double total_time_ms = (double)total_cycles / (SYSTEM_FREQ_MHZ * 1000.0);
    
    printf("\n--- PERFORMANCE REPORT ---\n");
    printf("Total Output Pixels: %d\n", OUTPUT_H * OUTPUT_W);
    printf("Total Cycles: %llu\n", total_cycles);
    printf("  - DMA Cycles:     %llu (%.2f%%)\n", total_dma_cycles, (double)total_dma_cycles/total_cycles*100.0);
    printf("  - Compute Cycles: %llu (%.2f%%)\n", total_compute_cycles, (double)total_compute_cycles/total_cycles*100.0);
    printf("Estimated Time: %.4f ms\n", total_time_ms);
    printf("--------------------------\n");
}

void write_dram_to_file() {
    FILE* f = fopen("ofm_input_sharing.txt", "w");
    if (!f) {
        printf("Could not open ofm_input_sharing.txt for writing.\n");
        return;
    }
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}

void cleanup() {
    free(ifm_dram);
    free(weight_dram);
    free(ofm_dram);
}

int main() {
    dram_init();
    run_accelerator();
    write_dram_to_file();
    cleanup();
    return 0;
}