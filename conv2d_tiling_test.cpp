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
#define PARALLEL_CHANNELS 16    // Số channel xử lý song song

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
    FILE* f_ifm = fopen("params/ifm.txt", "r");
    if(f_ifm) {
        char line[64];
        int idx = 0;
        while(fgets(line, 64, f_ifm)) ifm_dram[idx++] = (int8_t)atoi(line);
        fclose(f_ifm);
    } else {
        // Fallback nếu không có file: init random hoặc zero
        memset(ifm_dram, 1, INPUT_H * INPUT_W * INPUT_C); 
    }

    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, sizeof(int8_t));
    // Load Weights
    FILE* f_w = fopen("params/weights.txt", "r");
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
    }

    ofm_dram = (int32_t*)malloc(OUTPUT_H * OUTPUT_W * OUTPUT_F * sizeof(int32_t));
}

// 2. MÔ PHỎNG BUFFER & DMA
int8_t buffer_ifm[BUFFER_SIZE_BYTES];
int8_t buffer_weight[BUFFER_SIZE_BYTES];

// Hàm trả về số cycle tiêu tốn cho việc load DMA
int dma_load_buffers(int ho, int wo, int pass_idx) {
    // Reset buffer
    memset(buffer_ifm, 0, BUFFER_SIZE_BYTES);
    memset(buffer_weight, 0, BUFFER_SIZE_BYTES);

    int channel_start = pass_idx * PARALLEL_CHANNELS; //tinh channel bat dau chay
    int buffer_ptr = 0; 
    int bytes_transferred = 0; // Đếm số byte thực tế cần load

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break; 

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // Fetch IFM
                int hi = ho * STRIDE + kh - PADDING;
                int wi = wo * STRIDE + kw - PADDING;
                int8_t val_ifm = 0;
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    // IFM: C->W->H
                    int dram_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c;
                    val_ifm = ifm_dram[dram_idx];
                }

                // Fetch Weight
                //WEIGHTS: F->C->W->H
                int w_dram_idx = kh * (KERNEL_W * INPUT_C * OUTPUT_F) + 
                                 kw * (INPUT_C * OUTPUT_F) + 
                                 current_c * OUTPUT_F + 0;
                int8_t val_w = weight_dram[w_dram_idx];

                buffer_ifm[buffer_ptr] = val_ifm;
                buffer_weight[buffer_ptr] = val_w;
                buffer_ptr++;
                
                // Mỗi phần tử load 2 byte (1 byte IFM + 1 byte Weight)
                // Ta tính tổng bytes cần chuyển vào buffer
                bytes_transferred += 2; 
            }
        }
    }

    // --- TÍNH TOÁN LATENCY ---
    // Tổng dung lượng cần load vào Buffer: 144 bytes (cho IFM) + 144 bytes (cho Weight) = 288 bytes?
    // Code trên ghi vào 2 buffer riêng, tổng dung lượng tối đa là 144 * 2 = 288 bytes.
    // Tuy nhiên buffer_size định nghĩa là 144 bytes chia sẻ hay riêng?
    // Giả sử 2 port DMA độc lập hoặc chung bus.
    // Ta tính tổng bytes load từ DRAM chia cho Bandwidth.
    // buffer_ptr là số phần tử (cặp). Tổng bytes = buffer_ptr * 2 (1 ifm + 1 weight).
    
    int total_bytes = buffer_ptr * 2; 
    
    // Số cycle = ceil(total_bytes / bus_width)
    // + Latency khởi tạo DMA (overhead), giả sử 0 hoặc 5 cycles. Ta lấy 0 cho lý tưởng.
    int cycles = (total_bytes + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    
    return cycles;
}

// 3. MÔ PHỎNG COMPUTE ENGINE
int32_t run_pe_array(int* cycles_taken) {
    int32_t partial_sum = 0;

    // Logic tính toán chức năng (Functional)
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        int base_idx = pe_id * MACS_PER_PE; 
        int32_t pe_acc = 0; 
        for (int k = 0; k < MACS_PER_PE; k++) {//tinh het so MAC cua 1 con PE
            int8_t a = buffer_ifm[base_idx + k];
            int8_t b = buffer_weight[base_idx + k];
            pe_acc += (int32_t)a * (int32_t)b;
        }
        partial_sum += pe_acc; // tong cua 48 con PE
    }

    // --- TÍNH TOÁN LATENCY ---
    // Các PE chạy song song -> Chỉ tốn thời gian của PE chậm nhất (đều nhau).
    *cycles_taken = PE_COMPUTE_CYCLES;

    return partial_sum;
}

// 4. CONTROLLER & REPORT
void run_accelerator() {
    printf("--- STARTING SIMULATION ---\n");
    printf("Specs:\n");
    printf("  - Frequency: %.1f MHz\n", SYSTEM_FREQ_MHZ);
    printf("  - DMA Bandwidth: %d Bytes/cycle\n", DRAM_BUS_WIDTH_BYTES);
    printf("  - PE Array: %d PEs (Parallel)\n", NUM_PE);
    printf("---------------------------\n");

    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS;//de dam bao luon lam tron len
    
    // Reset Stats
    total_dma_cycles = 0;
    total_compute_cycles = 0;

    // Main Loop
    for (int ho = 0; ho < OUTPUT_H; ho++) {
        for (int wo = 0; wo < OUTPUT_W; wo++) {
            
            int32_t final_accumulator = 0; //reset accum cho moi vi tri width

            for (int p = 0; p < num_passes; p++) {
                
                // 1. DMA Load
                int dma_c = dma_load_buffers(ho, wo, p);
                total_dma_cycles += dma_c;

                // 2. Compute
                int comp_c = 0;
                int32_t pass_result = run_pe_array(&comp_c);//PE tinh toan xong gan vao pass_result
                total_compute_cycles += comp_c;
                final_accumulator += pass_result; //cong ket qua cua cac PE vao accum
            }

            int out_idx = ho * OUTPUT_W + wo; // tinh vi tri luu trong output
            ofm_dram[out_idx] = final_accumulator;
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
    FILE* f = fopen("ofm/ofm.txt", "w");
    if (!f) return;
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