# Conv2D Optimization: Tiling, Weight Stationary, and Input Sharing

C++ implementation comparing three hardware-efficient Conv2D computation strategies for neural network accelerators.

## 🎯 Overview

This project implements and analyzes three Conv2D optimization techniques:

1. **Tiling Strategy**: Decompose input into tiles, reload weights per position (simple but high DRAM traffic)
2. **Weight Stationary**: Load weights once, reuse across 12,544 output positions (12,544× weight reuse)
3. **Input Sharing**: Process 4 pixels simultaneously with spatial overlap (~25% bandwidth savings)

**Problem Setup:**
- **IFM**: 112×112×32 | **Kernel**: 3×3×32 | **OFM**: 112×112×1
- **Stride**: 1, **Padding**: 1
- **Hardware**: 48 PE × 3 MACs/cycle = 144 MACs/cycle @ 100 MHz
- **Buffer**: 144 bytes on-chip

---

## 🔧 How It Works

**Conv2D Equation:**
$$\text{OFM}[h,w,f] = \sum_{kh,kw,c} \text{IFM}[h+kh, w+kw, c] \times \text{Weight}[kh,kw,c,f]$$

**Memory Hierarchy:**
- DRAM (large, slow) → Local Buffer (144 bytes, fast) → PE Array (48 units, compute)

---

## 📊 Three Implementation Methods

### 1. Tiling Strategy (`conv2d_tiling_test.cpp`)
**Concept**: Load tile per output position, reload weights each time  
**Config**: PARALLEL_CHANNELS = 16  
**Loop**: `for(ho, wo, channel_pass)`  
**Pros**: Simple control  
**Cons**: High DRAM traffic, weights reloaded 12,544 times

```
for each output (ho, wo):
    for each channel pass:
        Load IFM → Load Weights → Compute → Store
```

---

### 2. Weight Stationary (`conv2d_weight_stationary_test.cpp`)
**Concept**: Keep weights in buffer, stream different IFM tiles  
**Config**: C_PARALLEL = 16 (2 channel groups)  
**Loop**: `for(c_group) → for(ho, wo)`  
**Weight Reuse**: **12,544×** (loaded 2 times total!)  
**Pros**: ~88% reduction in DRAM traffic, energy-efficient  
**Cons**: More complex control

```
for each channel group:
    Load Weights (STATIONARY)
    for each output (ho, wo):
        Load IFM → Compute (with stationary weights) → Store
```

---

### 3. Input Sharing (`conv2d_input_sharing_test.cpp`)
**Concept**: Process 4 pixels simultaneously with spatial overlap  
**Config**: TILE_W = 4, C_SUB = 4  
**Loop**: `for(c_sub) → for(ho) → for(wo_tile)`  
**Overlap Savings**: ~25% fewer DRAM loads  
**Pros**: High data reuse, spatial parallelism  
**Cons**: Complex address handling

```
for each channel group:
    for each output row (ho):
        for each 4-pixel tile (wo_tile):
            Load IFM patch (with overlap) → Compute → Store
```

---

## 📁 Project Structure

```
├── conv2d_tiling_test.cpp                 # Tiling approach
├── conv2d_weight_stationary_test.cpp      # Weight stationary  
├── conv2d_input_sharing_test.cpp          # Input sharing
├── 2_sosanhxuatpho.py                     # Compare & visualize results
├── params/
│   ├── ifm.txt                            # Input (112×112×32)
│   └── weights.txt                        # Kernel (3×3×32)
├── golden_output/
│   └── ofm_golden.txt                     # Reference output
└── output_results/
    ├── ofm_tiling.txt
    ├── ofm_weight_stationary.txt
    └── ofm_input_sharing.txt
```

**Data Format:**
- IFM, Weights: int8 per line, one value per line
- Output: int32 per line, 12,544 total lines

---

## 🚀 Quick Start

**Compile:**
```bash
gcc -O3 -o conv2d_tiling conv2d_tiling_test.cpp -lm
gcc -O3 -o conv2d_ws conv2d_weight_stationary_test.cpp -lm
gcc -O3 -o conv2d_is conv2d_input_sharing_test.cpp -lm
```

**Run:**
```bash
./conv2d_tiling    # Outputs to ofm_tiling.txt
./conv2d_ws        # Outputs to ofm_weight_stationary.txt
./conv2d_is        # Outputs to ofm_input_sharing.txt

python 2_sosanhxuatpho.py   # Compare results
```

**Important**: Update file paths in C++ if data location differs:
```cpp
FILE* f_ifm = fopen("params/ifm.txt", "r");    // Adjust if needed
FILE* f_w = fopen("params/weights.txt", "r");
```

---

## � Performance Comparison

| Method | DRAM Bandwidth | Weight Reuse | Iterations | Complexity |
|--------|---|---|---|---|
| **Tiling** | High | 1× | 12,544 × 2 | Low |
| **Weight Stationary** | Low (~88% ↓) | **12,544×** | 2 × 12,544 | Medium |
| **Input Sharing** | Medium (~25% ↓) | 4× | 8 × 112 × 28 | High |

### Key Insights

1. **Weight Stationary Dominates**: 12,544× weight reuse = massive energy savings
   - Weights: 200 pJ/byte @ 28nm
   - Tiling: 12,544 × 9B × 200pJ ≈ 22.6 nJ/output
   - Weight Stationary: 1 × 9B × 200pJ ≈ 1.8 pJ/output
   - **Energy saving: ~12,000×**

2. **Input Sharing Tradeoff**: 
   - Saves ~25% DRAM but adds control overhead
   - Best for larger kernels (5×5, 7×7)
   - Control overhead limits gains for 3×3

3. **Bandwidth Limited**:
   - Peak: 14.4 GOPs @ 100MHz (144 MACs/cycle)
   - Realistic: ~2-3 GOPs due to DRAM bandwidth constraints

---

## 💾 Hardware Specification

| Parameter | Value | Note |
|-----------|-------|------|
| PE Count | 48 | Processing elements |
| MACs/PE/Cycle | 3 | Multiply-accumulate ops |
| **Peak MACs/Cycle** | **144** | 48 × 3 |
| Buffer | 144 bytes | On-chip SRAM |
| Clock | 100 MHz | System frequency |
| DRAM Bandwidth | 8 bytes/cycle | 64-bit bus |
| **Peak Throughput** | **14.4 GOPs** | 144 MACs × 100 MHz |

---

## 🔍 Understanding the Code

Each implementation has these main components:

1. **`dram_init()`** - Load IFM from file, allocate buffers
2. **`dma_load_buffers*()`** - Simulate data transfer (counts cycles)
3. **Main Loop** - Different for each method (see above)
4. **Compute** - MAC operations (1 cycle for 144 MACs)
5. **Output Storage** - Write results to file

The key difference: **how and when weights are loaded from DRAM**

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `File not found` | Check `params/ifm.txt` and `params/weights.txt` exist |
| `Compilation error` | Install GCC/MinGW, try: `gcc --version` |
| Wrong output | Verify paths in C++ code, check data types (int8, int32) |
| Huge differences in results | Check for off-by-one errors, padding handling |

---

## 📚 Related Work

- **Eyeriss** (MIT): Weight-stationary dataflow
- **DianNao** (INRIA): Tiling-based accelerator  
- **TPU** (Google): Systolic array with input sharing
- **Gemmini** (UC Berkeley): Flexible dataflow
