# High-Performance Matrix Transposition with CUDA

This project demonstrates the implementation and optimization of matrix transposition using NVIDIA CUDA. By transitioning from a **Naive** approach to an **Optimized** version using **Shared Memory** and **Unified Memory**, we achieved performance gains of up to **230x** on modern GPU hardware[cite: 2].

## 🚀 Performance Comparison (RTX 3060 Ti)

The following table summarizes the execution times and speedup ratios across different matrix sizes and tile dimensions:

| Matrix Size | TILE_DIM | Naive Time (ms) | Optimized Time (ms) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 2048 x 1024 | 32 | 27.197 ms | 0.167 ms | 162.85x |
| 4096 x 2048 | 32 | 23.614 ms | 0.539 ms | 43.78x |
| 8192 x 4096 | 32 | 93.816 ms | 2.028 ms | 46.27x |
| 2048 x 1024 | 16 | 27.890 ms | 0.121 ms | **230.81x** |

---

## 💡 Key Optimization Analysis

### 1. Memory Coalescing & Shared Memory
In the **Naive Kernel**, the write process to global memory is "strided" (non-sequential), which forces the hardware to issue multiple memory transactions for a single warp, leading to massive latency[cite: 2]. 
The **Optimized Kernel** solves this by using **Shared Memory** as a staging area. We load data into a shared tile in a coalesced fashion, transpose it locally, and write it back to global memory sequentially, ensuring maximum bandwidth utilization[cite: 2].

### 2. Impact of Tile Dimensions (TILE_DIM)
Our experiments show that decreasing `TILE_DIM` from 32 to 16 for a 2048x1024 matrix boosted the speedup from **162x to 230x**[cite: 2]. This indicates that smaller tiles may reduce **Shared Memory Bank Conflicts** and improve **L1 Cache hit rates** on the Ampere architecture (RTX 3060 Ti)[cite: 2].

### 3. Unified Memory & Synchronization
- **Unified Memory:** Utilizing `cudaMallocManaged` simplified memory management and allowed the CUDA driver to optimize data migration between the CPU and GPU[cite: 2].
- **Synchronization:** The `__syncthreads()` barrier was essential to prevent **Race Conditions**, ensuring all threads finished writing to the shared tile before any thread attempted to read from it[cite: 2].

---

## 🛠️ Requirements & Compilation

- **Hardware:** NVIDIA GPU (Tested on RTX 3060 Ti)
- **Compiler:** NVCC (CUDA Toolkit)

### How to Run:
```bash
nvcc -o transpose transpose.cu
transpose
