# CMP CUDA Optimization Patch Script for llama.cpp

This is an experimental patch script designed for **specific CUDA devices like NVIDIA 170HX**. It forces the use of CUDA Cores while avoiding Tensor Cores, yielding significant performance gains on certain models and quantizations.

---

## Usage

1. **Clone the llama.cpp source code**  
   ```bash
   git clone https://github.com/ggml-org/llama.cpp/
   cd llama.cpp
   ```

2. **Place this script (`optimize-cmp-cuda.py`) in the source directory (or anywhere else)**

3. **Run the script and interactively select patch groups**  
   ```bash
   python optimize-cmp-cuda.py
   ```
   The script will ask you which optimizations to enable (default is `n` for all):
   - Disable FP32 FMA contraction (add `-fmad=false`)
   - Split explicit FP32 `fmaf` calls
   Replace math functions with inline instruction approximations (`tanhf`, `log1pf`, `erff`, etc.)
   - Replace DP4A with DP2A (two `dp2a` instructions)
   - Move BF16/FP16 linear layers from Tensor Cores to CUDA Cores

   Answer `y` or `n` as needed.

4. **Build llama.cpp as usual**  
   ```bash
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release
   ```

5. **Revert all changes(If you need)**  
   ```bash
   python optimize-cmp-cuda.py --restore
   ```
   The script will restore all modified source files from the `.cmp-bak` backup files.

---

## Changes Overview

| Patch Group | Effect |
|-------------|--------|
| `fp32_fma_flag` | Add `-fmad=false` to CUDA compilation flags to prevent automatic FMA merging |
| `fp32_fma_split` | Manually split `fmaf` calls into `__fmul_rn` + `__fadd_rn` in source |
| `math_intrinsics` | Replace `tanhf`, `log1pf`, `erff` with inline approximate functions for higher throughput |
| `dp2a` | Replace `__dp4a` with two `dp2a` instructions (for devices supporting PTX) |
| `fp16_bf16_cuda_core` | Force FP16/BF16 matrix multiplications to use CUDA Cores (instead of Tensor Cores), adjusting MMA availability checks and arithmetic paths |

Together, these changes reduce Tensor Core dependency while optimising instruction scheduling on CUDA Cores, showing remarkable improvements on **170HX (compute capability 8.0)** and similar devices.

---

## Performance Comparison (Measured)

**Test Environment**  
- GPU: NVIDIA 170HX (8GB VRAM, CC 8.0)  
- Model: Gemma-4-12B-it-AEON-Abliterated-K4-BF16.i1-IQ4_XS (6.16 GiB, 4.25 bpw)  
- Tool: `llama-bench`, `-ngl -1`

| Stage | Before (t/s) | After (t/s) | Improvement |
|-------|-------------|-------------|-------------|
| pp512 | 355.59 ± 0.98 | **730.84 ± 4.47** | **+105%** |
| tg128 | 34.54 ± 0.02 | **63.34 ± 0.24** | **+83%** |

> **Conclusion**: With this specific combination, prompt processing speed more than doubles, and token generation speed nearly doubles.

---

## Important Notes

- ⚠️ This patch is **designed for experimental local builds only** and may not work well on all GPU architectures.
- On non‑170HX or other CC 8.x devices, performance may **decrease** – always keep a backup (the script automatically creates `.cmp-bak` files).
- If compilation fails, simply run `--restore` to revert.
- Use `--dry-run` to preview changes without writing files.
- Use `--no-backup` to skip backups (not recommended).
