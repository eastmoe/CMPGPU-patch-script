# CMP CUDA 优化补丁脚本 for llama.cpp

这是一个为 **NVIDIA 170HX 等特定 CUDA 设备** 设计的实验性补丁脚本，通过强制使用 CUDA Core 并避免 Tensor Core，在某些模型和量化下获得显著的性能提升。

## 用法

1. **下载 llama.cpp 源码**  
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   ```

2. **将本脚本（`optimize-cmp-cuda.py`）放入源码目录（或任意位置）**

3. **运行脚本，交互式选择补丁组**  
   ```bash
   python optimize-cmp-cuda.py
   ```
   脚本会询问你要启用哪些优化（默认均为 `n`）：
   - 禁用 FP32 FMA 收缩（添加 `-fmad=false`）
   - 拆分显式 FP32 `fmaf` 调用
   - 替换数学函数为内联指令实现（`tanhf`, `log1pf`, `erff` 等）
   - 将 DP4A 替换为 DP2A（两个 `dp2a` 指令）
   - 将 BF16/FP16 线性层从 Tensor Core 迁移到 CUDA Core

   按需回答 `y` 或 `n`。

4. **正常编译 llama.cpp**  
   ```bash
   mkdir build && cd build
   cmake .. -DGGML_CUDA=ON
   cmake --build . --config Release
   ```

5. **回退所有修改**  
   ```bash
   python optimize-cmp-cuda.py --restore
   ```
   脚本会从 `.cmp-bak` 备份文件恢复所有被修改的源文件。

---

## 修改内容概述

| 补丁组 | 作用 |
|--------|------|
| `fp32_fma_flag` | 在 CUDA 编译参数中添加 `-fmad=false`，防止自动 FMA 合并 |
| `fp32_fma_split` | 将源码中的 `fmaf` 手动拆分为 `__fmul_rn` + `__fadd_rn` |
| `math_intrinsics` | 用内联近似函数替换 `tanhf`, `log1pf`, `erff`，提高吞吐 |
| `dp2a` | 将 `__dp4a` 替换为两条 `dp2a` 指令（适用于支持 PTX 的设备） |
| `fp16_bf16_cuda_core` | 强制 FP16/BF16 矩阵乘法使用 CUDA Core（而非 Tensor Core），调整 MMA 可用性判断和几何计算路径 |

这些修改共同减少了 Tensor Core 依赖，同时优化了 CUDA Core 上的指令调度，在 **170HX (compute capability 8.0)** 等设备上表现突出。

---

## 性能对比（实测）

**测试环境**  
- GPU：NVIDIA 170HX (8GB VRAM, CC 8.0)  
- 模型：Gemma-4-12B-it-AEON-Abliterated-K4-BF16.i1-IQ4_XS (6.16 GiB, 4.25 bpw)  
- 测试工具：`llama-bench`，`-ngl -1`

| 阶段 | 修改前 (t/s) | 修改后 (t/s) | 提升 |
|------|-------------|-------------|------|
| pp512 | 355.59 ± 0.98 | **730.84 ± 4.47** | **+105%** |
| tg128 | 34.54 ± 0.02 | **63.34 ± 0.24** | **+83%** |

> **结论**：在特定组合下，补丁可将 Prompt Processing 速度提升一倍以上，Token Generation 速度提升近一倍。

---

## 注意事项

- ⚠️ 本补丁**专为实验性本地构建设计**，可能不适用于所有 GPU 架构。  
- 在非 170HX 或其他 CC 8.x 设备上，性能可能**下降**，请先备份源码（脚本自动生成 `.cmp-bak`）。  
- 若编译失败，使用 `--restore` 回退即可。  
- 你可以使用 `--dry-run` 预览修改内容而不实际写文件。  
- 使用 `--no-backup` 可跳过备份（不推荐）。
