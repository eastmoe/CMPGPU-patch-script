#!/usr/bin/env python3
"""
Patch llama.cpp CUDA sources for CMP-oriented local builds.

This script is intended for local experiments. It can add -fmad=false to the
CUDA NVCC flags, split explicit FP32 FMA calls, replace selected math functions
with unrestricted CUDA instructions, move selected FP16/BF16 linear-layer paths
away from Tensor Cores, and rewrite the CUDA ggml_cuda_dp4a wrapper to use two
dp2a instructions.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path


CMAKE_REL = Path("ggml/src/ggml-cuda/CMakeLists.txt")
COMMON_REL = Path("ggml/src/ggml-cuda/common.cuh")
CUDA_REL = Path("ggml/src/ggml-cuda/ggml-cuda.cu")
MMF_REL = Path("ggml/src/ggml-cuda/mmf.cu")
MMVF_REL = Path("ggml/src/ggml-cuda/mmvf.cu")
MMQ_REL = Path("ggml/src/ggml-cuda/mmq.cuh")
QUANTIZE_REL = Path("ggml/src/ggml-cuda/quantize.cu")
BACKUP_SUFFIX = ".cmp-bak"

CMAKE_OLD = "    set(CUDA_FLAGS -use_fast_math -extended-lambda)"
CMAKE_NEW = "    set(CUDA_FLAGS -use_fast_math -extended-lambda -fmad=false)"

CMP_FAST_MATH_ANCHOR = """#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
"""

CMP_FAST_MATH_PREVIOUS = """#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__)
#define expf  __expf
#define logf  __logf
#define powf  __powf
#define sinf  __sinf
#define cosf  __cosf
#define sqrtf __fsqrt_rn
#endif

#define WARP_SIZE 32
"""

CMP_FAST_MATH_NEW = """#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__)
static __device__ __forceinline__ float ggml_cuda_cmp_tanhf(float x) {
    if (x > 10.0f) {
        return 1.0f;
    }
    if (x < -10.0f) {
        return -1.0f;
    }

    const float e = __expf(__fmul_rn(2.0f, x));
    return __fdividef(__fadd_rn(e, -1.0f), __fadd_rn(e, 1.0f));
}

static __device__ __forceinline__ float ggml_cuda_cmp_log1pf(float x) {
    return __logf(__fadd_rn(1.0f, x));
}

static __device__ __forceinline__ float ggml_cuda_cmp_erff(float x) {
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p  = 0.3275911f;

    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax   = x < 0.0f ? -x : x;
    const float t    = __fdividef(1.0f, __fadd_rn(1.0f, __fmul_rn(p, ax)));

    float poly = __fadd_rn(__fmul_rn(a5, t), a4);
    poly = __fadd_rn(__fmul_rn(poly, t), a3);
    poly = __fadd_rn(__fmul_rn(poly, t), a2);
    poly = __fadd_rn(__fmul_rn(poly, t), a1);
    poly = __fmul_rn(poly, t);

    const float e = __expf(-__fmul_rn(ax, ax));
    return __fmul_rn(sign, __fsub_rn(1.0f, __fmul_rn(poly, e)));
}

#define expf   __expf
#define logf   __logf
#define powf   __powf
#define sinf   __sinf
#define cosf   __cosf
#define sqrtf  __fsqrt_rn
#define tanhf  ggml_cuda_cmp_tanhf
#define log1pf ggml_cuda_cmp_log1pf
#define erff   ggml_cuda_cmp_erff
#endif

#define WARP_SIZE 32
"""

EXPLICIT_FMA_OLD = "            cur_err = fmaf(err_diff, err_diff, cur_err);"
EXPLICIT_FMA_NEW = "            cur_err = __fadd_rn(__fmul_rn(err_diff, err_diff), cur_err);"

CUBLAS_MATH_OLD = "CUBLAS_TF32_TENSOR_OP_MATH"
CUBLAS_MATH_NEW = "CUBLAS_DEFAULT_MATH"

CUBLAS_GEMM_ALGO_OLD = "CUBLAS_GEMM_DEFAULT_TENSOR_OP"
CUBLAS_GEMM_ALGO_NEW = "CUBLAS_GEMM_DEFAULT"

DP4A_OLD = """#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
"""

DP4A_PREVIOUS = """#if !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;

    const short2 a_lo = make_short2(a8[0], a8[1]);
    const short2 a_hi = make_short2(a8[2], a8[3]);
    const char4  b4   = make_char4(b8[0], b8[1], b8[2], b8[3]);

    return __dp2a_hi(a_hi, b4, __dp2a_lo(a_lo, b4, c));
#elif defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
"""

MMA_AVAIL_OLD = """static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

static bool bf16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) ||
        GGML_CUDA_CC_IS_CDNA(cc) || cc >= GGML_CUDA_CC_RDNA3 ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}
"""

MMA_AVAIL_NEW = """static bool fp16_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

static bool bf16_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc) || cc >= GGML_CUDA_CC_RDNA3 ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}
"""

HALF2_MAD_OLD = """static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}
"""

HALF2_MAD_NEW = """static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc = __hfma2(v, u, acc);
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}
"""

MMF_SWITCH_OLD = """        case GGML_TYPE_F32:
            return ampere_mma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_BF16:
            return ampere_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
"""

MMF_SWITCH_NEW = """        case GGML_TYPE_F32:
            return ampere_mma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_F16:
            return !GGML_CUDA_CC_IS_NVIDIA(cc) && (amd_wmma_available(cc) || amd_mfma_available(cc));
        case GGML_TYPE_BF16:
            return !GGML_CUDA_CC_IS_NVIDIA(cc) && (amd_wmma_available(cc) || amd_mfma_available(cc));
"""

MMVF_HALF2_OLD = """                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
"""

MMVF_HALF2_NEW = """                    const half2 tmpy_h2 = make_half2(tmpy.x, tmpy.y);
                    sumh2[j] = __hfma2(tmpx, tmpy_h2, sumh2[j]);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] = __hfma2(tmpx_gate, tmpy_h2, sumh2_gate[j]);
                        }
                    }
"""

MMQ_MMA_SELECT_OLD = """#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<type, mmq_x, mmq_y, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, need_check>;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
"""

MMQ_MMA_SELECT_NEW = """#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<type, mmq_x, mmq_y, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, need_check>;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
"""


@dataclass(frozen=True)
class PatchSpec:
    key: str
    path_rel: Path
    old: str | tuple[str, ...]
    new: str
    expected_count: int = 1


DP4A_NEW = """#if !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    int a_lo;
    int a_hi;
    asm volatile("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(a_lo) : "r"(a));
    asm volatile("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(a_hi) : "r"(a));

    int r = c;
    asm volatile("dp2a.lo.s32.s32 %0, %1, %2, %0;" : "+r"(r) : "r"(a_lo), "r"(b));
    asm volatile("dp2a.hi.s32.s32 %0, %1, %2, %0;" : "+r"(r) : "r"(a_hi), "r"(b));
    return r;
#elif defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
"""

PATCHES = (
    PatchSpec("fp32_fma_flag", CMAKE_REL, CMAKE_OLD, CMAKE_NEW),
    PatchSpec("fp32_fma_split", QUANTIZE_REL, EXPLICIT_FMA_OLD, EXPLICIT_FMA_NEW),
    PatchSpec("math_intrinsics", COMMON_REL, (CMP_FAST_MATH_ANCHOR, CMP_FAST_MATH_PREVIOUS), CMP_FAST_MATH_NEW),
    PatchSpec("dp2a", COMMON_REL, (DP4A_OLD, DP4A_PREVIOUS), DP4A_NEW),
    PatchSpec("fp16_bf16_cuda_core", COMMON_REL, CUBLAS_MATH_OLD, CUBLAS_MATH_NEW),
    PatchSpec("fp16_bf16_cuda_core", COMMON_REL, MMA_AVAIL_OLD, MMA_AVAIL_NEW),
    PatchSpec("fp16_bf16_cuda_core", COMMON_REL, HALF2_MAD_OLD, HALF2_MAD_NEW),
    PatchSpec("fp16_bf16_cuda_core", CUDA_REL, CUBLAS_GEMM_ALGO_OLD, CUBLAS_GEMM_ALGO_NEW, expected_count=5),
    PatchSpec("fp16_bf16_cuda_core", MMF_REL, MMF_SWITCH_OLD, MMF_SWITCH_NEW),
    PatchSpec("fp16_bf16_cuda_core", MMVF_REL, MMVF_HALF2_OLD, MMVF_HALF2_NEW),
    PatchSpec("fp16_bf16_cuda_core", MMQ_REL, MMQ_MMA_SELECT_OLD, MMQ_MMA_SELECT_NEW),
)

PATCH_PATHS = tuple(dict.fromkeys(spec.path_rel for spec in PATCHES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch llama.cpp CUDA sources for CMP-oriented local builds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the changes without writing files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=f"Do not create {BACKUP_SUFFIX} backup files before patching.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help=f"Restore files from {BACKUP_SUFFIX} backups.",
    )
    return parser.parse_args()


def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        try:
            answer = input(f"{question} [{suffix}]: ").strip().lower()
        except EOFError as exc:
            raise SystemExit(f"Missing interactive answer for: {question}") from exc
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def select_patch_specs() -> list[PatchSpec]:
    selected_keys: set[str] = set()

    if prompt_yes_no("Disable FP32 FMA contraction by adding -fmad=false?", default=False):
        selected_keys.add("fp32_fma_flag")
    if prompt_yes_no("Split explicit FP32 fmaf calls into __fmul_rn + __fadd_rn?", default=False):
        selected_keys.add("fp32_fma_split")
    if prompt_yes_no("Replace selected math functions with unrestricted CUDA instruction helpers?", default=False):
        selected_keys.add("math_intrinsics")
    if prompt_yes_no("Replace DP4A with DP2A?", default=False):
        selected_keys.add("dp2a")
    if prompt_yes_no("Move BF16/FP16 FMA paths from Tensor Core to CUDA Core?", default=False):
        selected_keys.add("fp16_bf16_cuda_core")

    return [spec for spec in PATCHES if spec.key in selected_keys]


def resolve_source_dir(source_dir: str | None) -> Path:
    if not source_dir:
        try:
            source_dir = input("llama.cpp source directory: ").strip().strip('"')
        except EOFError as exc:
            raise SystemExit("Missing llama.cpp source directory.") from exc

    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Source directory does not exist: {root}")
    if any(not (root / path_rel).is_file() for path_rel in PATCH_PATHS):
        raise SystemExit(f"This does not look like a llama.cpp source directory: {root}")
    return root


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def backup(path: Path, no_backup: bool) -> None:
    if no_backup:
        return
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def unified_diff(path: Path, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def replace_exact(
    text: str,
    old: str | tuple[str, ...],
    new: str,
    path: Path,
    expected_count: int = 1,
) -> tuple[str, bool]:
    old_blocks = (old,) if isinstance(old, str) else old
    matches = [(old_block, text.count(old_block)) for old_block in old_blocks]
    total = sum(count for _, count in matches)

    if total == 0:
        if text.count(new) >= expected_count:
            return text, False
        detail = ", ".join(str(count) for _, count in matches)
        raise SystemExit(f"Expected {expected_count} known match(es) in {path}, found [{detail}].")

    if total != expected_count:
        detail = ", ".join(str(count) for _, count in matches)
        raise SystemExit(f"Expected {expected_count} known match(es) in {path}, found [{detail}].")

    changed = False
    for old_block, count in matches:
        if count > 0:
            text = text.replace(old_block, new)
            changed = True

    return text, changed


def patch_file(spec: PatchSpec, root: Path, dry_run: bool, no_backup: bool) -> bool:
    path = root / spec.path_rel
    before = read_text(path)
    after, changed = replace_exact(before, spec.old, spec.new, path, spec.expected_count)
    if not changed:
        print(f"Already patched: {path}")
        return False

    if dry_run:
        print(unified_diff(path, before, after))
        return True

    backup(path, no_backup)
    write_text(path, after)
    print(f"Patched: {path}")
    return True


def restore_file(path: Path, dry_run: bool) -> bool:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        print(f"No backup found: {backup_path}")
        return False

    if dry_run:
        current = read_text(path)
        original = read_text(backup_path)
        print(unified_diff(path, current, original))
        return True

    shutil.copy2(backup_path, path)
    print(f"Restored: {path}")
    return True


def main() -> int:
    args = parse_args()
    root = resolve_source_dir(None)

    if args.restore:
        changed = [restore_file(root / path_rel, args.dry_run) for path_rel in PATCH_PATHS]
    else:
        selected_specs = select_patch_specs()
        if not selected_specs:
            print("No patch groups selected.")
            return 0
        changed = [patch_file(spec, root, args.dry_run, args.no_backup) for spec in selected_specs]

    if args.dry_run:
        print("Dry run complete.")
    elif args.restore and any(changed):
        print("CMP CUDA patch restore complete.")
    elif args.restore:
        print("No backup files were restored.")
    elif any(changed):
        print("CMP CUDA patch complete. You can now configure and build llama.cpp normally.")
    else:
        print("No changes needed. CMP CUDA patch is already applied.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
