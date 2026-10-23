# Sparse feature byte addressing fix

Sparse convolution previously rejected feature tensors whose byte size reaches signed int32 limits. Keep sparse row IDs and pair tables int32, but compute feature byte offsets in int64 before multiplying by channels and element size. Both sparse iterator variants and the AOT/NVRTC dispatch guards are covered.

This is not general int64 indexing: pair-table size must remain below INT_MAX, and extreme channel strides/reset increments still have int32 limits. Validation covers FP16 and ordinary channel counts (up to 512), not every dtype, algorithm or dimension. Keep existing precision settings.

## Rebuild both extensions

Build and install the patched cumm first, then rebuild spconv against that cumm in a clean, isolated environment. The affected kernels are generated from cumm into spconv/core_cc; replacing cumm Python files alone does not fix an existing spconv binary. Rebuild cumm/core_cc as well to avoid descriptor ABI mismatch. Follow each repository's source-build instructions and use matching CUDA, Python, compiler and architecture settings. Do not mix locally built extensions with incompatible wheel versions.

## Validation evidence and limits

The isolated validation used cumm 0.7.11, spconv 2.3.8, Python 3.11, CUDA 11.8 and an RTX A6000 (SM86). The three modified cumm source files in this branch match the v0.7.11 originals before patching; this does not establish that all other current-branch code has been rebuilt or validated.

Passed: 162 small dense-reference forward/backward cases, 36 inference cases, 19 large functional cases (including >2 GiB, >4 GiB, all 27 kernel taps, and forced NVRTC forward/dgrad/wgrad), two seeds of 12 optimizer updates with FP32 master weights, Compute Sanitizer memcheck on small and >2 GiB cases, and small-case racecheck. Three stock above-boundary controls failed as expected. Normal FP16 numerical differences are accepted; no bitwise equivalence is required.

SM80 was compiled but A800 runtime has not been tested. Python 3.10 needs a separate build. These results are not evidence for arbitrary dtype/shape or current GitHub HEAD binaries. Initial total forward/backward benchmark ratios ranged from 0.945 to 1.042; one small-case forward alone slowed by about 20%, so no blanket performance guarantee is claimed.

## Opt-in boundary regression

After building both extensions, run from the repository root using the intended Python environment. These tests allocate multiple GiB and should run on a free GPU with ample memory (the validated device has 48 GiB):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_feature_address_boundary.py --ci 512 --co 16 --pairs
CUDA_VISIBLE_DEVICES=0 python tools/check_feature_address_boundary.py --ci 16 --co 512 --pairs
CUDA_VISIBLE_DEVICES=0 python tools/check_feature_address_boundary.py --ci 512 --co 16 --n 4194336 --pairs
```

The script uses independent analytic references, checks forward/input-gradient/weight-gradient, and asserts per-row errors around 2/4 GiB address boundaries and at the tail. Probe rows receive nonzero upstream gradients. Relative-L2 tolerance is 2%, allowing normal FP16 accumulation differences while checking addressing. Device selection and package installation are controlled externally; this is not an automatic CI test.

Companion cumm fork: https://github.com/ZoneLikeWonderland/cumm (branch `fix/sparse-feature-int64-offsets`). No spconv kernel source change is needed; rebuilding generates the corrected kernels.

The three additional per-row boundary checks passed in the pinned validation environment. Maximum per-probe relative L2 was 0.002426 (0.243%). Numeric results are in `sparse_feature_int64_boundary_results.json`.
