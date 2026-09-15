# wrf_pinn read speedup: plan

Branch: `speedup/pinn-read`. Goal: cut the per-case training-side read cost
(`.npy` -> `Case`), reported at ~11.9 s/case last week. The hypothesis to test:
**that number is not purely disk I/O** -- a large share is CPU work inside
`read_case`, done redundantly over ~4.7M rows/case.

## What "the read" actually is
`read_case` in [src/wrf_pinn/data/case.py](src/wrf_pinn/data/case.py) does, per case:
1. `np.load(path)` -- disk (float64 array, ~360 MB/case).
2. `data[:, coord_cols]` + `ascontiguousarray(..., float32)` -- fancy-index copy + f64->f32 cast.
3. same for targets.
4. `source.astype(int64)` -- another cast.
5. `_check_finite(coords)` -- full `isfinite(...).all()` scan.
6. `isfinite(targets)` -- second full scan (the mask).
7. `where(mask>0, targets, 0)` -- third pass + fresh allocation.

Steps 2-7 are ~6 full passes/allocations over millions of rows. The old
`perf_read_lasso.py` stage-2 timer wraps the whole thing, so all of that CPU
cost is being reported as "read." `profile_pinn_read.py` already splits it into
`np_load` / `case_build` / `h2d_transfer` -- but it calls the pre-rewrite
`orchestrator.run(spill_dir=...)` signature and won't run as-is.

## Step 0 -- measure first (no changes)
Guessing bottlenecks has burned us before; establish the split before optimizing.
- Fix `profile_pinn_read.py` to the new preprocessor API (folder-per-case,
  `work_dir=`), or write a tiny standalone splitter that reads existing `.npy`
  cases and times: `np_load` vs `case_build` (steps 2-7) vs `h2d_transfer`.
- Report ms/case for each on the real grid (ECC), ~20-50 cases.
- **Decision gate:** if `np_load` dominates -> it *is* I/O, stop here and pursue
  dtype/on-disk format. If `case_build` is a large slice -> the CPU-side wins below
  are real. Likely both; the split tells us where to spend effort.

## Candidate optimizations (ranked, pending Step 0)
All are wrf_pinn-side only. On-disk dtype is deliberately out of scope (a larger
cross-repo change to raise with the group separately).

1. **Offload the case-build math to the GPU (biggest lever here).** Steps 2-7 are
   exactly the vectorized array work that ran ~78x faster on the H100 in the
   preprocessor. The data ends up on the GPU anyway (`as_torch(device=...)`,
   `train_pinn.py:177`). So load raw, move the raw array to the GPU once, and do
   the slice / cast / finite-check / mask / zero-fill there as tensor ops, instead
   of ~6 CPU passes on the host and then a transfer. Turns host CPU cost into
   near-free device cost and folds the h2d transfer into the same move.
2. **Collapse the redundant passes.** `_check_finite` scans coords, then the mask
   scans targets, then `where` re-touches targets. Compute the target mask once
   and reuse; coords are guaranteed finite by the preprocessor's required-column
   drop, so the `_check_finite` full scan can become a cheap assertion (or gate it
   behind a debug flag). Fewer passes, fewer allocations. Helps the CPU path and
   the GPU path (fewer kernels).
3. **Avoid the fancy-index copies.** `data[:, coord_cols]` gathers non-contiguous
   columns. If coords/targets are contiguous column ranges in the schema
   (`x,y,z,t` then `u,v,w`), a slice view + single cast beats gather+copy.
4. **`np.load(mmap_mode="r")`** so the OS pages only the columns touched -- helps
   only if we stop materializing the whole array on the host; evaluate after (1).
5. **Cache the built tensors.** Training reads a case once then reuses
   (`train_pinn.py:177`), so within a run this is paid once. If cases are re-read
   across runs, a `.pt` cache skips build entirely. Only worth it if the workflow
   re-reads.

## Constraints / contract to preserve
- Output `Case` must be byte-identical in behavior: coords float32 & finite,
  targets float32 with NaN zero-filled, `target_mask` float32 marking measured
  entries, `source` int64. Tests in
  [tests/test_pipeline.py](tests/test_pipeline.py) encode this -- keep them green.
- `train_fasteddy_lcc.py` asserts coords/targets in [0,1] and `minmax_01`; don't
  change normalization semantics.
- Only operate on the `speedup/pinn-read` branch. User runs all commits + cluster
  syncs. Measure on ECC (real grid) before claiming any speedup.

## Deliverable shape
Step 0 profile -> pick from (1)-(5) by evidence -> implement smallest set that
moves the number -> re-measure on ECC -> one slide with before/after split.
