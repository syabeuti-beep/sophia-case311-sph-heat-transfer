# Verification report

## Completed checks

1. PDF extraction was performed with PyMuPDF.
2. Case 3-1-1 conditions were extracted from the paper text and Figure/Table area.
3. SOPHIA code locations were inspected:
   - `function_SPH_DEM_COUPLING.cuh`
   - `function_TIME_ISPH.cuh`
   - `function_OUTPUT.cuh`
   - `ISPH.cuh`
   - `function_PROP.cuh`
   - `Parameters.cuh`
4. A separate case workspace was created; the original SOPHIA source was not modified.
5. Full input generation succeeded.
6. Input sanity validation passed.
7. Basic static delimiter-balance checks passed for modified `.cuh` files.
8. `make -n` confirmed the compile command.
9. Input particle plots were generated and inspected visually.
10. `input.txt` was changed back to the original SOPHIA-friendly format: one particle per tab-separated line, plus one header line.
11. `gpu_count_particle_numbers2` was restored to count particles as `line_count - 1`, and probed with the generated input: 360,001 total lines -> 360,000 particles.
12. VTK output interval was changed to every 100 steps. With `dt = 1.0e-4 s`, this writes a VTK file every `0.01 s`.
13. Legacy VTK metadata was checked statically: `POINTS` count now uses the same `p_type > 1000` condition as the write loop, and `FIELD FieldData` declares 12 arrays matching the 12 arrays actually written.
14. The copied SOPHIA solid-fluid heat-transfer expression was re-audited against Imatani & Sakai Eqs. (29)-(31).
15. Static search confirmed the old Ranz-Marshall-style terms (`Nu = 2 + 0.6 Re^0.5 Pr^(1/3)`, hard-coded `0.0518`, and `h_conv = Nu * ki / d`) are no longer present in `source_modified/*.cuh`.
16. Full input generation and input sanity validation were re-run after the equation patch.
17. A real `make` compile attempt was made and failed only because `nvcc` is not installed on this Mac mini.

## Input file format and particle count

The current `input.txt` format is:

```text
first line: labels for 18 variables
remaining lines: one particle per line, with 18 tab-separated values
```

Therefore the original SOPHIA particle-count convention is valid again:

```text
num_part = total_input_lines - 1 header line
         = 360,001 - 1
         = 360,000
```

This avoids the previous failure mode where a scalar-per-line file had 6,480,000 data lines and was interpreted as 6,480,000 particles.

## Input particle plot files

```text
validation/input_plots/input_particles_overview.png
validation/input_plots/input_particles_3d_sample.png
validation/input_plots/input_plot_summary.md
```

The overview image shows the expected Case 3-1-1 rectangular fixed bed:

- x extent: approximately 0 to 200 mm
- y extent: approximately 0 to 40 mm
- z extent: approximately 0 to 40 mm
- DEM solid particles and gas SPH carrier particles both occupy the intended block.
- No obvious geometry inversion or missing region was visible in the XY, XZ, and YZ projections.

## Real tool output summary

```text
generated DEM=320000 gas=40000 total=360000
PASS DEM=320000 gas=40000
input.txt format: one particle per tab-separated line
input.txt lines: 360,001
input.txt data lines / particles: 360,000
solid-fluid heat transfer: Imatani & Sakai Eq. (29) Gunn Nu_fs + Eq. (30) Pr + Eq. (31) h_fs/Q_fs
old copied convection forms removed from source_modified/*.cuh: Ranz-Marshall Nu, hard-coded 0.0518, h_conv = Nu * ki / d
plot-output frequency: 100 steps = 0.01 s at dt=1.0e-4 s
VTK FIELD arrays: 12 declared / 12 written
make -n: nvcc -w -use_fast_math -arch=sm_60 -O3 -expt-relaxed-constexpr SOPHIA_gpu.cu -o SOPHIA_gpu -I./cub-1.8.0/ -lpthread
make: failed with `make: nvcc: No such file or directory`
```

## Blocker

CUDA compile/run was not performed on this Mac mini because `nvcc` is not installed/found.

```text
nvcc: NOT_FOUND
```

## CUDA machine next command

```bash
cd /Users/hojin/Documents/SOPHIA_cases/advanced-heat-transfer-sph-interpolation/case-3-1-1-sph-interpolation-dem-heat-transfer/source_modified
make
./SOPHIA_gpu 1
```
