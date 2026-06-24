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
10. `gpu_count_particle_numbers2` was patched and probed with the generated input: 6,480,000 scalar lines / 18 labels = 360,000 particles.

## Particle-count bug found after CUDA run

The generated `input.txt` was correct, but SOPHIA's original particle-count helper was wrong for this file layout.

`input.txt` format:

```text
first line: labels for 18 variables
remaining lines: one scalar value per line
```

The old `gpu_count_particle_numbers2` returned `line_count - 1`. For this case that meant:

```text
6,480,000 scalar value lines -> num_part = 6,480,000
```

Only the first 360,000 particle records were actually filled by `read_input`. The remaining 6,120,000 allocated `part1` slots stayed zero-initialized, so `p_type == 0 < 1000` and they were counted as SPH. That explains the observed SPH count:

```text
40,000 real gas SPH particles + 6,120,000 zero tail slots = 6,160,000 SPH particles
```

The fix is to count particles as:

```text
scalar_data_lines / number_of_input_labels = 6,480,000 / 18 = 360,000
```

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
input.txt size: 48,219,735 bytes
input.txt lines: 6,480,001
make -n: nvcc -w -use_fast_math -arch=sm_60 -O3 -expt-relaxed-constexpr SOPHIA_gpu.cu -o SOPHIA_gpu -I./cub-1.8.0/ -lpthread
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
