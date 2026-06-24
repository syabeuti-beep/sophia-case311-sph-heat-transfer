# Missing information and assumptions

## Missing or ambiguous from the PDF extraction

1. Case 3-1 exact x-profile sampling/binning procedure for Figure 16 was not fully specified in extracted text.
2. The PDF text extraction reports grid size inconsistently:
   - body text: 6.0 mm
   - Table 4 extraction: 8.0 mm
   This package intentionally removes grid-based solid heat solve, so this ambiguity is documented but not used.
3. The exact wall/inlet/outlet implementation in the paper's CFD solver is not reproduced one-to-one in SOPHIA.
4. Full CUDA runtime verification was not possible on this Mac mini because `nvcc` was not found.

## Explicit assumptions made

1. Coordinate system:
   - x: gas flow / bed length direction, 0 to 0.200 m
   - y: bed height/thickness direction, 0 to 0.040 m
   - z: bed width/depth direction, 0 to 0.040 m
2. DEM primitive cubic lattice centers are placed every 1.0 mm.
3. Gas phase is represented as SPH carrier particles at 2.0 mm spacing with 1.0 m/s x-velocity and 293.15 K temperature.
4. No coarse-grain ratio is applied.
5. Solid-solid thermal diffusion is contact-state-free and depends on SPH-neighbor interpolation, not DEM contact overlap.
6. Effective conductivity is implemented as a smooth void-fraction analogue:
   `k_gas + (k_solid-k_gas)*alpha_s^1.5`.

## Recommended next verification on CUDA machine

1. Compile with `make` under `source_modified`.
2. Run `./SOPHIA_gpu 1`.
3. Inspect DEM VTK output fields:
   - `temperature_surface`
   - `dtemp_dt`
   - `sph_solid_volume_fraction`
   - `sph_ks_eff`
4. Bin/average `temperature_surface` along x and compare shape with paper Figure 16.
