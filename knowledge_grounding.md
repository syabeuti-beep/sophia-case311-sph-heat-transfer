# Knowledge grounding

## 논문에서 추출한 Case 3-1-1 핵심 조건

Source PDF:

`/Users/hojin/Downloads/advanced-heat-transfer-model-for-eulerian-lagrangian-simulations-of-industrial-gas-solid-flow-systems.pdf`

논문 핵심 아이디어:

- 기존 DEM 열전달은 contact area/contact duration/spring constant에 민감합니다.
- 논문은 DEM motion은 Lagrangian/DEM으로 두되, heat transfer는 Eulerian framework/grid에서 풉니다.
- solid heat conduction은 particle contact state가 아니라 void fraction 기반 effective thermal conductivity로 표현합니다.
- Case 3은 coarse-grained DEM과 원래 particle system의 solid temperature distribution 비교입니다.

Case 3-1-1:

- fixed bed
- primitive cubic lattice
- particles in point contact
- void fraction reported: 0.476
- original particle system, no coarse graining
- particle diameter: 1.0 mm
- number of particles: 320,000
- solid initial temperature: 90 C = 363.15 K
- gas inlet temperature: 20 C = 293.15 K
- gas superficial velocity: 1.0 m/s
- time step: 1.0e-4 s
- physical properties from Table 1:
  - solid density: 2500 kg/m3
  - solid cp: 840 J/(kg K)
  - solid k: 1.4 W/(m K)
  - gas density: 1.2 kg/m3
  - gas viscosity: 2.0e-5 Pa s
  - gas cp: 1010 J/(kg K)
  - gas k: 0.025 W/(m K)

## User-requested deviation from the paper

The paper's heat solve is grid/Eulerian. The requested implementation is:

```text
Do not solve DEM heat transfer on grid.
Use the paper's void-fraction-based idea, but compute heat transfer through SPH interpolation.
Do not reproduce the coarse-grain model.
Validate Case 3-1-1.
```

Therefore, this package intentionally replaces the grid heat solve with a DEM-neighbor SPH interpolation model.

## Model implemented

For each DEM particle i:

1. Search neighboring DEM particles j using SOPHIA's existing cell list.
2. Estimate local solid volume fraction:

```text
alpha_s_i = sum_j V_j W_ij
```

3. Compute an effective solid conductivity:

```text
ks_eff = k_gas + (k_solid - k_gas) alpha_s_i^1.5
```

This is a pragmatic grid-free analogue of the paper's void-fraction-controlled effective conductivity. It is not the exact printed grid formula because the user explicitly requested no grid heat solve.

4. Compute SPH heat diffusion using a Brookshaw-style Laplacian:

```text
dT_i/dt += ks_eff/(rho_i cp_i) * sum_j 2 V_j (T_i - T_j) dW/dr / (r_ij + 0.01 h_ij)
```

5. Use the paper's solid-fluid heat-transfer law for DEM-gas convection:

```text
Nu_fs = (7 - 10 eps + 5 eps^2)(1 + 0.7 Re_p^0.2 Pr^(1/3))
        + (1.33 - 2.4 eps + 1.2 eps^2) Re_p^0.7 Pr^(1/3)
Pr = C_pf mu_f / k_f
h_fs = Nu_fs k_f / d_p
Q_fs = h_fs (6/d_p) (T_f - T_s)
```

6. Add the solid-solid diffusion term to the paper-aligned DEM-gas convective heat-transfer term.

## Known ambiguity

The extracted text says Case 3 grid size was 6.0 mm, while Table 4 extraction shows 8.0 mm. Since the requested modification removes the grid heat solve, this ambiguity does not affect the implemented solid heat-transfer operator. The generated gas SPH carrier spacing is 2 mm.
