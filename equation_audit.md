# Equation audit: Imatani & Sakai (2025) Case 3-1-1

This file records the paper equations that are implemented or approximated in the SOPHIA copy under `source_modified/`.

## Paper energy equations

The paper solves heat transfer in an Eulerian framework. The fluid and solid phase equations are printed as Eqs. (20) and (21):

```math
\frac{\partial T_f}{\partial t}+\mathbf{u}\cdot\nabla T_f
= \frac{1}{\rho_f C_{pf}}\nabla\cdot(k_f^{eff}\nabla T_f)
+\frac{Q_{fs}}{\rho_f C_{pf}}+\frac{Q_{fw}}{\rho_f C_{pf}}
```

```math
\frac{\partial ((1-\varepsilon)T_s)}{\partial t}+\nabla\cdot(\mathbf{v}_{CGM}T_s)
= \frac{1}{\rho_s C_{ps}}\nabla\cdot(k_s^{eff}\nabla T_s)
+\frac{Q_{fs}}{\rho_s C_{ps}}+\frac{Q_{sw}}{\rho_s C_{ps}}
```

For this SOPHIA package, the user requested no Eulerian grid heat solve. Therefore:

- the DEM particle motion remains Lagrangian;
- gas-to-solid convection is applied to DEM particles using the paper's solid-fluid correlation;
- the solid-phase diffusion part is approximated by an SPH/Brookshaw operator over DEM neighbors;
- DEM contact area, contact duration, and spring constant are not used in the heat-transfer calculation.

## Solid-fluid heat transfer: Eqs. (29)-(31)

The paper uses the Gunn packed/fluidized-bed correlation:

```math
Nu_{fs}=(7-10\varepsilon+5\varepsilon^2)\left(1+0.7Re_p^{0.2}Pr^{1/3}\right)
+(1.33-2.4\varepsilon+1.2\varepsilon^2)Re_p^{0.7}Pr^{1/3}
```

with

```math
Pr = \frac{C_{pf}\mu_f}{k_f}
```

and the volumetric solid-fluid heat source form:

```math
Q_{fs}=h_{fs}a_p(T_f-T_s),\qquad h_{fs}=\frac{Nu_{fs}k_f}{d_p},\qquad a_p=\frac{S_p}{V_p}=\frac{6}{d_p}
```

### SOPHIA implementation

Implemented in:

```text
source_modified/function_SPH_DEM_COUPLING.cuh
```

Important code mapping:

- `sophia_case311_gunn_nusselt(eps, rep, pr)` implements Eq. (29).
- `Pr_* = heat_capacity(...) * VISCOSITY_AIR / k_*` implements Eq. (30) for the Case 3-1-1 gas branch.
- `h_conv_* = Nu_* * k_* / d_i` and `dq_vol = h_conv_* * (6/d_i) * (T_f - T_s)` implement Eq. (31).
- `P3_dem[i].dtemp = dq_vol/(rho_i C_{ps,i}) + sph_dem_heat_dtemp` applies the temperature source to DEM particle `i`.

Compared with the previous copied SOPHIA state, this replaces the Ranz-Marshall-style form
`Nu = 2 + 0.6 Re^0.5 Pr^(1/3)` and removes the incorrect use of solid conductivity `ki` in the gas/air convection coefficient.

## DEM-neighbor SPH solid diffusion approximation

Because this package intentionally avoids the paper's Eulerian grid heat solve, the solid effective conduction term is approximated as a particle-neighbor sum.

For DEM particle `i`, neighboring DEM particles `j` are searched using SOPHIA's existing DEM cell list. The local solid fraction is estimated by

```math
\alpha_{s,i}=\sum_{j\in DEM} V_j W_{ij},\qquad V_j=\frac{m_j}{\rho_j}
```

A grid-free effective solid conductivity is then assigned as

```math
k_{s,i}^{eff}=k_f+(k_s-k_f)\alpha_{s,i}^{1.5}
```

The temperature diffusion source is applied using a Brookshaw-style SPH Laplacian:

```math
\left(\frac{dT_i}{dt}\right)_{ss}
=\frac{k_{s,i}^{eff}}{\rho_i C_{ps,i}}
\sum_{j\in DEM}2V_j(T_i-T_j)\frac{\partial W_{ij}/\partial r}{r_{ij}+0.01h_{ij}}
```

where `W_ij` and `dW_ij/dr` are SOPHIA's `calc_kernel_wij` and `calc_kernel_dwij`.

## Case 3-1-1 constants used

From the paper/Table 1 and Case 3 text:

- `rho_s = 2500 kg/m3`
- `C_ps = 840 J/(kg K)`
- `k_s = 1.4 W/(m K)`
- `rho_f = 1.2 kg/m3`
- `mu_f = 2.0e-5 Pa s`
- `C_pf = 1010 J/(kg K)`
- `k_f = 0.025 W/(m K)`
- `d_p = 1.0 mm`
- gas inlet/superficial velocity: `1.0 m/s`
- initial solid/gas temperatures: `363.15 K` / `293.15 K`

## Remaining intentional deviation

The exact paper model is Eulerian/grid-based. This repository is intentionally a grid-free SOPHIA adaptation requested by the user, so the solid conduction term is not a bit-for-bit reproduction of Eqs. (20)-(28). The paper-aligned part newly enforced in this revision is the solid-fluid heat-transfer correlation, Eqs. (29)-(31).
