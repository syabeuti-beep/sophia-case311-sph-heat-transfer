#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, copy
from pathlib import Path

LABELS = [1,2,3,4,5,6,7,8,9,10,11,12,27,28,29,30,31,32]

DEFAULT = {
    'domain_m': {'x': [0.0, 0.200], 'y': [0.0, 0.040], 'z': [0.0, 0.040]},
    'dem': {'diameter_m': 0.001, 'density_kg_m3': 2500.0, 'specific_heat_j_kgk': 840.0,
            'thermal_conductivity_w_mk': 1.4, 'initial_temperature_k': 363.15,
            'h_m': 0.002, 'p_type': 1001},
    'gas': {'spacing_m': 0.002, 'density_kg_m3': 1.2, 'viscosity_pa_s': 2.0e-5,
            'specific_heat_j_kgk': 1010.0, 'thermal_conductivity_w_mk': 0.025,
            'temperature_k': 293.15, 'superficial_velocity_m_s': 1.0, 'h_m': 0.002, 'p_type': 3},
    'solver': {'dt_s': 1.0e-4, 'end_time_s': 10.0, 'output_frequency': 1000,
               'kappa': 2.0, 'sound_speed_m_s': 20.0, 'reference_density': 1.2}
}

def centers(lo, hi, d):
    n = int(round((hi-lo)/d))
    for i in range(n):
        yield lo + (i+0.5)*d

def make_particles(cfg):
    dom, dem, gas = cfg['domain_m'], cfg['dem'], cfg['gas']
    particles=[]
    d=dem['diameter_m']; r=0.5*d; rho=dem['density_kg_m3']
    vol=4.0/3.0*math.pi*r**3; m=rho*vol; ri=0.4*m*r*r
    idx=0
    for x in centers(*dom['x'], d):
        for y in centers(*dom['y'], d):
            for z in centers(*dom['z'], d):
                particles.append([x,y,z,0.0,0.0,0.0,m,dem['p_type'],dem['h_m'],dem['initial_temperature_k'],0.0,rho,r,ri,idx,0.0,0.0,0.0])
                idx += 1
    dg=gas['spacing_m']; rhog=gas['density_kg_m3']; mg=rhog*dg**3
    for x in centers(*dom['x'], dg):
        for y in centers(*dom['y'], dg):
            for z in centers(*dom['z'], dg):
                particles.append([x,y,z,gas['superficial_velocity_m_s'],0.0,0.0,mg,gas['p_type'],gas['h_m'],gas['temperature_k'],0.0,rhog,0.0,0.0,-1,0.0,0.0,0.0])
    return particles, idx, len(particles)-idx

def write_input(path, particles):
    with path.open('w', encoding='utf-8') as f:
        f.write('\t'.join(map(str, LABELS))+'\n')
        for row in particles:
            for v in row:
                f.write(f'{v}\n')

def write_solv(path, cfg):
    s=cfg['solver']
    path.write_text(f'''# SOPHIA Input: Solver Setup\t// Case 3-1-1 SPH-interpolation DEM heat transfer
solver_type (WCSPH/ISPH/DFSPH):\t\t\t\t\t\t\tISPH

dimension(1/2/3): \t\t\t\t\t\t\t\t\t\t\t3
property_table(YES/NO):\t\t\t\t\t\t\t\t\tNO

kernel type (Gaussian/Quartic/Quintic/Wendland[2/4/6]):\t\tWendland6
filter type (Shepard/MLS): \t\t\t\t\t\t\t\t\tShepard
density calculation (Direct/Continuity): \t\t\t\t\tContinuity
time integration (Euler/Predictor_Corrector): \t\t\t\tPredictor_Corrector
fluid type (Liquid/Gas): \t\t\t\t\t\t\t\t\tGas
simulation type (Single_Phase/Two_Phase):\t\t\t\t\tSingle_Phase

pressure-force solve(YES/NO): \t\t\t\t\t\t\t\tYES
viscous-force solve(YES/NO): \t\t\t\t\t\t\t\tYES
turbulence-model (Laminar/k-lm/k-e/SPS/HB):                 Laminar
artificial-viscous-force solve(YES/NO): \t\t\t\t\tNO
gravitational-force solve(YES/NO): \t\t\t\t\t\t\tNO
surface-tension-force solve(YES/NO): \t\t\t\t\t\tNO
surface-tension-model (Potential/Curvature)\t\t\t\t\tPotential
interface-sharpness-force solve(YES/NO):\t\t\t\t\tNO
boundary-force solve(YES/NO):\t\t\t\t\t\t\t\tNO

Conduction(YES/NO):\t\t\t\t\t\t\t\t\t\t\tYES
Boussinesq-natural-convection solve(YES/NO):\t\t\t\tNO
Concentration-diffusion(YES/NO):\t\t\t\t\t\t\tNO
XSPH solve(YES/NO): \t\t\t\t\t\t\t\t\t\tNO
kernel-gradient-correction solve(NO/KGC/FPM/DFPM/KGF):\t\t\t\tKGC
delta-SPH solve(NO/Molteni/Antuono):\t\t\t\t\t\t\t\t\tNO
particle-shifting solve(YES/NO):\t\t\t\t\t\t\tNO
penetration_box solve(YES/NO):           NO
noslip_boundary (YES/NO):                NO
open boundary (OPEN/PERIODIC/NO):\t\t\t\tNO
switch_ptype (YES/NO):          NO

reference-pressure (Pa): \t\t\t\t\t\t\t\t\t0
sound-speed:                                               {s['sound_speed_m_s']}
reference-density-eos:                                      {s['reference_density']}
gamma (for EOS): \t\t\t\t\t\t\t\t\t\t\t1
kappa: \t\t\t\t\t\t\t\t\t\t\t\t\t\t{s['kappa']}
XSPH-coefficient:\t\t\t\t\t\t\t\t\t\t\t0.0
Boundary-coefficient:\t\t\t\t\t\t\t\t\t\t0.00005

minimum-iteration:\t\t\t\t\t\t\t\t\t\t\t3
maximum-iteration:\t\t\t\t\t\t\t\t\t\t\t10
pressure-convergence criterion\t\t\t\t\t\t\t\t10
density-convergence criterion\t\t\t\t\t\t\t\t0.005
pressure-relaxation factor\t\t\t\t\t\t\t\t\t1

time-step (sec): \t\t\t\t\t\t\t\t\t\t\t{s['dt_s']}
start-time (sec):\t\t\t\t\t\t\t\t\t\t\t0.0
end-time (sec):\t\t\t\t\t\t\t\t\t\t\t\t{s['end_time_s']}

cell-initialization frequency: \t\t\t\t\t\t\t\t1
z-indexing (YES/NO)\t\t\t\t\t\t\t\t\t\t\tNO
timestep update (YES/NO)\t\t\t\t\t\t\t\t\tNO
neighbor cell type (3X3/5X5)\t\t\t\t\t\t\t\t3X3
filtering frequency: \t\t\t\t\t\t\t\t\t\t40
density-renormalization frequency: \t\t\t\t\t\t1
temperature-filtering frequency:\t\t\t\t\t\t\t10
plot-output frequency:\t\t\t\t\t\t\t\t\t    {s['output_frequency']}
plot-variables:                         5 rho pres p_type temp dtemp_dt

velocity-limit (m/s): \t\t\t\t\t\t\t\t\t\t40.0

Xmargin(-): \t\t\t\t\t\t\t\t\t\t\t\t0.0
Xmargin(+): \t\t\t\t\t\t\t\t\t\t\t\t0.0
Ymargin(-): \t\t\t\t\t\t\t\t\t\t\t\t0.0
Ymargin(+): \t\t\t\t\t\t\t\t\t\t\t\t0.0
Zmargin(-): \t\t\t\t\t\t\t\t\t\t\t\t0.0
Zmargin(+): \t\t\t\t\t\t\t\t\t\t\t\t0.0
decouple-stride:                                            1
''', encoding='utf-8')

def write_data(path, cfg):
    g, d = cfg['gas'], cfg['dem']
    path.write_text(f'''#p_type 3
//T
293.15, 313.15, 333.15, 353.15, 373.15
//h
0, 0, 0, 0, 0
//k
{g['thermal_conductivity_w_mk']}, {g['thermal_conductivity_w_mk']}, {g['thermal_conductivity_w_mk']}, {g['thermal_conductivity_w_mk']}, {g['thermal_conductivity_w_mk']}
//mu
{g['viscosity_pa_s']}, {g['viscosity_pa_s']}, {g['viscosity_pa_s']}, {g['viscosity_pa_s']}, {g['viscosity_pa_s']}
#end
#p_type 1001
//T
293.15, 313.15, 333.15, 353.15, 373.15
//h
0, 0, 0, 0, 0
//k
{d['thermal_conductivity_w_mk']}, {d['thermal_conductivity_w_mk']}, {d['thermal_conductivity_w_mk']}, {d['thermal_conductivity_w_mk']}, {d['thermal_conductivity_w_mk']}
//mu
0, 0, 0, 0, 0
#end
''', encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case-root', default='.'); ap.add_argument('--demo', action='store_true')
    args=ap.parse_args(); case=Path(args.case_root).resolve(); cfg=copy.deepcopy(DEFAULT)
    if args.demo:
        cfg['domain_m']={'x':[0.0,0.020],'y':[0.0,0.008],'z':[0.0,0.008]}; cfg['solver']['end_time_s']=0.01
    particles, n_dem, n_gas = make_particles(cfg)
    inp=case/'source_modified/input'; inp.mkdir(parents=True, exist_ok=True)
    write_input(inp/'input.txt', particles); write_solv(inp/'solv.txt', cfg); write_data(inp/'data.txt', cfg)
    (case/'case_generator/case311_config.txt').write_text(repr(cfg), encoding='utf-8')
    report=case/'validation/case311_generation_report.md'; report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f'# Case 3-1-1 input generation report\n\nDEM particles: {n_dem}\nGas SPH particles: {n_gas}\nTotal: {len(particles)}\nDomain: {cfg["domain_m"]}\n\nThe original paper uses an Eulerian grid heat-transfer solve. This case keeps the fixed-bed geometry but replaces solid heat conduction with DEM-neighbor SPH interpolation in function_SPH_DEM_COUPLING.cuh.\n', encoding='utf-8')
    print(f'generated DEM={n_dem} gas={n_gas} total={len(particles)}')
if __name__ == '__main__': main()
