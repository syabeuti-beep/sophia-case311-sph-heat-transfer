from __future__ import annotations
from pathlib import Path

LABEL_NAMES = {1:'x',2:'y',3:'z',4:'ux',5:'uy',6:'uz',7:'m',8:'p_type',9:'h',10:'temp',11:'pres',12:'rho',13:'rho_ref',27:'rad',28:'ri',29:'dem_idx',30:'wx',31:'wy',32:'wz',36:'vol_power',37:'buffer_type'}

def write_input_txt(path: Path, particles: list[dict], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('	'.join(str(x) for x in labels) + '
')
        for p in particles:
            for label in labels:
                key = LABEL_NAMES[label]
                if key not in p:
                    raise ValueError(f'missing field {key!r} for label {label}')
                f.write(f"{p[key]}
")

def write_solv_txt(path: Path, cfg: dict) -> None:
    solver=cfg.get('solver', {})
    dt=solver.get('dt', 1e-7); time_end=solver.get('time_end', 1e-3)
    out_freq=solver.get('output_frequency', 1000); dim=solver.get('dimension', 3)
    text = f"""# SOPHIA Input: Solver Setup	// generated input
solver_type (WCSPH/ISPH/DFSPH):							ISPH

dimension(1/2/3): 											{dim}
property_table(YES/NO):									NO

kernel type (Gaussian/Quartic/Quintic/Wendland[2/4/6]):		Wendland6
filter type (Shepard/MLS): 									Shepard
density calculation (Direct/Continuity): 					Continuity
time integration (Euler/Predictor_Corrector): 				Predictor_Corrector
fluid type (Liquid/Gas): 									Liquid
simulation type (Single_Phase/Two_Phase):					Single_Phase

pressure-force solve(YES/NO): 								YES
viscous-force solve(YES/NO): 								YES
turbulence-model (Laminar/k-lm/k-e/SPS/HB):                 Laminar
artificial-viscous-force solve(YES/NO): 					NO
gravitational-force solve(YES/NO): 							YES
surface-tension-force solve(YES/NO): 						NO
surface-tension-model (Potential/Curvature)					Potential
interface-sharpness-force solve(YES/NO):					NO
boundary-force solve(YES/NO):								NO

Conduction(YES/NO):											NO
Boussinesq-natural-convection solve(YES/NO):				NO
Concentration-diffusion(YES/NO):							NO
XSPH solve(YES/NO): 										YES
kernel-gradient-correction solve(NO/KGC/FPM/DFPM/KGF):				KGC
delta-SPH solve(NO/Molteni/Antuono):									Antuono
particle-shifting solve(YES/NO):							NO
penetration_box solve(YES/NO):           NO
noslip_boundary (YES/NO):                NO
open boundary (OPEN/PERIODIC/NO):				NO
switch_ptype (YES/NO):          NO

reference-pressure (Pa): 									0
sound-speed:                                               200.0
reference-density-eos:                                      1000.0
gamma (for EOS): 											1
kappa: 														2
XSPH-coefficient:											0.1
Boundary-coefficient:										0.00005

minimum-iteration:											3
maximum-iteration:											10
pressure-convergence criterion								10
density-convergence criterion								0.005
pressure-relaxation factor									1

time-step (sec): 											{dt}
start-time (sec):											0.0
end-time (sec):												{time_end}

cell-initialization frequency: 								1
z-indexing (YES/NO)											NO
timestep update (YES/NO)									NO
neighbor cell type (3X3/5X5)								3X3
filtering frequency: 										40
density-renormalization frequency: 						1
temperature-filtering frequency:							10
plot-output frequency:									    {out_freq}
plot-variables:                         3 rho pres p_type

velocity-limit (m/s): 										40.0

Xmargin(-): 												0.0
Xmargin(+): 												0.0
Ymargin(-): 												0.0
Ymargin(+): 												0.0
Zmargin(-): 												0.0
Zmargin(+): 												0.0
decouple-stride:                                            1
"""
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding='utf-8')

def write_data_txt(path: Path, cfg: dict) -> None:
    mu=cfg.get('fluid', {}).get('mu', 0.001)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""#p_type 0
//T
10, 100, 200, 300, 400
//h
0, 1000, 2000, 3000, 4000
//k
1, 1, 1, 1, 1
//mu
{mu}, {mu}, {mu}, {mu}, {mu}
#end
""", encoding='utf-8')
