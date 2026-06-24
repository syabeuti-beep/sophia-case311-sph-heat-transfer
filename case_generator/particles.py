from __future__ import annotations
from geometry import regular_block

def generate_particles(cfg: dict) -> list[dict]:
    geom = cfg['geometry']
    spacing = geom['spacing']
    h = cfg.get('particles', {}).get('smoothing_length') or spacing * 1.3
    fluid = cfg.get('fluid', {})
    rho = fluid.get('rho', 1000.0)
    ux, uy, uz = fluid.get('velocity', [0.0, 0.0, 0.0])
    temp = fluid.get('temperature', 293.0)
    pres = fluid.get('pressure', 0.0)
    vol = spacing ** 3
    m = rho * vol
    particles=[]
    for x, y, z in regular_block(geom['domain'], spacing):
        particles.append({'x': x, 'y': y, 'z': z, 'ux': ux, 'uy': uy, 'uz': uz,
                          'm': m, 'p_type': 1, 'h': h, 'temp': temp, 'pres': pres, 'rho': rho})
    return particles
