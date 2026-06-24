#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
LABELS=[1,2,3,4,5,6,7,8,9,10,11,12,27,28,29,30,31,32]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case-root', default='.')
    case=Path(ap.parse_args().case_root).resolve(); inp=case/'source_modified/input/input.txt'
    errors=[]
    n_dem=n_gas=parser_particle_count=0
    vals=[]
    minT=1e9; maxT=-1e9
    if not inp.exists():
        errors.append(f'missing {inp}')
    else:
        with inp.open() as f:
            labels=[int(x) for x in f.readline().split()]
            if labels != LABELS: errors.append(f'labels mismatch: {labels}')
            data_lines=0
            for line_no, line in enumerate(f, start=2):
                s=line.strip()
                if not s: continue
                row=s.split()
                data_lines += 1
                if len(row)!=len(LABELS):
                    errors.append(f'line {line_no}: expected {len(LABELS)} tab-separated values, got {len(row)}')
                    if len(errors)>=10: break
                    continue
                vals.extend(row)
                ptype=int(float(row[7])); temp=float(row[9]); rad=float(row[12])
                minT=min(minT,temp); maxT=max(maxT,temp)
                if ptype>1000:
                    n_dem+=1
                    if abs(rad-0.0005)>1e-12 and len(errors)<10: errors.append(f'bad DEM radius at particle {n_dem+n_gas}: {rad}')
                else:
                    n_gas+=1
                    if ptype!=3 and len(errors)<10: errors.append(f'unexpected SPH p_type {ptype}')
                for v in row:
                    x=float(v)
                    if math.isnan(x) or math.isinf(x): errors.append('nonfinite value')
            parser_particle_count = data_lines
            if parser_particle_count != n_dem + n_gas:
                errors.append(f'SOPHIA parser particle-count mismatch: data_lines={parser_particle_count}, classified={n_dem+n_gas}')
            if parser_particle_count != 360000:
                errors.append(f'expected 360000 total particles for full Case 3-1-1, got parser_count={parser_particle_count}')
            if n_dem != 320000: errors.append(f'expected 320000 DEM particles for full Case 3-1-1, got {n_dem}')
            if n_gas != 40000: errors.append(f'expected 40000 gas SPH carrier particles for full Case 3-1-1, got {n_gas}')
    for rel in ['source_modified/function_SPH_DEM_COUPLING.cuh','source_modified/function_OUTPUT.cuh','source_modified/input/solv.txt','source_modified/input/data.txt']:
        if not (case/rel).exists(): errors.append(f'missing {rel}')
    report=case/'validation/input_sanity_report.md'; report.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        report.write_text('# Input sanity report\n\nStatus: FAIL\n\n'+'\n'.join(f'- {e}' for e in errors)+'\n', encoding='utf-8')
        print(f'FAIL {len(errors)} issues; see {report}')
        raise SystemExit(1)
    report.write_text(f'# Input sanity report\n\nStatus: PASS\n\nDEM particles: {n_dem}\nGas SPH particles: {n_gas}\nTotal particles: {parser_particle_count}\nInput data lines: {parser_particle_count}\nInput labels: {len(LABELS)}\nInput format: one particle per tab-separated line\nTemperature range: {minT} - {maxT} K\n', encoding='utf-8')
    print(f'PASS DEM={n_dem} gas={n_gas} report={report}')
if __name__ == '__main__': main()
