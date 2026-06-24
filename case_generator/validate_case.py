#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
VALID_LABELS={1:'x',2:'y',3:'z',4:'ux',5:'uy',6:'uz',7:'m',8:'p_type',9:'h',10:'temp',11:'pres',12:'rho',13:'rho_ref',27:'rad',28:'ri',29:'dem_idx',30:'wx',31:'wy',32:'wz',36:'vol_power',37:'buffer_type'}

def validate_input(path: Path):
    lines=path.read_text(encoding='utf-8').splitlines()
    if not lines: return ['input.txt is empty']
    labels=[int(x) for x in lines[0].split()]
    errors=[]
    for lab in labels:
        if lab not in VALID_LABELS: errors.append(f'unknown label {lab}')
    vals=lines[1:]
    if len(vals) % len(labels) != 0: errors.append(f'value count {len(vals)} not divisible by label count {len(labels)}')
    for i,v in enumerate(vals[:10000]):
        try:
            x=float(v)
            if math.isnan(x) or math.isinf(x): errors.append(f'non-finite value at data line {i+2}: {v}')
        except ValueError:
            errors.append(f'non-numeric value at data line {i+2}: {v}')
    return errors

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case-root', default='.')
    case=Path(ap.parse_args().case_root).resolve(); errors=[]
    required=['manifest.yaml','reproduction_request.yaml','experiment_spec.yaml','source_modified','case_generator/config.yaml','source_modified/input/solv.txt','source_modified/input/data.txt','source_modified/input/input.txt']
    for rel in required:
        if not (case/rel).exists(): errors.append(f'missing {rel}')
    inp=case/'source_modified/input/input.txt'
    if inp.exists(): errors += validate_input(inp)
    report=case/'validation/input_sanity_report.md'; report.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        report.write_text('# Input Sanity Report

Status: FAIL

'+'
'.join(f'- {e}' for e in errors)+'
', encoding='utf-8')
        print(f'FAIL: {len(errors)} issue(s). See {report}'); raise SystemExit(1)
    report.write_text('# Input Sanity Report

Status: PASS
', encoding='utf-8')
    print(f'PASS: see {report}')
if __name__ == '__main__': main()
