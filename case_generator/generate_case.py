#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None
from particles import generate_particles
from write_sophia_input import write_input_txt, write_solv_txt, write_data_txt

def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise SystemExit('PyYAML is required: python3 -m pip install --user pyyaml')
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case-root', default='.', help='case workspace root')
    args=ap.parse_args()
    case=Path(args.case_root).resolve()
    cfg=load_yaml(case/'case_generator'/'config.yaml')
    labels=cfg.get('labels', [1,2,3,4,5,6,7,8,9,10,11,12])
    particles=generate_particles(cfg)
    input_dir=case/'source_modified'/'input'
    write_solv_txt(input_dir/'solv.txt', cfg)
    write_data_txt(input_dir/'data.txt', cfg)
    write_input_txt(input_dir/'input.txt', particles, labels)
    report=case/'validation'/'input_generation_report.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f'# Input Generation Report\n\nGenerated particles: {len(particles)}\n', encoding='utf-8')
    print(f'generated {len(particles)} particles under {input_dir}')
if __name__ == '__main__':
    main()
