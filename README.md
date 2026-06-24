# Case 3-1-1 SPH-interpolation DEM heat-transfer package

## 목적

이 case는 Imatani & Sakai (2025)의 Case 3-1-1 fixed bed 검증 조건을 SOPHIA 복사본에 맞춘 것입니다.
논문 원래 아이디어는 DEM 입자의 접촉 상태가 아니라 void fraction 기반의 Eulerian/grid heat-transfer equation으로 고체상 온도장을 푸는 것입니다.
여기서는 사용자의 요청에 맞게 grid heat solve를 사용하지 않고, DEM 입자 이웃에 대한 SPH interpolation/Laplacian 방식으로 고체상 열전달을 계산하도록 수정했습니다.

## 중요한 범위

- coarse grain model은 구현하지 않았습니다.
- 원본 SOPHIA 코드는 수정하지 않았습니다.
- 수정은 이 case 내부의 `source_modified/` 복사본에만 들어 있습니다.

## 주요 파일

```text
source_modified/function_SPH_DEM_COUPLING.cuh   # SPH interpolation DEM heat conduction 추가
source_modified/function_OUTPUT.cuh             # sph_solid_volume_fraction, sph_ks_eff 출력 추가
source_modified/ISPH.cuh                        # Case 3-1-1 fixed-bed domain patch
source_modified/Parameters.cuh                  # Case 3-1-1 gas property/inlet constants
source_modified/function_PROP.cuh               # Table 1 물성 반영
source_modified/input/input.txt                 # 생성된 입자 입력, 360000 particles
source_modified/input/solv.txt                  # solver 설정
source_modified/input/data.txt                  # property table
case_generator/generate_case311_sph_heat.py     # input 생성기
case_generator/validate_case311.py              # input sanity checker
validation/case311_generation_report.md
validation/input_sanity_report.md
validation/input_plots/input_particles_overview.png
validation/input_plots/input_particles_3d_sample.png
validation/input_plots/input_plot_summary.md
experiment_spec.yaml
patch_summary.yaml
```

## 생성된 입자 수

- DEM solid particles: 320,000
- gas SPH carrier particles: 40,000
- total: 360,000

DEM은 200 mm x 40 mm x 40 mm 영역에 1 mm primitive cubic lattice로 배치됩니다.
이는 `200 * 40 * 40 = 320000` 입자입니다.

## 열전달 수정 개념

기존 코드에는 DEM particle이 주변 SPH gas/fluid 온도를 kernel interpolation으로 받아 convective heat transfer를 계산하는 부분이 이미 있었습니다.
이번 수정은 여기에 solid-solid heat diffusion 항을 추가했습니다.

핵심 구현 위치:

```cpp
source_modified/function_SPH_DEM_COUPLING.cuh
```

추가된 개념:

```text
1. 각 DEM particle i 주변 DEM neighbor j 탐색
2. SPH kernel로 local solid volume fraction alpha_s 추정
3. alpha_s에서 effective solid conductivity ks_eff 계산
4. Brookshaw-style SPH heat Laplacian으로 dT/dt 계산
5. 기존 fluid-particle convective dT/dt와 합산
```

접촉 면적, 접촉 시간, DEM spring constant는 열전달식에 넣지 않았습니다.

## 실행 준비

CUDA/NVIDIA GPU 환경에서 다음을 실행합니다.

```bash
cd /Users/hojin/Documents/SOPHIA_cases/advanced-heat-transfer-sph-interpolation/case-3-1-1-sph-interpolation-dem-heat-transfer/source_modified
make
./SOPHIA_gpu 1
```

이 Mac mini에는 현재 `nvcc`가 없어 제가 실제 CUDA compile/run까지는 수행하지 못했습니다.

## 입력 재생성

전체 Case 3-1-1 입력 재생성:

```bash
cd /Users/hojin/Documents/SOPHIA_cases/advanced-heat-transfer-sph-interpolation/case-3-1-1-sph-interpolation-dem-heat-transfer
python3 case_generator/generate_case311_sph_heat.py --case-root .
python3 case_generator/validate_case311.py --case-root .
```

입력 생성기는 항상 입자 위치 확인용 그림도 함께 저장합니다.

```text
validation/input_plots/input_particles_overview.png
validation/input_plots/input_particles_3d_sample.png
validation/input_plots/input_plot_summary.md
```

직접 다시 그리고 싶으면 다음을 실행합니다.

```bash
python3 case_generator/plot_input_particles.py --case-root .
```

작은 smoke-test 입력을 만들고 싶으면 다음을 쓸 수 있습니다.

```bash
python3 case_generator/generate_case311_sph_heat.py --case-root . --demo
```

주의: `--demo`는 full Case 3-1-1이 아니라 빠른 테스트용 작은 입력입니다.

## 출력 확인

VTK solid output에 다음 field가 추가됩니다.

```text
temperature_surface
Q_sd
dtemp_dt
sph_solid_volume_fraction
sph_ks_eff
rad
```

Case 3-1-1 검증에서는 x 방향 평균 solid temperature profile을 Figure 16과 비교하면 됩니다.
