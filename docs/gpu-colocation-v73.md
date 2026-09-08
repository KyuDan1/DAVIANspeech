# GPU 하나에서 여러 추론 작업 실행

## 범위

사용자의 요청에 따라 연구 서버의 B200 하나에 v73 추론 프로세스를 여러 개 올리는
처리량 검사를 추가했다. 각 프로세스에는 XLS-R, SPEAR, EAT 세 모델이 함께 존재한다.
모델, 학습 checkpoint, 확률 결합식은 변경하지 않는다.

서로 다른 길이의 파일을 encoder의 하나의 padded batch로 묶지 않는다. 이전 SPEAR
검사에서 발견한 이웃 파일 의존성을 재도입하지 않도록 파일 단위 독립 추론을 유지하고,
별도 프로세스가 같은 GPU를 공유한다. 이 방식은 GPU 메모리를 더 쓰지만 학습이나
평가 지표를 바꾸려는 조치가 아니다.

## 측정 방법

`scripts/benchmark_gpu_colocation_v73.py`는 기존에 구현 동등성 검사에 사용한
development 80개를 대상으로 활성 프로세스 1/2/4개를 각각 세 차례 비교한다.
파일을 프로세스에 중복 없이 배분하고 원래 순서로 복원한다. 처리 결과는 저장된
v73 확률과 최대 오차 0.0005 이내여야 하며, 병렬/직렬 결과는 bit-exact여야 한다.

모델 네 세트는 측정 내내 상주한다. 음원 디코딩과 모델 로딩은 측정에서 제외한다.
조건 실행 순서를 번갈아 바꾸고 wall-clock 중앙값으로 가장 빠른 병렬도를 고른다.
작업당 CPU intra-op thread 2개, inter-op thread 1개로 제한한다.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONWARNINGS=ignore::FutureWarning \
/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python \
  scripts/benchmark_gpu_colocation_v73.py \
  --output reports/component_composition_v73/gpu_colocation_benchmark
```

출력이 존재하면 덮어쓰지 않는다. 실제 완료 보고서는 위 디렉터리의 `report.json`이며,
`frozen.json`이나 중간 `trials.jsonl`만 있는 상태는 완료가 아니다.

## 해석 제한

- 메모리가 넉넉하더라도 GPU 연산·메모리 대역폭·CPU가 포화되면 병렬화가 더 느릴 수 있다.
  가장 많은 프로세스 수를 무조건 선택하지 않는다.
- 미사용 평가 자료나 정답별 정확도를 사용해 병렬도를 선택하지 않는다.
- 짧은 개발 음원에서의 추론 측정이다. 긴 60초 파일, 디코딩 포함 전체 작업,
  다른 학습 작업과 동시에 수행할 때의 성능까지 보증하지 않는다.
- B200의 연구 실행 설정이며 L4 22.4GiB 제출 설정으로 그대로 복사하지 않는다.
  제출용 메모리와 1,200파일 전체 시간은 별도 검증해야 한다.
- 자동 제출, 기존 패키지 변경, 다른 사용자의 프로세스 중단은 하지 않는다.

## 완료 결과와 실행 기준

2026-09-05, GPU 0의 NVIDIA B200에서 전체 9회 측정을 완료했다.

| 활성 프로세스 | 80파일 처리 중앙값 | 직렬 대비 처리량 |
|---|---:|---:|
| 1개 | 9.072초 | 1.00배 |
| 2개 | 7.117초 | 1.27배 |
| 4개 | 6.755초 | 1.34배 |

4개 병렬은 직렬 대비 시간이 약 25.5% 줄었다. 2개에서 4개로 늘리는 추가 이득은
약 5.4%이므로 메모리 여유에 비례해 속도가 증가하지는 않는다. 현재 후보의 B200
연구 추론에는 여유가 있을 때 **GPU당 4개 프로세스, 프로세스당 CPU thread 2개**를
기준으로 사용한다. GPU 점유 상태를 먼저 확인하며 다른 작업이 있으면 조정한다.
학습 작업이나 다른 모델의 최적 병렬도까지 동일하다고 가정하지 않는다.

각 프로세스의 peak allocated 메모리는 12,179MiB, peak reserved는 약 13,172MiB였다.
네 프로세스의 reserved 합계는 약 51.4GiB이며 CUDA context 등은 별도다.
9회 모두 직렬 결과와 bit-exact이고 기존 저장 확률과 최대 오차는 3.87e-14였다.

관련 실행 분할/복원 테스트 10개와 기존 조합 테스트 8개가 통과했다.
보고서: `reports/component_composition_v73/gpu_colocation_benchmark/report.json`.
벤치마크의 자식 프로세스들은 완료 후 정상 종료했다. 성능 측정용 반복 추론이며,
새 정확도 평가·학습·ZIP 생성·API 제출은 하지 않았다.
