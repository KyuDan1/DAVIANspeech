# v80 개발 결과: 구간 감독은 도움이 되지만 기존 모델을 아직 대체하지 못함

4 epoch × 512 실제/가짜 pair(총 4,096 waveform)의 학습과 개발 음성 4,016개의 평가가
완료됐다. 각 epoch의 pair는 별도 seed로 생성했으며 실제 사용 source/DSP trace를 저장했다.
보호 eval 및 Suno는 학습 또는 마지막 checkpoint 선택에 사용하지 않았다.

| Voice 조건 | 기존 global head | local + 파일 정답만 | local + 구간 감독 |
|---|---:|---:|---:|
| 전체 4,016개 | **12.62%** | 20.74% | 17.03% |
| 음성만 316개 | **18.03%** | 19.30% | 19.30% |
| 실제 음악, RR 대 FR | **12.54%** | 21.30% | 16.76% |
| 가짜 음악, RF 대 FF | **12.76%** | 21.19% | 18.27% |

수치는 Voice EER이며 낮을수록 좋다. 기존 global head는 같은 실행에서 새로 계산했다.
구간 감독은 같은 local 구조의 파일 정답 학습보다 전체 EER을 **3.71%p** 줄였지만,
global head보다 **4.41%p** 높다. 따라서 전체 Voice 모델 교체나 제출은 하지 않는다.

## 드러난 문제

0.5 고정 threshold의 실제 음성 오탐률은 global 7.87%, file-only 61.60%, dense 32.73%다.
가짜 음성을 더 민감하게 찾는 동시에 실제 음성도 많이 Fake로 올리는 방향으로 변했다.
그러나 EER도 악화되어, 이 문제를 0.5 threshold 보정만으로 해결된다고 볼 수는 없다.

학습은 짧은 가짜 구간을 삽입한 실제/가짜 pair에만 집중했다. 기존 일반 TRAIN 파일의
분포를 유지하는 손실은 쓰지 않았다. 이것이 일반 개발셋 성능 저하의 원인일 수 있지만,
데이터 구성과 local pooling 효과를 모두 분리한 대조 실험이 아니므로 확정하지 않는다.

원래 목표인 긴 파일의 부분 가짜 검출 효과는 별도 확인한다. **이미 결과를 본 v61/v74
bank**에서의 진단이며 새로운 blind 검증은 아니다. 두 고정 head와 global head를 모두
평가하고, 이 자료에 맞춰 threshold/가중치를 탐색하지 않는다. 전체 점수·음악·CPS 개선은
Voice만의 결과에서 주장하지 않는다.

## 구현 대응 검증

별도 native 추론 구현으로 개발 80파일을 정방향/역방향 총 160회 실행했다.
학습 스크립트의 최종 개발 확률과 최대 차이는 1.11e-16이고, 파일 순서를 바꿔도
출력은 bit-exact였다. 이웃 파일에 의존하지 않는다. 실제 실행된 파일의 결과이며
기본 테스트만으로 동등성을 주장한 것이 아니다.

학습+개발 평가 시간은 모델 준비 후 약 20분 42초, peak CUDA 약 8,487MiB였다.
두 head가 같은 frozen XLS-R forward를 공유했다. B200 연구 수치이며 제출 L4 보증이 아니다.

- 전체 학습 결과: `reports/local_voice_v80/full/report.json`
- 조건별 분석: `reports/local_voice_v80/context_audit/report.json`, `slices.csv`
- native 추론: `src/local_voice_inference_v80.py`
- 구현 동등성: `reports/local_voice_v80/inference_equivalence/report.json`
- 긴 파일 진단 실행: `scripts/evaluate_local_voice_long_v80.py`

## 긴 부분 가짜 음성 진단 완료

| 기존에 노출된 합성 bank | 공식 최고 anchor Voice | global parent | local 파일 감독 | local 구간 감독 |
|---|---:|---:|---:|---:|
| uniform, 720개 | 42.50% | 36.11% | **25.56%** | 27.50% |
| fixed, 2,160개 | 39.35% | 37.78% | **26.39%** | 26.85% |

4개 shard의 2,880개 예측, 입력 hash, 원래 anchor 출력 hash를 검증한 뒤 계산했다.
긴 파일에서는 두 local 학습 모두 개선됐지만, 구간 정답 추가는 파일 정답만 쓴 것보다
우수하지 않았다. 즉 현재 증거는 **긴 파일의 짧은 가짜 구간에 특화된 학습/집계의 효과**를
지지하지만, dense supervision 자체의 우월성이나 일반적인 성능 향상을 입증하지 않는다.
개발셋 악화와 함께 보면 전면 교체보다는 원래 음성 분포를 보존하는 학습이 다음 과제다.
이 bank는 기존 노출 자료이므로 새로운 blind 성능이나 공식 점수 예상으로 취급하지 않는다.

최종 결과: `reports/local_voice_v80/long_exposed_diagnostic/report.json`.

공식 최고 제출은 여전히 0.77743이다. v80은 공식 제출되지 않았다.
