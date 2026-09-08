# v86: 원래 TRAIN 분포를 유지하는 XLS-R File 시간 readout

## 완료된 개발 결과: 제출 후보로 채택하지 않음

4 epoch 학습과 개발 4,577개 평가는 완료됐다. 모델 로드 이후 학습+평가
1,202.79초, 최대 CUDA allocated 8,485.24 MiB이며 L4 실행시간 측정은 아니다.

| File EER ↓ | 기존 parent | 학습 global | 학습 local |
|---|---:|---:|---:|
| 전체 4,577개 | 19.82% | 20.43% | 21.69% |
| 혼합 3,700개 | 19.78% | 20.11% | 22.16% |
| 순수 음악 561개 | 9.98% | 8.56% | 12.12% |
| RR 대 FR | 16.22% | 16.97% | 21.41% |
| RR 대 RF | 31.03% | 31.89% | 31.57% |
| RR 대 FF | 9.19% | 8.97% | 12.32% |

F/R 순서는 음성/음악이다. 이 개발셋에서는 **진짜 음성+가짜 음악(RF)**가
더 어렵다. 다만 parent는 XLS-R 단일 File head이고 공식 최고 전체 앙상블이 아니므로
이 수치를 공식 모델의 조건별 EER 또는 비공개 평가 분포라고 해석하면 안 된다.
혼합 contrast는 같은 RR 925개를 공유한다. 독립적인 세 평가집단이 아니다.

global 대비 local이 전체와 세 혼합 contrast 모두 나빠졌다. 이번 고정 표현과
약한 File supervision 조건에서는 짧은 구간 pooling이 일반성을 개선하지 못했다.
이는 모든 시간 모델의 불가능성을 입증하는 결과가 아니라 이번 통제 실험의 기각이다.
0.5 기준 전체 real 오탐도 parent 16.95% → global 23.91% → local 32.42%지만,
이 고정 threshold 수치는 EER와 다르며 threshold 조정으로 EER 악화를 해결할 수 없다.

완료 입력 감사는 원래 TRAIN 2,061 draw + 합성 TRAIN 2,035 draw = 4,096이다.
합성에 7개 layout, 5개 채널, 5개 길이가 모두 포함됐고 원래 TRAIN inventory 대응과
File OR label 검사를 통과했다. metadata의 원천 성분 순수성을 청취로 입증한 것은 아니다.

- 최종 결과: `reports/file_local_readout_v86/full/report.json`
- 조건별 수치: `reports/file_local_readout_v86/development_slices/metrics.csv`
- 완료 입력 감사: `reports/file_local_readout_v86/audit_completed/report.json`

별도 native 추론 구현도 개발 80개 × 정/역순 2회 = 160회 검사를 완료했다.
저장 확률과 최대 차이 1.11e-16, 파일 순서 반전 시 출력 bit-exact였다.
이는 구현 대응과 파일 독립성 검사이며 정확도 개선의 증거는 아니다.
`reports/file_local_readout_v86/native_equivalence/report.json`에 기록했다.

긴 부분 가짜 진단 2,880개는 동일 고정 head로 별도 실행한다.
이 데이터는 이미 사용된 **exposed diagnostic**이며 새 blind 검증이라고 부르지 않는다.
`scripts/evaluate_file_local_long_v86.py`가 입력/모델/코드 hash를 고정하고
동일 GPU에 4개의 encoder worker를 올려 나누어 추론한다. 예측은 파일별 독립적이며
학습, 가중치 탐색 또는 자동 제출을 하지 않는다. 아래 실행 전/중간 상태는 과거 기록이다.

### 긴 오디오 2,880개 평가 완료

4 worker가 각각 720개씩 완료했고 모든 입력·모델 checksum과 행/ID 대응 검사를
통과했다. 각 worker 모델 로드 후 약 796–802초였고 L4 벤치마크는 아니다.

| File EER ↓ | 공식 최고 anchor 코드 | XLS-R parent | 학습 global | 학습 local |
|---|---:|---:|---:|---:|
| uniform 720개 | 41.39% | 32.78% | 33.33% | 30.28% |
| fixed 2,160개 | 31.20% | 34.81% | 35.09% | 31.02% |

local은 긴 두 진단에서 parent/global보다 좋아졌다. 따라서 짧은 구간 정보가
전혀 쓸모없다는 해석은 맞지 않는다. 그러나 일반 개발 전체 19.82%→21.69%와
혼합 19.78%→22.16% 악화는 남는다. fixed에서 전체 anchor 대비 차이는 0.19%p로
작으며 표본 재사용을 고려한 유의성이 입증된 것도 아니다.
이 결과를 근거로 사후 router threshold/ensemble weight를 탐색하거나 바로 제출하지 않는다.
일반 오디오의 성능 보존 없이 긴 부분 가짜 개선만으로 전체 교체를 정당화할 수 없다.
완료 결과: `reports/file_local_readout_v86/long_exposed/report.json`.

목표는 실제 0.85 이상이다. v85에서는 단순 성분 확률 재결합과 기존 EAT 직접 File
head 교체가 모두 실패했다. 이번에는 **XLS-R의 File용 latent projection**에서
시간 구간별 통계를 계산하는 학습 실험을 한다.

## 기존 실험과 차이

- v80은 Voice projection과 가짜 음성 삽입 pair만 사용했다. 이번에는 File projection,
  File loss, RR/FR/RF/FF, 순수 음악/음성, 부분 가짜 음악까지 다룬다.
- v71/v72는 EAT 표현과 dense component supervision이었다. 이번에는 XLS-R 표현,
  File-only 약한 지도학습이며 평가 구간 정답으로 학습하지 않는다.
- v60 WPT backbone 학습과 달리 이번에는 backbone/projection을 고정한다.
  먼저 시간 readout 자체의 차이를 같은 표현에서 분리한다. 이 실험을 전체 backbone
  학습 또는 기존 실패 방법과 완전히 무관한 새 SOTA 구조라고 부르지 않는다.

## 통제 비교

두 head 모두 기존 File output weight/bias에서 동일하게 초기화한 193 parameter
linear head다. 같은 TRAIN draw, 같은 frozen encoder forward를 공유한다.

- global: 10.24초 window 전체의 mean/std → File logit.
- local: 약 1초 token neighborhood mean/std, 약 0.2초 간격 → File logit.
- 둘 다 모든 유효 readout logit을 LME(T=5)로 합쳐 파일 정답 BCE로 학습한다.
- XLS-R attention은 window 전체 문맥을 보므로 local pooling이 1초 receptive field인
  독립 encoder나 음원 분리라는 뜻은 아니다.
- 겹친 window의 local 통계는 window별로 취급한다. 유일한 시간 가중치로 재분배하지
  않으며, 비교 대상 global도 같은 window 구성을 쓴다.

원래 parent File(T=2)는 최종 개발 때 별도로 보고한다. 이 기준과 학습 모델의 차이는
학습+T 변화가 포함되므로, pooling의 통제 효과는 **학습 global 대 학습 local**로 판단한다.

## 데이터와 선택

strict TRAIN 18,738행과 raw source catalog 1,767개의 기존 identity/protected-role
검사를 다시 실행한다. draw의 50%는 원래 TRAIN, 50%는 TRAIN 원천 합성이다.
원본 sampling은 corpus→File class→성분 조합을 균형화한다. 합성은
4/8/16/32/60초, 0.5/1/2/4초 삽입, clean/G711/G722/Opus/transcode를 포함한다.

4 epoch × 1,024 draw, 마지막 checkpoint를 먼저 저장한 다음 development 4,577개를
한 번 평가한다. 보호 eval/Suno는 학습이나 checkpoint 선택에 사용하지 않는다.
TRAIN-only smoke를 통과한 동일 코드/config만 full training으로 진행한다.
개발 성능 개선을 공식 제출 점수 예상으로 사용하지 않는다.

구현: `scripts/train_file_local_readout_v86.py`, `src/file_local_readout_v86.py`.
설정: `configs/file_local_readout_v86.yaml`.

## 실행 상태

기본 테스트 3개와 TRAIN-only 4 draw smoke를 통과했다. 두 head 모두 업데이트됐고
frozen encoder/projection의 gradient는 없었다. smoke는 원래 TRAIN,
순수 음성, 부분 가짜 음악, 순차 혼합 및 실제 codec을 포함했다.
모델 로드 후 smoke 13.07초, peak CUDA 8,485 MiB.

`reports/file_local_readout_v86/full`에서 4×1,024 draw 전체 실행을 시작했다.
결과가 완료되기 전 학습 성공, 개발 EER 개선 또는 공식 성능 향상을 주장하지 않는다.

## 중간 입력 감사 및 후속 검증 준비

실제 처리한 첫 433 draw를 별도 read-only 도구로 확인했다.
원래 TRAIN 224, 합성 209였고, 합성에는 7개 layout과 5개 채널이 모두 포함됐다.
File real/fake는 전체 193/240이었다. 합성 File label의 성분 OR 일치와 원래 TRAIN의
row/ID/corpus 대응을 확인했다. 학습 도중의 snapshot이며 최종 분포를 대신하지 않는다.
성분 조합은 제작 metadata 기준이다. 원천 음악에 숨은 보컬이 전혀 없다는 청취 검증을
이 수치가 대신하지 않으며, 이번 학습은 File 정답만 사용한다.

- 중간 감사: `reports/file_local_readout_v86/audit_midrun/report.json`
- 완료 후 native 구현: `src/file_local_inference_v86.py`
- 완료 후 대응 검사: `scripts/check_file_local_v86.py`

후속 대응 검사는 개발 80개 파일을 정방향/역방향으로 처리해 학습 스크립트의
최종 저장 확률과 비교한다. 아직 실행 전이며 통과했다고 주장하지 않는다.

조건별 분석 도구 `scripts/analyze_file_readouts_v86.py`도 완료 전에 준비했다.
parent/global/local의 DATASET·ID·5개 정답 열이 일치하는지 먼저 검사하고,
단독/혼합, corpus, channel, layout 및 RR-vs-FR/RF/FF별 File EER과 표본 수를
함께 저장한다. 0.5 고정 threshold의 FPR/FNR은 EER와 분리해 보고한다.
이 도구는 가중치·threshold를 학습하거나 checkpoint를 재선택하지 않는다.
