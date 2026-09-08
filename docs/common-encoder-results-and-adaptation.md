# 공통 encoder 비교 결과와 다음 학습 — 2026-09-05

## 결론

최고 공식 제출은 여전히 0.7774307884다. 아래 값은 **개발셋 결과**이며,
공식 1등이나 예상 제출 점수가 아니다. 현재 강점은 encoder별로 다르다.
출력 residual의 조합 수를 늘리는 대신 EAT 내부 표현을 직접 학습하는 실험을 시작했다.

**후속 완료 결과:** EAT adapter의 6 epochs 학습과 전체 4,577파일 단독 재평가를
마쳤다. 동결 control 대비 ADS 0.820812→0.837857, File EER 18.88%→16.89%,
Voice EER 21.14%→19.50%, Music EER 14.18%→12.91%였다.
control의 5개 확률과 6 epochs의 sampler draw가 기존 동결 실험과 정확히 같았다.
이는 통제된 학습 개입의 개발 개선이지 공식 점수 0.837857을 받았다는 뜻이 아니다.

## 동일 조건의 동결 encoder + 공통 mean head 결과

개발 4,577파일, 동일 학습 샘플링/6 epochs/전체 구간 coverage. mean 조건끼리 표기했다.
음성·음악 EER은 해당 성분이 있는 파일에서만 계산한다.

| Encoder | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| EAT Large | 18.88% | 21.14% | **14.18%** | 0.820812 |
| XLS-R 2B | 19.84% | **12.62%** | 23.56% | 0.804873 |
| SPEAR (수정 전, 잠정) | 19.08% | 14.84% | 17.88% | 0.821267 |
| WavLM Large | 26.79% | 20.49% | 28.68% | 0.739034 |

각 checkpoint의 원래 task-specific classifier가 아니라 **같은 구조의 새 head**를
학습한 비교다. 예를 들어 XLS-R 행을 NII 배포 모델의 원래 classifier 성능이라고
해석하면 안 된다. encoder들의 사전학습 데이터·학습 이력은 같지 않다.
WavLM도 6 epochs 학습을 완료했다. 이 학습량의 **동결 표현 + 얕은 head**에서는
EAT/XLS-R보다 약했다. 이를 WavLM fine-tuning까지 불가능하다는 결론으로 확장하지
않는다. 수정된 SPEAR는 별도 완료 결과로 추가 비교한다.

세 encoder의 실제 학습 draw 순서가 6 epochs 모두 같음을 비교기가 검사했다.
6 epochs의 24,576 draws에서 실제 방문한 서로 다른 train 행은 10,165개다.
18,738개 모두를 본 학습이라고 표현하지 않는다.

### 혼합 오디오에서

동일 mixed 3,700파일에서 Voice/Music EER:

- EAT: 21.03% / 15.19%
- XLS-R: 12.49% / 25.14%
- SPEAR 수정 전: 14.38% / 19.35% — 독립 추론 문제로 잠정치

XLS-R 음악 성능은 music-only 561파일에서는 7.84%지만 mixed에서는 25.14%다.
따라서 "음악을 전혀 못 읽는다"보다 **혼합 상태에서 음악의 진위를 구별하는 데
약점이 있다**는 해석이 더 정확하다. 단, 두 집단의 원천 분포도 달라 순수한 혼합의
인과 효과로 단정하지 않는다.

코덱 개발 600파일의 XLS-R Voice EER은 clean 6.67%, G.711 8.33%, G.722 5.00%
지만 Opus narrowband 31.67%, G.711→Opus 33.33%다. 전화품질을 하나의 조건으로
묶으면 이 차이가 가려진다. EAT도 같은 Opus 조건의 Music EER이 31.67%/36.67%다.

근거: `reports/common_encoder_probe/comparison/`의 overall/by_type/codec_channels CSV.
개발셋은 generator-family 완전 미지 조건이 아니고 실제 test의 구성비도 모른다.

## 독립 예측 검증: 점수와 별개로 반드시 확인

EAT mean은 80파일의 저장된 batch 예측과 단독 예측이 최대 약 3×10⁻⁸ 차이로
일치했고, 역순 반복도 bit-exact였다. **전체 4,577파일 역순 검증은 아니다.**

SPEAR는 최대 확률 차이 0.197727로 실패했다. 원래 4파일 batch를 그대로 재생하면
저장 CSV와 약 3×10⁻⁸ 이내로 일치한다. 문제는 CSV나 checkpoint 교체가 아니었다.
동일 2.68초 target을 긴 neighbor와 묶으면 Music 확률이 0.412에서 0.214로 바뀐다.
native fbank의 batch 최대 길이 padding 이후 convolution/encoder를 통과하는 경로와
관련된 길이 의존성이 재현됐다. 작은 batch-shape 수치 차이도 별도로 관찰됐다.
모든 내부 층의 원인을 하나씩 분해했다는 주장은 하지 않는다.

수정 경로 `spear_independent`는 각 window를 native encoder에 하나씩 통과시킨 뒤
이미 계산된 token만 padding하여 head에서 묶는다. 4개 재현 사례의 token 차이는 0,
확률 차이는 최대 1.49×10⁻⁷이었다. 기존 학습/예측 결과를 덮어쓰지 않고 같은 학습
조건으로 새 head를 학습 중이다. 구간별 GPU 실행이 독립적이라는 뜻이지 다른
실험과 같은 GPU를 공유할 수 없다는 뜻은 아니다.

XLS-R도 80파일 검사에서 최대 0.004267 차이로 고정 tolerance 0.0005를 넘었다.
차이가 작다고 기준을 완화하지 않았다. 파일 단독 실행은 역순 반복과 bit-exact였다.
선택된 mean checkpoint를 바꾸지 않고 전체 개발셋 4,577개를 파일 단독으로
재평가했다. **File/Voice/Music EER과 ADS는 위 표와 동일**했다. 파일 단독 예측 CSV를
별도 보존했고, 3개 고정 파일의 재실행도 bit-exact였다. 확률이 원래 batch CSV와
전부 같다는 주장은 하지 않는다. 새 정확도 비교에는 단독 예측 CSV를 사용한다.

WavLM도 80파일 검사에서 최대 0.031710 차이가 발견됐다. 단독 역순은 bit-exact였다.
이 역시 원인을 과장해 단정하지 않고, 같은 고정 mean checkpoint를 전체 파일 단독으로
재평가 중이다. 현재 WavLM 표는 저장 batch 기준이며 새 제출에 사용하지 않는다.

근거: `spear_batch_diagnosis/`, `spear_batch_diagnosis_corrected/`,
`xlsr_independent_mean/`, `xlsr_file_local_mean/`, `wavlm_independent_mean/`.
이 실패를 과거 제출 pipeline 전체에 그대로 일반화하지 않는다.

## EAT 표현 학습: 무엇이 달라지는가

`configs/common_eat_adaptation.yaml`:

- 원본 혼합 오디오 → 같은 EAT frontend/초기 20개 block 공유.
- control: 마지막 block까지 동결, 공통 mean head만 학습.
- adapted: 마지막 4개 block 뒤에 bottleneck 32 adapter를 넣고 head와 함께 학습.
- 추가 학습 adapter 파라미터는 274,564개. base 가중치는 동결한다.
- adapter 마지막 projection은 0으로 초기화한다. 실제 가중치 smoke에서
  최초 control/adapted/native token이 **bit-exact**임을 확인했다.
- 같은 train/development, seed, 4,096 draws × 6 epochs, batch 4,
  10.24초 전체 구간 coverage, File/Voice/Music/Presence loss와 선택 기준을 유지한다.
- adapter LR 0.0003, head LR 0.001. 학습 중 adapter만 representation을 변경한다.
  데이터 전체 통계나 다른 평가 파일을 참조하지 않는다.

7개 단위 테스트 및 실제 CUDA smoke에서 gradient, frozen base 무변경,
초기 동등성, checkpoint/CSV 출력을 확인했다. smoke 점수는 정확도 근거가 아니다.
본 실험은 GPU 0에서 수정 SPEAR 학습과 함께 진행한 뒤 완료했다. 개발 개선은 확인했으나
공식 개선·미지 원천 일반화는 아직 미확인이다.

### Epoch 2 중간 결과 (최종 선택 아님)

| 동일 학습량 비교 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| 동결 control | 21.10% | 26.12% | 17.93% | 0.788456 |
| 마지막 4-block adapter | 19.69% | 23.85% | 16.19% | 0.805278 |

동일 시점의 세 진위 EER이 모두 개선됐다. 사전 고정한 6 epochs를 계속 진행한다.
6-epoch 동결 모델/공식 앵커 대비 승리 또는 미지 생성기 일반화의 증거는 아니다.
근거: `reports/common_eat_adaptation/full/history.json`.
새 코드 관련 통합 테스트 48개가 통과했다. 이는 탐지 성능 검증을 대신하지 않는다.

## 정리된 다른 실험

- v65 넓은 음량 증강: 실제 File slot 교체 개발 EER 23.33%→24.56%, 채택하지 않음.
- v67 저대역 Music: 고정 10% 결합에서 Music EER 25.33%→24.67%.
  사전 개선 기준 1%p에는 미달하므로 채택하지 않음. 위상 추가도 단독 기준에서
  magnitude보다 우수하지 않았다. 이는 위상 정보가 일반적으로 쓸모없다는 뜻은 아니다.

## 남은 판단

1. WavLM/수정 SPEAR/paired EAT 완료 결과 및 파일 단독 예측을 확인한다.
2. 4초 미만의 기존 개발 파일이 있음을 확인했으므로 header 기반으로 길이 분포와
   대회 범위 4–60초의 별도 점수를 산출한다. 사후 진단이며 locked 검증이 아니다.
3. 개발 기준으로 후보를 먼저 고정한 뒤 아직 안 본 source/generator 및 긴 파일에서
   검증한다. Suno 등 보호된 eval은 train이나 checkpoint 선택에 사용하지 않는다.
4. 그 후 공식 앵커 대비 파일별 개선, L4 60분/VRAM, 오프라인·ZIP 조건을 확인한다.

지금 단계에서는 추가 ZIP/API 제출을 하지 않는다.

## 길이 감사 완료

4,577개 실제 오디오 header를 읽었다. **1,232개(26.92%)가 4초 미만**이며,
최소 0.764초, 최대 20.2초다. 30–60초 파일은 하나도 없다. 대회 범위 4–60초에
들어가는 개발 파일은 3,345개지만, 이것도 실제로는 4–20.2초 범위만 다룬다.

4초 이상만 남긴 mean head의 File/Voice/Music EER 및 ADS:

| Encoder | File | Voice | Music | ADS |
|---|---:|---:|---:|---:|
| EAT | 19.34% | 23.31% | 14.82% | 0.812190 |
| XLS-R (단독 재평가 전) | 19.53% | 15.70% | 21.59% | 0.806202 |
| SPEAR (수정 전) | 19.46% | 18.10% | 17.00% | 0.815505 |
| WavLM | 27.81% | 23.63% | 27.47% | 0.731296 |

개발 길이 mismatch를 발견한 뒤 작성한 **사후 진단**이며 head/epoch 선택 기준을
이 표에 맞춰 다시 바꾸지 않는다. 30–60초 순차·부분 조작·겹침에 대한 개발 stress
와 아직 보지 않은 원천의 prospective 검증이 별도로 필요하다.
근거: `reports/common_encoder_probe/duration_audit/`.

## 완료 모델 검증과 코덱 개발 진단

- EAT adapted: 80파일 저장 batch/단독 예측 최대 차이 2.96×10⁻⁸,
  역순 bit-exact. 별도로 전체 4,577파일 단독 재평가에서도 EER/ADS가 같았다.
- WavLM: 전체 단독 재평가 ADS 0.739457. 기존 결론은 유지한다.
- 수정 SPEAR: 6 epochs 완료. mean의 File/Voice/Music EER은
  18.81%/14.54%/17.46%, ADS 0.824492. 80파일 단독 동등성 검사를 통과했다.
  최대 확률 차이는 2.37×10⁻⁷이고 역순 bit-exact였다.

기존 공식 앵커를 개발 codec 600에서 비교할 때 EAT adapted 전체 대체 ADS는
0.7600으로 앵커 0.7660보다 낮다. **전체 대체하지 않는다.**
다른 4개 출력을 그대로 두고 Music만 직접 대체한 진단은 Music EER
25.33%→21.33%, ADS 0.7660→0.7780이다. 별도 ensemble weight/calibration은 맞추지 않았다.
120개 base를 채널 전체와 묶은 500회 paired bootstrap의 Music EER 차이 구간은
[-9.00%p, +1.17%p]로 0을 포함한다. 공식 개선 확정이나 통계적으로 확실한 승리가 아니다.

근거: `reports/common_eat_adaptation/final_audit/`.
긴 파일 스트레스는 [v70 문서](long-component-stress-v70.md)를 참고한다.
