# v89: 진짜/가짜 음악 × 진짜/가짜 음성 2×2 순위 학습

## 최종 결과: 기각, 제출하지 않음

고정된 4 epoch 학습과 개발 4,577파일 평가를 완료했다. Music-present 4,261개에서
기존 EAT parent가 두 후보보다 좋았다.

| Music EER ↓ | parent | factorial BCE | factorial rank+invariance |
|---|---:|---:|---:|
| 전체 4,261개 | **12.91%** | 14.27% | 14.50% |
| 혼합 3,700개 | **13.89%** | 15.19% | 15.14% |
| 단독 음악 561개 | **5.35%** | 6.06% | 6.42% |
| Voice-real 혼합 | **13.51%** | 14.81% | 14.70% |
| Voice-fake 혼합 | **14.92%** | 15.57% | 15.78% |

ranked 조건은 학습 중 BCE를 plain 조건보다 낮추고, voice-swap logit MSE를
epoch 1 평균 1.2472에서 epoch 4 평균 0.6673으로 낮췄다. 그러나 개발 EER은
오히려 더 나빠졌다. 즉 개입 목적함수 최적화에는 성공했지만 새로운 파일의
Music 순위 일반화에는 실패했다. loss 감소를 탐지 개선으로 바꿔 말하지 않는다.

두 모델 모두 단독 음악과 두 혼합 foreground 조건에서 parent보다 나빴으므로
hard router로 일부 유형에만 적용할 근거도 없다. 개발 결과를 본 뒤 residual
weight를 탐색하지 않고 두 후보를 기각한다. 공식 현재 Music EER 31.71%를 줄였다는
증거는 없으며 ZIP/API 제출을 만들지 않는다.

학습+평가 시간은 모델 로드 뒤 2,002.09초, peak CUDA allocated 3,311.02 MiB다.
단일 파일 native 검사는 80파일×정/역순×두 모델=320회를 통과했다. 저장 예측과
최대 차이는 두 모델 모두 1.11e-16, 순서 반전 출력은 bit-exact였다.

- 전체 결과: `reports/music_factorial_v89/full/report.json`
- 조건별 결과: `reports/music_factorial_v89/development_slices/metrics.csv`
- 구현 대응: `reports/music_factorial_v89/native_equivalence/report.json`

아래 내용은 결과 전에 고정한 설계와 진행 기록이다.

## 목표와 v87과의 차이

공식 현재 Music EER은 31.71%로 가장 높다. v87은 같은 음악에 진짜/가짜 음성을
교체했지만 기존 EAT parent 12.91% 대비 BCE 13.80%, 일관성 13.94%로 개발
Music EER이 악화돼 기각했다. 같은 음악 안에서는 Music 정답이 하나이므로
가짜 음악을 진짜 음악보다 직접 높게 정렬하는 paired 신호가 없었다.

v89는 한 TRAIN draw에 다음 네 파일을 함께 만든다.

```text
진짜 음악 + 진짜 음성    진짜 음악 + 가짜 음성
가짜 음악 + 진짜 음성    가짜 음악 + 가짜 음성
```

같은 음성 열에서 `가짜 음악 logit > 진짜 음악 logit`을 직접 학습한다. 같은 음악
행에서는 두 음성 개입 전후 logit을 가깝게 만든다. 이것은 source separation이나
생성형 변환이 아니며, 네 raw TRAIN 원천 파형을 동일 규칙으로 혼합한다.

## 데이터 통제

- 진짜/가짜 Music 모두 `competition_v2_bad_presence`와 `echoes_v1`을 정확히
  50:50 확률로 선택한다. 각 bank 안에서는 generator 표기를 균등 선택한다.
- 음성은 진짜/가짜 generator family를 각각 균등 선택한다.
- 4/8초, 동시·음성 먼저·음악 먼저, SNR −10/−5/0/5/10dB,
  clean/G711/G722/Opus-NB/G711→Opus를 사용한다.
- 네 view가 같은 geometry, SNR, channel seed를 쓴다. codec 전 peak scale도
  네 view 전체에서 하나만 계산해 label별 음량 단서를 줄인다.
- 음악 원천에 보컬이 있을 가능성 때문에 Music 정답만 사용한다.
  File/Voice/Presence 정답은 이 합성에서 만들지 않는다.
- 평가셋과 Suno 보호셋은 학습 및 checkpoint 선택에 쓰지 않는다.

단위 테스트는 동일 음악/동일 음성 개입의 예상 동등성, 순차 구간 공유,
잘못된 입력 거절과 bank 확률을 포함해 관련 10개가 통과했다.

## 같은 입력을 쓰는 두 조건

부모는 완료된 EAT-large adapted model이다. 두 모델은 동일한 초기 head/adapter,
동일 draw와 optimizer 설정을 사용한다. EAT 원래 base는 고정하고 기존 마지막
4-block adapter와 head만 조정한다.

| 모델 | 합성 4-way loss |
|---|---|
| `factorial_bce` | 네 Music BCE 평균 |
| `factorial_ranked` | 같은 BCE + 0.2×paired rank + 0.02×voice-swap logit MSE |

draw 절반은 기존 Music-present TRAIN 원본 replay이고 절반은 4-way 합성이다.
4 epochs × 768 draws, head/adapter LR 1e-4, weight decay 0.01,
LME T=2이며 마지막 epoch를 개발 결과 전에 고정 저장한다.

TRAIN-only 4-draw CUDA smoke는 통과했다. 두 모델 모두 finite loss/gradient와
adapter 실제 변경, 고정 base gradient 부재를 확인했다. 모델 로드 뒤 9.11초,
peak CUDA allocated 3,302.52 MiB였으며 정확도 또는 L4 실행시간 증거는 아니다.

전체 실행은 `reports/music_factorial_v89/full`에서 시작했다. 완료 전에는 성능 개선,
공식 점수 향상 또는 제출 후보라고 주장하지 않는다. 두 조건 중 하나가 개발에서
이겨도 source-disjoint/전화/긴 부분 가짜 조건을 따로 확인해야 한다.
