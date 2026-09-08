# v87: 배경 음악을 고정한 전경 음성 개입 실험

## 최종 결과: 두 후보 모두 제출하지 않음

`full_v2`의 4 epoch 학습과 개발 4,577파일 평가가 정상 완료됐다.
Music-present 4,261개에서 계산한 Music EER은 다음과 같다.

| 조건 | 기존 EAT parent | BCE 재학습 | 음성 교체 일관성 추가 |
|---|---:|---:|---:|
| 전체 Music EER | **12.91%** | 13.80% | 13.94% |
| 혼합 3,700개 | **13.89%** | 14.70% | 14.59% |
| 단독 음악 561개 | **5.35%** | 5.70% | 6.06% |
| codec mixed dev 600개 | 21.33% | **15.67%** | 18.33% |
| source-disjoint mixed 200개 | **8.00%** | 12.00% | 9.00% |
| 그 전화 변형 200개 | **15.00%** | 21.00% | 20.00% |

codec 개발집단은 좋아졌지만 다른 원천 집단과 전체 평균은 나빠졌다.
전화 변형에서도 원천에 따라 방향이 반대이므로 "전화 품질을 해결했다"고 말할 수 없다.
일관성 항도 BCE 대비 일관된 우위를 만들지 못했다. 개발 결과에 맞춰 추가 가중치나
router threshold를 탐색하지 않고 두 단독 교체 후보를 기각한다.
특정 코퍼스 편향이 원인이라는 가설은 남지만 이 결과만으로 인과가 입증된 것은 아니다.

실제 소비한 draw는 원래 TRAIN replay 2,073개 + 합성 pair 2,023개였다.
17개 가짜 generator 표기, 세 혼합 형태, 다섯 codec 조건이 모두 사용됐다.
학습+개발 평가 시간은 모델 로드 후 1,494.90초, 최대 CUDA allocated 3,313.01 MiB다.
두 adapter의 실제 가중치 변경, 고정 base gradient 부재, 입력/코드 hash 유지를 확인했다.
이 수치는 L4 시간 보증 또는 공식 점수 개선의 증거가 아니다.

- 완료 학습: `reports/music_counterfactual_v87/full_v2/report.json`
- 조건별 평가: `reports/music_counterfactual_v87/development_slices/metrics.csv`
- 실제 전체 draw 감사: `reports/music_counterfactual_v87/draw_audit_final/report.json`

아래 설계/진행 상태는 이 최종 결과 이전의 기록이다.

## 왜 이 실험인가

공식 현재 Music EER은 v84 probe로 31.71%다. XLS-R 단일 File head의 개발 진단에서도
RR 대 RF가 RR 대 FR보다 어려웠고 v86 local pooling으로 해결되지 않았다.
개발의 조합별 결과를 비공개 평가셋의 정답 분포로 해석하지 않는다.

v52는 clean→codec consistency를 mixture의 저장 통계 head에 적용했지만 새로운
원천에서 실패했다. 여기서는 **같은 음악에 진짜/가짜 전경 음성을 교체하는 개입**을
사용한다. 가설은 Music 판정이 전경 음성의 진위에 끌려가는 현상을 줄이는 것이다.
전화 채널 일관성을 새 방법으로 다시 이름 붙인 실험이 아니다.

## 현재 구현 범위: 데이터, 아직 모델 학습 아님

`src/music_counterfactual_data_v87.py`는 엄격한 TRAIN raw catalog에서 음악 하나,
진짜 음성 하나, 가짜 음성 하나를 선택한다. 같은 음악 crop에 두 음성을 각각 섞고
두 파일 모두 같은 Music 정답을 부여한다. File/Voice/Presence 정답은 만들지 않는다.
음악 원천에 보컬이 있을 수 있으므로 전경 음성의 진위를 전체 Voice 정답으로
간주하는 과거 레이블 위험을 피한다. Music 원천 정답의 신뢰성까지 입증한 것은 아니다.

- Music label 및 원천 generator별 균등 선택.
- 4/8초 native 무작위 crop. 짧은 원천에만 동일한 타일링 처리를 사용하고 trace로 표시.
- 동시, 음성 먼저 순차, 음악 먼저 순차의 세 형태.
- SNR −10/−5/0/5/10dB, clean/G711/G722/Opus-NB/이중 codec.
- pair의 음악 crop·음량·채널 설정·순서를 고정하고 전경 음성만 교체.
- codec 전 동일 pair peak scale을 사용해 전경 교체가 음악 음량을 바꾸지 않게 한다.
  이는 TRAIN 합성 과정이고 평가 추론의 파일 간 통계 공유가 아니다.
- codec은 비선형일 수 있어 변환 후 음악 성분까지 수학적으로 같다고 주장하지 않는다.
  원본 음악에서 분리·확산·생성 모델을 실행하지 않는다.
- 원천 payload SHA, 가수/곡 group, crop 위치, 타일링, codec seed를 기록한다.

네 단위 테스트는 동일 전경의 출력 동등성, 순차 음악 구간 불변, native crop
결정성, 잘못된 입력 거절을 검사했다. 실제 원천/codec 32 pair smoke는
`scripts/smoke_music_counterfactual_v87.py`로 별도 실행한다.
모델 학습 성공이나 성능 개선을 이 데이터 테스트가 대신하지 않는다.

실제 TRAIN smoke도 완료됐다. 32쌍을 각각 두 번 생성해 모든 출력과 trace가
bit-exact였다. Music real/fake는 19/13쌍, 3개 혼합 형태와 5개 채널이 모두
포함됐다. 원천 사용 slot 96개 중 32개는 길이가 짧아 타일링됐다.
이는 고유 원천 96개 또는 반복 없는 자연 장면이라는 뜻이 아니다.
결과: `reports/music_counterfactual_v87/data_smoke/report.json`.
후속 학습은 타일링 비율을 계속 기록하고 자연 원본 replay를 함께 사용해야 한다.

## 다음 학습의 통제 조건

후속 구현에서는 EAT Music 표현 adapter를 사용하되 다음 두 조건을 같은 초기값,
동일 TRAIN draw, 동일 epoch 수로 비교해야 한다.

1. 두 view 각각의 Music BCE만 사용.
2. 동일 BCE에 두 view의 Music logit 일관성 항 추가.

원래 TRAIN Music-present 예제도 replay해 특정 합성 구조에만 적응하지 않도록 한다.
평가셋/Suno는 학습에 넣지 않으며 마지막 epoch를 저장한 후 개발을 평가한다.
File/CPS를 바꾸거나 상수 probe를 추가 제출하는 실험이 아니다.
선택된 새 head는 Music만 바꾸고 다른 네 출력을 보존할 수 있어야 한다.
단일 파일 추론에는 짝이나 다른 평가 파일이 필요 없어야 한다.

단순 pair 데이터 증가 효과와 consistency 효과를 구분하고, 순수 음악/혼합 및
미지 원천·codec별 악화 여부를 확인하기 전 제출하지 않는다. 새 학습은 아직 시작하지 않았다.

## 학습 구현 및 CUDA smoke 완료 (위 설계 이후)

`scripts/train_music_counterfactual_v87.py`에서 두 EAT 모델을 같은 GPU에 올렸다.
부모는 `reports/common_eat_adaptation/full/adapted/head.pt`이며, 각각 마지막 4개
block의 기존 adapter와 공통 head를 Music loss로 학습한다. 원래 EAT base는 고정한다.
두 조건의 초기 adapter/head state는 bit-exact로 같음을 검사한다.
Music-only 배포 시 다른 네 출력은 새 모델 출력으로 교체하면 안 된다.

- 고정 seed 20260907, 4 epoch × 1,024 draw, 마지막 epoch만 저장 후 개발 평가.
- draw 절반 확률은 원래 TRAIN Music-present 파일 replay, 나머지는 음악 고정 pair.
- replay는 기존 corpus/File-class/component 균형 sampler를 사용한다.
  Music-label 완전 균등인 sampler라고 주장하지 않는다.
- BCE 조건은 pair 두 view의 Music BCE 평균, invariant 조건은 여기에
  `0.1 × (logit_real_voice − logit_fake_voice)^2`를 추가한다.
- head/adapter LR 모두 1e-4, weight decay 0.01, gradient clipping 5, LME T=2.
- native encoder는 window별 단독 처리하고, 평가 파일 간 정보는 공유하지 않는다.
- full 실행 전에 smoke의 config 및 코드·가중치·원천 hash가 모두 같은지 검사한다.

TRAIN-only 4 draw CUDA smoke가 완료됐다. 원래 파일 replay 2개와 합성 pair 2개를
처리했고 두 모델 모두 adapter gradient와 실제 가중치 변경을 확인했다.
원래 base parameter의 gradient는 없었다. 모델 로드 후 8.39초,
최대 CUDA allocated 3,091.68 MiB였다. 두 loss 숫자의 차이는 추가 항이 있으므로
정확도 우열로 해석하지 않는다.

결과: `reports/music_counterfactual_v87/train_smoke/report.json`.
같은 코드/config의 full 실행을 `reports/music_counterfactual_v87/full`에서 시작했다.
아직 개발 Music EER이나 공식 개선 결과는 없다.

### 첫 full 실행의 검증 중단과 재시작

첫 실행은 epoch 1의 157 draw를 기록한 뒤 다음 예제에서
"모든 step에서 adapter gradient가 0보다 커야 한다"는 assertion으로 중단됐다.
부모 가중치로부터 새로 시작하는 `full_v2`에서는 모든 adapter gradient의 연결 및
finite 여부를 매 step 검사하고, 0인 경우 loss/logit/label을 출력한다.
최종 adapter 가중치가 실제 초기값에서 변했는지도 계속 검사한다.
정확히 포화된 올바른 BCE 예제는 float32 gradient가 0일 수 있지만, 최초 중단이
그 경우였는지는 재현 로그를 본 뒤 판단한다. 이유 없이 검사를 없앤 것은 아니다.

원래 실패 실행과 157개 trace는 보존했다. 원래 trainer의 snapshot SHA가 실패 실행의
frozen manifest와 같은 것도 확인했다. 변경 코드의 TRAIN-only smoke를 다시 실행해
두 조건의 loss와 peak memory가 이전 smoke와 같고 검사를 통과함을 확인한 뒤
`reports/music_counterfactual_v87/full_v2`를 시작했다. 부분 학습을 이어 붙이지 않는다.

재현 실행의 같은 epoch 1 draw 157에서 BCE logit **16.969118**, 정답 **1**,
loss **4.2698e-8**일 때 gradient가 0인 것을 확인했다. 따라서 최초 중단은
연결이 끊긴 adapter가 아니라 올바른 양성 예제의 float32 BCE gradient 포화였다.
수정 실행은 이 지점을 지나 draw 200 이상 학습을 진행했다. 이 사실은 전체 학습의
성공이나 일반화 성능을 증명하지 않으며, 최종 가중치 변경 검사는 계속 남아 있다.

완료 모델의 Music만 반환하는 native 구현은
`src/music_counterfactual_inference_v87.py`에 준비했다. 완료된 full checkpoint만
허용하고 단일 window encoder 호출을 유지한다. 아직 완료 checkpoint에 대한
저장 확률 대응 검사는 실행 전이다.

완료 후 검증 명령은 다음과 같다. 학습 프로세스가 아직 살아 있는 상태에서 실행해
부분 산출물을 최종 결과로 해석하지 않도록 두 도구 모두 완료 report를 요구한다.

```bash
python scripts/analyze_music_counterfactual_v87.py --output reports/music_counterfactual_v87/development_slices
python scripts/check_music_counterfactual_v87.py --output reports/music_counterfactual_v87/native_equivalence
```

개발 분석은 parent/BCE/invariant의 ID와 5개 정답을 대응시킨 후 Music-present만
평가한다. 전체/단독 음악/혼합, 채널, 원천 코퍼스, Voice-real/Voice-fake 조건을
따로 보고하며 새로운 가중치나 threshold를 선택하지 않는다.
native 검사는 고정 간격 개발 80개를 두 모델 각각 정방향/역방향으로 처리해
저장 확률과 tolerance 2e-6 이내, 순서 반전 bit-exact인지 확인한다.
이는 모든 개발 파일의 순서 독립성 또는 L4 시간 보증을 대신하지 않는다.

## 실제 음악 원천 노출 감사 (1,106 draw 시점)

원래 TRAIN replay 566개, 합성 pair 540개를 실제 trace로 확인했다. 합성에는
세 혼합 형태와 다섯 채널이 모두 들어갔다. 가짜 음악 원천의 17개 generator 표기가
모두 등장했지만 MusicGen 등 같은 계열의 이름이 있어 17개 독립 생성기 계열이라고
해석하면 안 된다. 사용한 bank/group 조합은 REAL 109개, FAKE 152개였다.
이는 PCM으로 확인한 독립적인 원곡 수가 아니다.

중요한 잠재적 혼동 요인도 있다. 합성 REAL 음악 278회 중 177회(63.67%)가
`competition_v2_bad_presence` 출처지만 FAKE 262회 중에는 80회(30.53%)다.
즉 Music 정답과 원천 코퍼스가 상관되어 있다. generator별 균등 선택만으로
코퍼스별 진위 분포까지 균등해지는 것은 아니다. 원천 출처 흔적에 의존하는
shortcut의 가능성이며, 실제 모델이 그 shortcut을 쓴다는 인과 증거는 아직 없다.

현재 두 조건은 동일한 trace를 사용하므로 consistency의 통제 비교는 유지한다.
진행 중 sampler를 바꾸거나 새로운 corpus balancing을 섞지 않는다. 최종 결과는
원천 코퍼스별로 함께 해석하고, 새 학습을 한다면 source-bank와 label을 교차해
균형화하는 데이터 대조군을 별도로 고정해야 한다.

근거: `reports/music_counterfactual_v87/draw_audit_snapshot/report.json`,
재현 도구: `scripts/audit_music_draws_v87.py`. 학습 완료 전 snapshot이며 최종 분포는 아니다.
