# Dedicated channel-invariant Music head v52

2026-09-04 기준. 이 실험은 v18의 File/Voice/Music 공동 head에서 Music 표현이 희석되는
문제를 피하려고 **원본 혼합 오디오 통계만 읽는 Music 전용 head**를 새로 학습했다.
결론부터 말하면 허용 개발셋에서는 강했지만 source-disjoint codec blind에 일반화되지
않아 **제출 후보에서 제외(reject)** 한다. 기존 `music_only_v51`의 3.75% residual을
유지하는 것이 더 안전하다.

## 최종 결정

| 방법 | codec dev v4 Music EER | retired blind v5 Music EER | blind v6 Music EER | 결정 |
|---|---:|---:|---:|---|
| exact v50 Music | 0.27333 | 0.19167 | 0.25833 | anchor |
| 기존 music_only_v51, 3.75% | 0.25333 | **0.18333** | 미사용 | 유지 |
| 새 KL head seed04, 3.75% | **0.25000** | 0.20000 | 0.25833 | reject |
| 새 KL head seed04, dev 선택 20% | **0.19667** | 0.25000 | 0.26667 | reject |
| 새 KL head seed04 단독 | 0.21000 | 0.32500 | 0.34167 | reject |
| 새 KL head seed05 단독 | 0.22500 | 0.33750 | 0.26667 | reject |

`blind_v5`는 퇴역한 retrospective 진단으로만 보았고, `blind_v6`는 진단으로만 한 번
보았다. 둘 다 checkpoint, seed, residual weight 선택에는 사용하지 않았다. 허용된
development만 사용하면 seed04 20%가 선택되지만 v5에서 EER가 `+0.05833` 악화되므로
명백한 과적합이다. 동일한 보수 가중치 3.75%에서도 기존 v51이 v4에서는 0.00333
불리한 대신 v5에서 0.01667 우수해 일반성 기준 Pareto 선택은 기존 v51이다.

## 모델과 학습

새 head는 source separation이나 stem을 전혀 사용하지 않는다. v50이 원본 혼합
오디오에서 이미 계산하는 EAT 768차원 통계와 SPEAR 13-layer 1280차원 통계를 읽는다.

- EAT/SPEAR를 각 128차원으로 투영한다.
- stream/view/stat/layer embedding을 붙인다.
- Music 전용 attentive-statistics query 하나가 모든 원본-mixture token을 읽는다.
- 주 목적함수는 Music BCE와 batch 내부 positive-negative pairwise rank다.
- Voice/File BCE는 합계 5%의 학습 안정화 보조일 뿐, 추론 출력에는 쓰지 않는다.
- 동일 source의 clean을 teacher, codec을 student로 둔 비대칭 Bernoulli-KL을 0.5로
  적용한다. teacher에는 dropout을 끄고 gradient를 흘리지 않는다.

초기 버전은 확률 Smooth-L1과 stochastic teacher를 사용했다. dev v4 20% residual은
0.19000까지 좋아졌지만 v5가 0.26667로 악화됐다. 또한 동일 원본의 clean↔codec 평균
절대 예측 변화가 v5에서 anchor `0.1860`보다 큰 `0.2851`이었다. 이 결함을 직접-logit
gradient를 갖는 KL과 deterministic teacher로 고쳤지만 최종 seed04도 v5에서
`0.2862`였으므로 **학습 source에서의 consistency가 새로운 음악/음성 source로
전이되지 않은 것**이 실패 원인이다.

## 데이터와 누수 통제

학습 입력은 `output/dual_domain_stats_v1`에 미리 저장한 원본-mixture 통계와 아래 train
partition뿐이다.

- `channel_invariant_factorial_train_v1`
- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2/truth_train.csv`

Music-present 17,760행, Music source 8,396개, 검증된 clean→codec pair 1,600개다. sampling
mass는 `bank → Music label → generator → source` 순서로 균등화해 반복 codec이나 긴
source가 batch를 지배하지 않게 했다. `temporal_mixed_train_v2`의 전체 truth가 아니라
`truth_train.csv` ID만 통계에서 다시 골라 사용하며, partition guard가 모든 train
manifest를 검사한다.

checkpoint early stopping과 residual weight 선택은 다음 8개 허용 development의 Music
score `0.5 × mean + 0.5 × worst`만 사용한다.

- mixfake music dev, factorial dev, external mixed
- source-disjoint mixed / mixed-equal / music
- telephone mixed dev, codec mixed dev v4

두 KL seed의 단독 checkpoint maximin은 seed04 `0.82896`, seed05 `0.82322`다. 사전에
정한 규칙(두 번째 seed가 첫 번째보다 동등 이상일 때만 ensemble)에 따라 seed05는
ensemble에서 제외했다. authorized residual grid는 0, 1, 2.5, 3.75, 5, 7.5, 10, 15,
20%이며 seed04 20%가 `0.84311`로 선택됐다. 이 선택에 v5/v6 결과는 들어가지 않았다.

## RF/FR/FF/RR와 혼합 방식

`RF/FF`는 fake Music positive, `RR/FR`은 real Music negative다. 단일 component case에는
두 Music label이 함께 없으므로 case별 EER를 만들 수 없다. 대신 아래는 3.75% 후보의
평균 score이며, EER는 Music label 양쪽이 있는 Voice-real/Voice-fake slice로 보고한다.

| set | RR | FR | RF | FF | Voice-real EER | Voice-fake EER |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 0.1786 | 0.1947 | 0.5496 | 0.5268 | 0.2200 | 0.2867 |
| v5 | 0.2777 | 0.2660 | 0.6221 | 0.6191 | 0.2167 | 0.1833 |
| v6 diagnostic | 0.2661 | 0.2521 | 0.5477 | 0.6586 | 0.2833 | 0.2333 |

3.75% 후보의 혼합 방식별 Music EER는 다음과 같다.

| set | concurrent | partial overlap | sequential |
|---|---:|---:|---:|
| v4 | 0.2300 | 0.3200 | 0.2000 |
| v5 | 0.2000 | 0.1750 | 0.2250 |
| v6 diagnostic | 0.3250 | 0.2500 | 0.1000 |

## Codec별 결과와 과적합 위치

아래는 3.75% 후보의 codec별 Music EER다.

| set | clean | G.711 | G.722 | Opus-NB | G.711→Opus |
|---|---:|---:|---:|---:|---:|
| v4 | 0.08333 | 0.25000 | 0.10000 | 0.33333 | 0.38333 |
| v5 | 0.08333 | 0.20833 | 0.12500 | 0.25000 | 0.16667 |
| v6 diagnostic | 0.16667 | 0.20833 | 0.20833 | 0.41667 | 0.33333 |

20% dev-selected 후보에서 v4는 모든 channel이 좋아지지만, v5는 Opus와 transcode가
각각 anchor `0.20833/0.16667`에서 `0.25000/0.25000`으로 악화된다. 단독 expert는
v5 Opus/transcode가 `0.45833/0.47917`까지 무너졌다. v4에 포함된 generator/channel
조합의 개선을 보고 weight를 키우면 unseen source의 narrowband ranking을 크게
망가뜨리는 전형적인 domain shortcut이다. phone router로 이 현상을 감추기보다,
blind에서 양방향 개선된 기존 v51의 작은 고정 residual을 유지한다.

## 구현과 재현물

- 모델/loss: `src/channel_invariant_music_head.py`
- 안전한 logit ensemble/residual: `src/channel_invariant_music_inference.py`
- 학습: `scripts/train_channel_invariant_music_head.py`
- selection과 blind 분리 평가: `scripts/evaluate_channel_invariant_music_v52.py`
- seed 결과: `reports/channel_invariant_music_v52/kl_seed_20260904/`,
  `kl_seed_20260905/`
- 최종 표: `reports/channel_invariant_music_v52/kl_final_evaluation/`
- 테스트: `tests/test_channel_invariant_music_head.py`,
  `tests/test_channel_invariant_music_inference.py`

전용 head 관련 focused test 9개와 전체 누수 guard를 포함한 26개 test가 통과했다. 이
실험에서는 ZIP 생성, 커밋, push, 제출을 하지 않았고 `src/artifactnet_detector.py`도
수정하지 않았다.
