# v47: component-query MHFA와 monotone File compositor

## 목표와 결론

2026-09-04 리더보드 1위는 Total `0.837980`, ADS `0.821160`, CPS
`0.98932`다. 팀 최고 v18은 Total `0.7616379312`, ADS `0.7363412698`,
CPS `0.9893078836`이다. 같은 CPS에서 1위를 넘으려면 ADS가
`0.821164`보다 커야 하므로, 가중 EER을 `0.263659 -> 0.178836` 아래로 약
32% 줄여야 한다.

v47은 v44 위에 두 가지를 더한다.

1. EAT patch와 SPEAR temporal bin을 함께 읽는 separation-free
   component-query MHFA를 File `2.5%`, Music `5%`만 결합한다.
2. 최종 Voice/Music fake 확률의 noisy-OR를 File에 logit `30%` 결합한다.

세 잠금 평가 ADS는 `0.77582 / 0.82636 / 0.86522`였다. 평균은
`0.82246`이지만 실제 비공개 test의 구성과 generator가 다르므로 이를 공식 예상
ADS로 해석하지 않는다. 공식 hidden 정렬은 먼저 v44, 그 다음 v47 한 번씩만
제출해 확인해야 한다.

## 실제 제출에서 확인한 원칙

| 제출 | 변경 | ADS | v18 대비 |
|---|---|---:|---:|
| v18 | channel/component invariant 5% | **0.736341** | 0 |
| v19 | 유사 seed 3개, 10% | 0.734214 | -0.002127 |
| v32 | File attention 20%, phone route 25% | 0.723413 | -0.012929 |
| v33 | v32 + Music residual 50% | 0.714841 | -0.021500 |
| inv4-w30 | invariant 4-head 30% | 0.723976 | -0.012365 |

따라서 새 representation의 저비율 residual만 허용했다. hard audio-type/phone
router와 큰 expert weight는 쓰지 않았다. File의 30% 보정은 새 모델 30%가 아니라,
대회 정의 `File fake = Voice fake OR Music fake`를 단조 확률식으로 반영한 것이다.

## 오류 셀 분석

v44/v45 locked 예측을 File/Voice/Music 진위, layout, channel, generator로 다시
분해했다. 핵심 병목은 음악 단독이 아니라 **동시에 겹친 fake voice + real
music의 File 판정**이었다.

| 조건 | 기존/patch 결과 |
|---|---:|
| factorial concurrent VF+MR File EER | v18 `0.480`, v45 `0.520` |
| phone simultaneous VF+MR File EER | v18/v45 `0.320` |
| phone simultaneous VR+MF File EER | v18 `0.370` -> v45 `0.200` |
| factorial layout ADS | concurrent `0.718`, partial `0.7445`, sequential `0.830` |
| phone Opus-NB v45 File/Voice/Music EER | `0.337/0.240/0.070` |

EAT patch는 fake music 증거에는 강하지만 fake voice 증거를 File로 보내는 데 약했다.
따라서 Music expert의 비중만 키우지 않고 component별 query, mixed 4-cell 보조
분류, noisy-OR File compositor를 사용했다. 상세 재현 결과는
`reports/eat_patch_graph_v1/cell_audit_v1/`에 있다.

## 모델

source separation 없이 원본 mixture만 사용한다. 별도 backbone은 추가하지 않고
v44가 이미 계산하는 특징을 재사용한다.

- EAT: view 3개, layer `[1,3,5,7,9,11]`, 시간 node 38개와 주파수 node 8개
- SPEAR: view 3개, bin 8개, layer 4개, 각 bin의 4개 통계
- SPEAR에는 bin 방향을 보존하는 signed 1차/2차 차분을 추가
- Voice fake, Music fake, File fake, Voice presence, Music presence의 5개 query
- query별 SSL layer weight와 8-head attentive mean/std pooling
- token-local SwiGLU 두 층; 전체 token self-attention을 쓰지 않아 추론비용 제한
- 학습 파라미터 `627,291`

학습 목표는 presence가 있는 component에만 적용하는 Voice/Music BCE, File BCE,
positive-negative rank loss, RR/RF/FR/FF 4-cell 보조 loss, presence 보조 loss다.
File/Music/Voice가 동시에 같은 위치를 보도록 강제하지 않고 EAT와 SPEAR stream을
각각 pooling한 뒤 결합한다. 두 encoder의 6초/10초 grid가 다르기 때문이다.

관련 연구에서 가져온 핵심은 unified EAT+XLS-R layer fusion, MHFA attentive
statistics, partial fake의 signed temporal difference, 4-cell factorial mixing이다.

- <https://arxiv.org/html/2608.29021>
- <https://arxiv.org/html/2512.08319>
- <https://arxiv.org/html/2507.15101>
- <https://arxiv.org/html/2605.23201>

## 데이터 격리

학습 18,360개는 다음 train partition만 사용했다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2` train split
- `channel_invariant_factorial_train_v1`

각 corpus와 Voice/Music 진위·존재 cell, 가능한 generator/source를 균형화했다.
checkpoint는 source/generator-disjoint dev 7개에서 File:Music 공식 비중
`0.5:0.3`의 평균과 최악 domain을 반씩 반영해 선택했다. factorial holdout,
phone factorial, YuE, Suno는 학습과 epoch 선택에 사용하지 않았다.

선택 epoch 34의 dev 결과:

| 지표 | 값 |
|---|---:|
| 7-domain 평균 ADS | `0.824244` |
| 최악 ADS | `0.759091` |
| 평균 File EER | `0.167284` |
| 평균 Music EER | `0.114158` |

## 잠금 평가와 ablation

먼저 exact v18에 v44 patch File/Music 5%를 적용하고, component-query를 File
2.5%/Music 5% 적용했다. Voice와 두 Presence는 exact v18 그대로다.

| 평가 | v18 ADS | v44 | + query(v46) | + OR30(v47) |
|---|---:|---:|---:|---:|
| factorial holdout | 0.745506 | 0.757896 | 0.772000 | **0.775818** |
| phone factorial | 0.733857 | 0.790929 | 0.818429 | **0.826357** |
| YuE cross-component | 0.828448 | 0.849252 | 0.849252 | **0.865216** |

v47의 최종 EER:

| 평가 | File EER | Voice EER | Music EER |
|---|---:|---:|---:|
| factorial | 0.23236 | 0.22286 | 0.21143 |
| phone | 0.17829 | 0.16750 | 0.17000 |
| YuE | 0.12228 | 0.10208 | 0.17742 |

OR30은 hard cell도 개선했다.

| VF+MR File contrast | query까지 | OR30 |
|---|---:|---:|
| factorial concurrent | 0.520 | **0.440** |
| phone simultaneous | 0.310 | **0.220** |
| YuE concurrent | 0.375 | **0.250** |

아직 목표인 factorial File `<=0.15`에는 미달한다. 다음 학습은 Voice용 XLS-R
evidence를 File compositor가 직접 보게 하되, RR/RF/FR/FF와 concurrent/partial,
Opus-NB를 같은 질량으로 구성해야 한다.

## 구현과 배포 검증

- 모델: `src/component_query_mhfa.py`
- 추론: `src/component_query_mhfa_inference.py`
- 학습: `scripts/train_component_query_mhfa.py`
- 잠금 채점: `scripts/score_component_query_mhfa.py`
- 빌더: `scripts/build_component_query_v46_submission.py`
- 전체 테스트: `136 passed`
- v46 Suno 13파일 전체 CUDA entrypoint 성공, File/Music `13/13 > 0.5`
- v47 Suno 3파일 CUDA entrypoint 성공, 임시 EAT/SPEAR cache 모두 삭제

최종 v47 archive:

- `component_query_or_v47.zip`
- 압축/해제 크기: `7,913,418,225 / 7,913,394,695` bytes
- entry 117개, 중복 0, unsafe path 0, CRC 오류 0
- 최상위는 `model/`, `script.py`, `requirements.txt`만 존재
- SHA-256:
  `0ddb1d9d3d21def4f2e3b761e074f52145db45caebfe5c2198976a0af2ce25ed`
- requirements: `onnxruntime-gpu==1.23.2`

v44 패키지의 SPEAR helper에는 temporal-bin export가 없었다. 첫 smoke에서 이를
발견했고, 빌더가 최신 `anchor_spear_stats_fusion.py`, `spear_detector.py`,
`spear_temporal_bins.py`를 반드시 포함하도록 수정했다.

공식 제출 순서는 다음과 같이 잠근다.

1. `eat_patch_graph_v44.zip`: 새 EAT patch representation의 hidden 방향 확인
2. v44가 상승하면 `component_query_or_v47.zip`: query와 논리적 File 보정 확인
3. 둘 다 상승한 뒤에만 WPT 5% Voice/File residual을 검토
