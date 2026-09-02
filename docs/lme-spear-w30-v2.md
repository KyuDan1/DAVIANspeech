# LME + SPEAR w30 v2 제출

## 결론

`lme_spear_v1.zip`의 실제 점수는 Total `0.7444743492`, ADS
`0.7172857143`, CPS `0.9891720635`로 LME와 SPEAR 단독 제출을 모두 넘었다.
따라서 이번 제출은 검증된 구조를 유지하고 **SPEAR의 File/Music 결합 비중만
`0.10`에서 `0.30`으로 변경**한다. Voice와 두 Presence 출력은 바꾸지 않는다.

현재 1위의 ADS `0.80354`와는 `0.0862543` 차이다. CPS 차이는 `0.007898`이지만
총점에서 CPS 가중치는 10%뿐이므로, 이번 단계의 우선순위는 계속 ADS다.

## 방법

기존 LME anchor는 다음 세 경로를 유지한다.

1. HTDemucs voice stem을 NII XLS-R-2B anti-deepfake로 판별한다.
2. HTDemucs music stem의 XLS-R와 원본 오디오의 ArtifactNet을 50:50으로 합친다.
3. 4초 XLS-R window는 max 대신 `logmeanexp(temperature=5)`로 집계한다.

SPEAR는 원본 오디오를 직접 처리하므로 source separation으로 생길 수 있는 생성
흔적의 손실과 separator artifact를 겪지 않는다. 얕은 layer-2의 RR/RF/FR/FF
joint head가 File을, generator-balanced mixed-music head가 Music을 보조한다.

```text
FILE_FAKE  = 0.70 × LME_anchor_file  + 0.30 × SPEAR_joint_file
MUSIC_FAKE = 0.70 × LME_anchor_music + 0.30 × SPEAR_music
VOICE_FAKE = LME_anchor_voice
Presence   = PANNs anchor 그대로
```

최종 Music을 다시 max/soft-OR로 File에 강제 반영하는 후보도 시험했지만 YuE
generator-disjoint audit에서 일관되지 않았다. 따라서 File용 joint expert와 Music용
expert를 독립적으로 단순 평균한다.

## 실제 제출로 확인된 출발점

| 제출 | ADS | CPS | Total |
| --- | ---: | ---: | ---: |
| LME v1 | 0.7138571 | 0.9893250 | 0.7414039 |
| SPEAR v15 | 0.7135317 | 0.9891721 | 0.7410958 |
| LME + SPEAR w10 | **0.7172857** | 0.9891721 | **0.7444743** |

두 변경은 완전히 가산적이지 않지만 결합 이득은 실제 비공개 평가에서 확인됐다.

## 다중 평가군 검증

Factorial은 제출 코드와 동일한 LME anchor를 8 GPU로 다시 추론한 결과다. 나머지
평가군은 서로 다른 원천과 생성기를 포함한다. 값은 ADS다.

| SPEAR 비중 | Factorial v2 LME | YuE audit | Competition v2 | Competition v3 |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.656632 | 0.666518 | 0.822729 | 0.746747 |
| 0.20 | 0.679758 | 0.729601 | 0.829909 | 0.763477 |
| **0.30** | **0.693714** | **0.762799** | **0.841529** | **0.782856** |

Factorial에서 w10→w20은 clean, MP3, OGG, noisy, telephone, stereo 모든 채널에서
비열등이었고, 동시 혼합 `+0.00156`, 부분 중첩 `+0.02400`, 순차 혼합
`+0.02622` ADS였다. w30은 네 전체 평가군에서 w20보다도 높았다.

로컬에서는 w40~w50이 더 높은 평가군도 있었지만 채택하지 않았다. 과거 제출에서
합성 평가셋에 강하게 맞춘 head와 router가 실제 비공개 평가에서 역전된 사례가
있기 때문에, 실제 양의 방향이 확인된 expert의 비중만 세 배로 올린 보수적인
선택이다.

## 연구 동향과의 대응

- MixFake는 mixed-source 환경에서 speech SSL의 semantic-centric 한계를 지적하고,
  분리 대신 원본 입력에서 frequency/texture 신호를 함께 쓰는 multi-stream prompt
  tuning을 제안한다: <https://arxiv.org/abs/2605.23201>
- Attention-based MoE는 서로 다른 inductive bias를 가진 expert를 입력별로 결합해
  SAFE 2025의 세 task에서 1위를 기록했다: <https://arxiv.org/abs/2509.17585>
- ASVspoof5 ParallelChain 시스템은 waveform, mel, vocoder augmentation으로 학습한
  여러 모델의 ensemble을 사용했다: <https://www.isca-archive.org/asvspoof_2024/tran24_asvspoof.html>
- 다만 ASVspoof5의 한 시스템은 split 간 domain gap에서 학습형 fusion이 악화되어
  단순 weighted average를 최종 사용했다: <https://www.isca-archive.org/asvspoof_2024/tran24_asvspoof.pdf>

현재 데이터 규모와 과거 역일반화를 고려하면, 이번 w30은 SOTA 연구의
complementary-expert 원칙을 따르되 overfit되기 쉬운 attention gate 대신 단순
평균을 사용한 단계다. 다음 큰 폭의 개선은 원본 audio의 시간축 SPEAR token과
frequency/texture prior를 함께 학습하고 generator-family 및 codec을 통째로 제외한
검증에서 통과시키는 multi-stream component head여야 한다.

## 재현 파일

- 정확한 LME shard: `reports/factorial_v2_lme_exact/anchor_shard_*.csv`
- weight 및 slice 결과: `reports/factorial_v2_lme_exact/spear_weight_grid.csv`
- 평가 코드: `scripts/evaluate_lme_spear_weight.py`
- 제출 디렉터리: `submit_lme_spear_w30_v2/`
- 제출 ZIP: `lme_spear_w30_v2.zip`
