# MixFake multi-stream prompt MoE v101

## 결론

v101은 공식 최고 제출인 `best_eat_music_v83.zip`을 그대로 anchor로 두고,
원본 혼합 오디오만 입력받는 작은 WPT/Spectra prompt 전문가를 logit 공간에서
보수적으로 결합한다. Voice/Music stem을 새로 만들지 않으므로 분리기가 생성한
artifact에 의존하지 않는다. Presence 두 열은 변경하지 않는다.

고정 구성은 다음과 같다.

- 하나의 joint 전문가, 3 temporal views: File weight 0.40, Voice weight 0.12
- exact-codec로 추가 학습한 Music 전문가, 1 view: Music weight 0.28
- 최종 File 점수와 `1-(1-Voice)(1-Music)`을 weight 0.15로 결합
- 모든 weight는 development에서 고정하며 locked split에서는 선택하지 않음

## 왜 이 구조인가

공식 probe로 역산한 현재 v83의 EER은 File 0.2317, Voice 0.1756,
Music 0.3171이다. Music이 가장 어렵지만 File 가중치가 0.5로 가장 크므로 두
축을 동시에 개선해야 한다. 큰 모델을 교체하면 이전 공식 ablation처럼 쉽게
회귀하므로, 검증된 anchor를 유지하는 residual MoE를 선택했다.

joint 모델은 Voice 0.2, Music 0.3, File 0.5의 대회 ADS 가중치로 학습했고,
clean/telephone paired consistency를 함께 최적화했다. Music 전문가는 G.711,
G.722, narrow-band Opus, G.711→Opus를 균형 있게 렌더링한 별도 4-codec bank로
추가 학습했다.

## 개발셋 결과

아래 development 수치는 모두 source-disjoint 또는 기존 고정 development이며,
이 결과만으로 구조와 weight를 확정했다.

| 평가축 | 지표 | anchor | v101 고정 결합 |
|---|---:|---:|---:|
| v93 clean 320 | File EER | 0.0750 | 0.0396 |
|  | Voice EER | 0.0875 | 0.0688 |
|  | Music EER | 0.0688 | 0.0438 |
| v93 Opus 320 | File EER | 0.2896 | 0.2479 |
|  | Voice EER | 0.2750 | 0.2250 |
|  | Music EER | 0.3063 | 0.2625 |
| codec/layout 600 | File EER | 0.2333 | 0.2067 |
|  | Voice EER | 0.2033 | 0.1967 |
|  | Music EER | 0.2133 | 0.2033 |
| broad 4,577 | File EER | 0.1689 | 0.1533 |
| broad 4,016 voice-present | Voice EER | 0.1950 | 0.1815 |
| broad 4,261 music-present | Music EER | 0.1291 | 0.1202 |
| broad | proxy ADS | 0.8379 | 0.8510 |

각 축의 난이도와 anchor 구현이 달라 절대값을 공식 점수로 간주하면 안 된다.
중요한 관찰은 같은 고정 결합이 clean, 여러 코덱, 전화품질, 넓은 다중 corpus에서
모두 전체 EER을 낮췄다는 점이다.

## Frozen locked one-shot

구조와 weight를 고정한 뒤, train/development와 voice identity 및 music-song group이
겹치지 않는 locked 480개를 딱 한 번 평가했다. 결과를 본 뒤에는 어떤 weight도
다시 선택하지 않는다.

| 지표 | v83 anchor | v101 | 변화 |
|---|---:|---:|---:|
| File EER | 0.0500 | 0.0333 | -0.0167 |
| Voice EER | 0.0583 | 0.0417 | -0.0167 |
| Music EER | 0.1083 | 0.0750 | -0.0333 |
| ADS | 0.9308 | 0.9525 | +0.0217 |

lockbox가 공식셋보다 쉬우므로 ADS 0.9525를 공식 예상치로 해석해서는 안 된다.
다만 세 EER이 동시에 개선되어, development weight selection이 unseen identity와
unseen song group에도 일반화했다는 강한 통과 신호다.

## Music 후보 선택

두 exact-codec Music seed를 비교했다. random-prompt seed33은 일부 혼합 slice에
강했지만, paper-like checkpoint에서 warm-start한 seed35가 네 축 중 세 축에서
더 안정적이었고 최적 weight도 0.23~0.30에 모였다. 따라서 seed35와 0.28을
고정했다.

| 평가축 | anchor | seed35 최적 결합 |
|---|---:|---:|
| v93 clean | 0.0688 | 0.0438 |
| v93 Opus | 0.3063 | 0.2563 |
| codec/layout 600 | 0.2133 | 0.2033 |
| broad music-present 4,261 | 0.1291 | 0.1183 |

## 해석과 한계

- 공식 점수의 주 병목은 CPS가 아니라 ADS다. CPS 0.9893을 1.0까지 올려도 총점
  증가는 약 0.0011뿐이다.
- local improvement를 공식셋에 그대로 대입할 수 없다. 현재 후보의 현실적인
  공식 총점 예상 범위는 약 0.79~0.81이다.
- broad audit에서 일부 concurrent/music-only slice는 joint File 결합이 악화됐다.
  그러나 기존 router 계열의 공식 회귀를 고려해, 이번 후보에는 복잡한 hard
  routing을 추가하지 않았다.
- 사용자 Suno 13곡에서 기존 파이프라인은 File/Music fake를 13/13 검출하지만
  singing Voice presence는 0/13만 0.5 이상이었다. 공식 CPS는 이미 높지만 향후
  vocal-presence 전용 개선이 필요한 실제 오류 유형이다.

## 재현 산출물

- joint checkpoint: `reports/mixfake_multistream_v97/seed34_joint_robust/multistream_prompt.pt`
- codec Music checkpoint: `reports/mixfake_multistream_v96/seed35_music_exactcodec_paperwarm/multistream_prompt.pt`
- inference: `src/mixfake_multistream_file_inference_v99.py`
- package builder: `scripts/build_mixfake_joint_music_moe_v101.py`
- one-shot lockbox report: `reports/mixfake_joint_music_moe_v101/locked_one_shot/report.json`
- one-shot lockbox runner: `scripts/run_locked_mixfake_candidate_v101.py`

locked 결과는 구조와 weight를 완전히 고정한 뒤 한 번만 기록한다. 그 결과를 보고
weight를 다시 고르는 것은 금지한다.
