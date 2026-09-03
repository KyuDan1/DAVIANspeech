# Routed long-horizon music v27

2026-09-03 기준. v27은 실제 리더보드 최고점인
`channel_invariant_moe_v18.zip`(Total `0.7616379312`, ADS `0.7363412698`,
CPS `0.9893078836`)에 **routed long-horizon Music expert 한 단계만** 추가한
통제 제출이다. 로컬 작업 폴더는 후속 실험으로 변할 수 있으므로, builder가 실제 제출
ZIP을 직접 풀어서 base로 사용한다
(`scripts/build_routed_long_horizon_v27_submission.py`).

## 결론: router와 MoE를 같이 쓰되 역할을 분리한다

모든 파일에서 하나의 expert만 고르는 hard router는 채택하지 않았다. 오디오 형태나
두 모델의 불일치로 expert를 고르면 development와 holdout에서 최적 방향이 바뀌었다.
일반 오디오에서는 average-risk와 Group-DRO head의 logit MoE가 가장 안정적이었다.

반면 전화 채널은 독립적인 narrow-band router의 의미가 명확하고 실제 통신환경이라는
대회 배경과도 일치한다. 따라서 router는 **전화 도메인만 판정**하고, authenticity
결정은 해당 도메인에서 검증된 expert가 담당한다.

```text
non-phone: 0.4 * average-risk + 0.6 * Group-DRO  (logit MoE)
phone:     1.0 * average-risk                    (hard domain route)

Music: old Music와 expert를 logit 0.4 결합
File:  Music Presence >= 0.7일 때만 old File과 expert를 logit 0.2 결합
Voice/CPS: 변경 없음
```

새 head는 separator 출력이 아니라 원본 오디오의 EAT/SPEAR start/middle/end 통계를
사용한다. 이 통계는 v18이 이미 계산하므로 backbone pass와 source separation이 추가되지
않는다.

## router ablation

아래는 v18을 정확히 재구성하고 동일한 최종 결합을 적용한 결과다. EER은 낮을수록,
ADS는 높을수록 좋다.

| 구성 | Factorial Music EER | Phone Music EER | YuE Music EER | Factorial+Phone ADS |
|---|---:|---:|---:|---:|
| v18 | 0.2743 | 0.3450 | 0.1935 | 0.72218 |
| 고정 두-head MoE | 0.2229 | 0.1975 | **0.1613** | 0.79176 |
| 전화-domain route + MoE | **0.2229** | **0.1750** | **0.1613** | **0.79346** |

전화 내부를 다시 Music-only/Mixed로 hard route하면 Phone ADS가 `0.81314`에서
`0.807~0.808`로 하락했다. 비전화 파일을 Voice Presence로 hard route해도 Factorial과
YuE가 열화했다. 즉 현재 근거로는 전화 여부까지만 routing하고, 그 안에서 다시 잘게
분기하지 않는 것이 맞다.

## MERT temporal 보조 expert 실험

기존 SOFIA-MERT encoder pass에서 13개 layer별 start/middle/end mean/std를 저장하고,
평가 데이터를 학습에 넣지 않은 채 train 역할 데이터만으로 선형 Music head를 학습했다.

| MERT feature | 7개 dev 평균 Music EER | 최악 dev EER | Phone audit | YuE audit |
|---|---:|---:|---:|---:|
| 시간 분산만 | 약 0.411 | 0.465 | 0.488 | 0.339 |
| mean+분산, Group-DRO | **0.214** | **0.275** | **0.195** | **0.161** |

시간 분산만으로는 거의 무작위여서 “AI 음악은 리듬이 전형적일 것”이라는 가설을
지지하지 않았다. mean+분산 MERT를 기존 expert에 섞으면 개별 Factorial/YuE는
좋아졌지만 Factorial+Phone을 한 번에 정렬한 ADS는 `0.79346 → 0.79085`로 낮아졌다.
전화에서 MERT를 끄는 route까지 적용한 값이다. 서로 다른 corpus의 score calibration이
어긋난다는 뜻이므로 v27에는 MERT temporal head를 넣지 않았다.

Leave-one-generator-out도 Brev/Suno에는 견고했지만 Udio/Mubert에는 실패했다. 따라서
Suno 적중률만 보고 이 head를 제출하는 것은 일반성 목표와 맞지 않는다.

## 단일변수 보장과 검증

- base: 실제 성공 제출물 `channel_invariant_moe_v18.zip`
- 추가 파일: `model/src/long_horizon_music_inference.py`, 1.4MB
  `model/long-horizon-music/ensemble.npz`
- 변경 파일: `script.py`만 변경
- 3파일 GPU end-to-end smoke 성공, 전화 route `1/3`
- 결과 5개 확률은 모두 finite이며 `[0, 1]` 범위
- 임시 EAT/SPEAR/telephone 파일은 실행 후 제거됨
- 전체 테스트: `69 passed`
- ZIP: `7,053,716,866` bytes, 압축 해제 `7,909,203,346` bytes
- ZIP member `110`, 중복 `0`, CRC 오류 `0`, 최대 member `2,387,980,808` bytes
- SHA-256: `7b4b6d41230c5ce2dd8091c16b7a1307e73454d86cd1d06f46ddacb562bcd71a`

실제 리더보드에서는 v18과의 차이가 이 expert 하나뿐이므로, 점수 변화가 Music/File
long-horizon route의 효과인지 직접 판정할 수 있다. 로컬 개선은 실제 0.8을 보장하지
않으며, 실제 점수가 나온 뒤 Music-only와 File 결합을 각각 probe하는 것이 다음 순서다.
