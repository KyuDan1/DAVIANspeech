# CPS-only v13 leaderboard ablation

## 목적

`cps_v13`은 실제 최고 제출 `lme_spear_v1`의 ADS 경로를 한 글자도 바꾸지 않고,
두 presence 확률만 개선한다. dual-domain ADS까지 동시에 바꾸는 v10과 달리 실제
리더보드에서 CPS 변화의 인과를 정확히 확인하기 위한 단일 ablation이다.

## ADS 불변 증거

실제 최고 패키지와 다음 파일의 SHA-256이 동일하다.

- `pipeline.py`: `5247ae12588f93bf80d6d4c1a1d6e12fe088b9f78dddaf1bd7315b7de967f437`
- `anchor_spear_fusion.py`: `29a9d71e0bf782908bacfd614b6f50aebd81a9b199ff03c77beb00b7f9a67522`

1,200개 전체 end-to-end 재실행에서도 File/Voice/Music fake 확률은 기존
LME+SPEAR 결과와 최대 절대 차이 `0.0`, Spearman 상관 `1.0`이었다. 따라서 실제
ADS도 기존 최고값 `0.7172857143`을 재현해야 한다.

## CPS 결과

동일한 1,200개 union에서 다음 결과를 얻었다.

| 지표 | 실제 최고 ADS 경로 | CPS-only v13 |
|---|---:|---:|
| Voice Presence AUC | 0.992400 | 0.999043 |
| Music Presence AUC | 0.984365 | 0.998730 |
| CPS | 0.988383 | 0.998887 |

이 로컬 baseline CPS `0.988383`은 실제 baseline `0.989172`와 절대 차이
`0.000789`로 가까웠다. 비공개 분포가 완전히 같다고 가정할 수는 없지만, 기존
평가셋 중 leaderboard와 가장 잘 정렬된 지표다.

## 방법

- Voice: PANNs 0.65 + EAT AudioSet 0.35
- Music: PANNs 0.10 + EAT AudioSet 0.90, 이후 EAT latent probe 0.40
- 전화채널 Voice: 기존 Voice presence와 telephone head를 logit 공간에서
  0.90:0.10으로 결합
- presence는 FAKE 확률이나 File 점수의 gate로 사용하지 않음
- 분리 없이 원본 mixture만 사용

전화 라우터는 union 1,200개 중 68개만 보정했다. 67개는 의도한 telephone
변형이고 1개는 false positive였지만, 전체 Voice AUC도 상승했다.

## 실행 검증

- 1,200개 LME anchor 앞단: 약 17분 37초
- EAT presence + telephone head + SPEAR: 약 2분 10초
- 관련 단위 테스트: 14 passed
- 평가 서버의 기존 LME+SPEAR 실측 35분에 후처리 증가분을 더해도 60분 이내

실제 CPS가 목표 `0.99707`에 도달해도 총점 상승은 약 `0.00079`다. 이 제출은
CPS와 error propagation을 정리하기 위한 것으로, 0.8점 달성의 주된 다음 과제는
File/Music EER 개선이다.
