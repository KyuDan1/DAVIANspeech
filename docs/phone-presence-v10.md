# Telephone-aware CPS refinement (v10)

## 결론

`dual_cps_v10`은 기존 v8의 ADS 출력을 유지하면서 전화채널의
`VOICE_PRESENT_PROB`만 보정하는 제출 후보이다. 보정은 원본 오디오를 분리하거나
재합성하지 않고, 전화 라우터와 이미 추출한 EAT latent 통계만 사용한다.

로컬 1,200개 union 재현에서 세 FAKE 확률 열은 v8과 값과 순위가 완전히
동일했다. Voice Presence AUC는 `0.997965 -> 0.999043`, Music Presence AUC는
`0.998730`으로 동일하여 CPS가 `0.998348 -> 0.998887`로 상승했다.

## 왜 이 변경을 했는가

기존 presence ensemble의 가장 큰 약점은 음악에 음성이 작게 묻힌 전화 음질
샘플이었다. 특히 -6 dB 이하의 동시 혼합에서 PANNs는 음성을 음악으로 덮어
판단하는 경향이 있었다. 전화 factorial 평가에서 Voice Presence AUC는
PANNs `0.956603`, EAT `0.986569`, 기존 ensemble `0.982047`이었다.

Presence 오판을 FAKE 확률의 gate로 사용하면 ADS까지 훼손할 수 있다. 따라서
이번 후보에서는 presence와 authenticity를 분리했다.

- 전화 보정은 `VOICE_PRESENT_PROB`에만 적용한다.
- `FILE_FAKE_PROB`, `VOICE_FAKE_PROB`, `MUSIC_FAKE_PROB`는 보정하지 않는다.
- `MUSIC_PRESENT_PROB`도 그대로 둔다.
- 각 파일은 독립적으로 라우팅하고, 다른 평가 파일의 통계는 사용하지 않는다.

## 학습 데이터와 누수 방지

전화 Voice Presence head의 학습 원천은 평가셋과 분리했다.

- Voice: `phone_router_voice_train_v1`의 EchoFake 등 독립 음성 원천
- Music: `multigen_music_presence_train_v1`의 SONICS 계열 독립 음악 원천
- 구성: music-only 400, voice-only 400, concurrent 600,
  partial-overlap 500, sequential 500
- SNR: -15, -10, -5, 0, 5, 10 dB
- 채널: 8 kHz resampling, PSTN bandpass, G.711, G.726, narrowband Opus

새 factorial train 2,400개와 기존 독립 전화 train을 합쳐 9,600개 latent
통계로 선형 head를 학습했다. `factorial_eval_1200_v2` dev/holdout과
`phone_factorial_1200_v1`은 학습 및 모델 선택에 넣지 않았다.

## 실험과 선택

처음 만든 전화 head를 0.5 비중으로 결합하면 강한 합성 전화셋에서는 좋았지만
다른 전화 코덱에서 calibration이 흔들렸다. 이를 기각하고, 여러 독립 평가
bank에서 동시에 퇴보하지 않는 고정 logit 비중 `0.10`을 선택했다.

| 평가 bank | 기존 Voice AUC | v10 Voice AUC | 변화 |
|---|---:|---:|---:|
| factorial dev | 0.998514 | 0.999257 | +0.000743 |
| factorial holdout | 0.998800 | 0.999143 | +0.000343 |
| 모든 telephone channel | 0.997989 | 0.999497 | +0.001508 |
| phone factorial | 0.982047 | 0.998906 | +0.016859 |

전화 라우터는 phone factorial 1,200개를 모두 전화로 인식했다. 일반 factorial
1,200개에서는 200개 telephone 변형과 false positive 1개만 라우팅했다.
보수적인 0.10 비중 덕분에 그 false positive를 포함한 전체 union에서도
Voice AUC가 상승했다.

## 구현

1. `TelephoneRouter`가 원본 오디오의 대역폭/코덱 특성으로 전화 여부를 판단한다.
2. EAT가 presence와 ADS용 latent 통계를 한 번에 계산한다.
3. 전화로 판단된 파일만 frozen linear Voice Presence head로 점수를 계산한다.
4. 기존 Voice Presence와 전화 head를 logit 공간에서 0.90:0.10으로 결합한다.
5. FAKE 확률은 이전 v8의 dual-domain ensemble 출력을 그대로 사용한다.

분리 모델을 추가하지 않았으므로 diffusion/소스 분리 과정에서 생성 artifact가
덮이는 문제도 없다. 모델 추가 용량은 약 74 KB이며 별도 추론 backbone도 없다.

## 검증 상태

- 관련 테스트: 14 passed
- 1,200개 end-to-end 앞단 + 후처리 로컬 환산: 약 26분 30초
- 과거 평가 서버 실행시간을 기준으로도 약 40분 수준으로 예상되어 60분 제한 내
- 패키지 최상위 구조: `model/`, `script.py`, `requirements.txt`
- 런타임 임시 EAT/SPEAR 통계는 최종 CSV 생성 뒤 삭제

로컬 CPS 수치는 방향성과 회귀 여부를 검증하는 값이며 실제 비공개 점수를
보장하지 않는다. 실제 개선 폭은 비공개 데이터의 전화채널 비율과 혼합 난이도에
따라 달라진다.
