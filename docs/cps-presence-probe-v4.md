# CPS 개선 및 error propagation 차단: presence probe v4

## 결론

`lme_spear_eat_presence_probe_v4.zip`은 현재 실제 최고점인
`lme_spear_v1`의 세 authenticity 확률을 그대로 유지하면서 presence 두 열만
개선한다. 즉 CPS 실험이 실패하더라도 새 presence 오판이 ADS로 전파되지 않는다.

- 기준 실제 점수: Total `0.7444743492`, ADS `0.7172857143`, CPS `0.9891720635`
- 제출 후보: `lme_spear_eat_presence_probe_v4.zip`
- SHA-256: `f254a37c3dad5761ac4c8fed64f5f3229bd265242df825a7a29efb3a9ffb92d4`
- ZIP 크기: 약 7.0 GiB
- 압축 해제 크기: 7,505,173,758 bytes
- ZIP 최상위 항목: `model/`, `script.py`, `requirements.txt`

실제 오디오 한 개를 전체 제출 엔트리포인트로 실행해 PANNs, HTDemucs,
XLS-R, ArtifactNet, EAT, SPEAR 로딩과 최종 CSV 생성을 확인했다. 이때 v4의
`FILE_FAKE_PROB`, `VOICE_FAKE_PROB`, `MUSIC_FAKE_PROB`는 현재 최고 패키지와
최대 절대 오차 `0.0`으로 일치했다. ZIP CRC와 8개 단위 테스트도 통과했다.

## 왜 presence와 authenticity를 분리했는가

기존 pipeline은 presence가 임계값을 넘은 component의 fake score만 File score에
반영하는 hard gate를 사용한다. 이 구조에서는 presence가 임계값 주변에서 조금만
틀려도 올바른 fake expert 전체가 탈락한다.

실제로 EAT를 추가한 뒤 File gate를 다시 계산했을 때 gate의 최적값이 데이터셋마다
달랐다. 같은 `0.6`도 어떤 재구성 셋에서는 File EER을 낮췄지만 factorial에서는
악화했다. 따라서 v4에서는 다음처럼 역할을 분리했다.

1. ADS는 검증된 `LME + SPEAR(weight=0.10)` 경로를 그대로 사용한다.
2. CPS만 PANNs, EAT AudioSet, EAT latent probe로 개선한다.
3. 새 presence 결과로 File score를 다시 gate하지 않는다.

향후 joint model이 충분히 검증되기 전까지 hard routing은 사용하지 않는다.

## Voice presence

원본 오디오의 시작·중간·끝 6초 view에서 EAT AudioSet evidence를 계산하고 기존
PANNs와 선형 결합했다.

```text
Voice presence = 0.65 * PANNs + 0.35 * EAT
```

EAT 비중을 0.00부터 1.00까지 훑었을 때 `0.35`가 5개 평가군의 최악 Voice AUC와
평균 Voice AUC를 함께 가장 안정적으로 개선했다. 학습한 Voice probe는 source
disjoint에서는 좋았지만 factorial holdout AUC가 `0.9339`였으므로 폐기했다.

## Music presence

Music은 다음 두 evidence를 사용한다.

```text
base music = 0.10 * PANNs + 0.90 * EAT AudioSet
probe music = sigmoid(EAT latent linear score / 5)
Music presence = 0.60 * base music + 0.40 * probe music
```

probe 입력은 source separation을 거치지 않은 원본 mixture의 EAT patch token이다.
시작·중간·끝 view 각각에서 mean, standard deviation, mean absolute temporal delta,
log Teager-Kaiser energy를 만든 뒤 view mean과 view max를 결합했다. 순차 혼합에서
음악이 한 view에만 존재하는 경우를 view max가 보존한다.

학습에는 voice-only 726개, 보호 데이터와 겹치는 6개를 제거한 music-only 474개,
그리고 mixed train 2,400개를 사용했다. eval identity와 겹치는 행은
`data_guard.py`로 학습 전에 차단했다. Fake/Real label은 presence 학습에 사용하지
않았으며, 목표는 오직 voice/music 존재 여부다.

## 결과

아래 가중치는 factorial development에서 결정한 뒤 고정했다. Holdout과 phone은
가중치 결정 후 한 번 평가했다.

| 평가군 | Voice AUC | 기존 Music AUC | v4 Music AUC | v4 CPS |
|---|---:|---:|---:|---:|
| factorial dev | 0.998514 | 0.996514 | 1.000000 | 0.999257 |
| factorial holdout | 0.998800 | 0.999657 | 1.000000 | 0.999400 |
| phone factorial | 0.982047 | 0.991691 | 1.000000 | 0.991023 |

학습 없는 PANNs+EAT 결합만으로도 full factorial CPS는
`0.994108 -> 0.997927`, phone factorial CPS는 `0.970331 -> 0.986869`로
상승했다. latent Music probe를 추가하면 위 표처럼 Music AUC가 더 개선됐다.

## 폐기한 실험

- 전화 데이터를 authenticity head 학습에 직접 크게 넣은 모델은 전화 dev를
  개선했지만 clean factorial ADS가 기존 3-seed ensemble `0.775247`에서
  `0.720961`로 하락해 폐기했다.
- 학습한 Voice presence probe는 source fingerprint에 민감해 폐기했다.
- 새 presence를 hard File gate에 전파하는 버전은 데이터셋별 방향이 일관되지 않아
  제출 후보에서 제외했다.

## 실제 제출에서 확인할 항목

v4의 실제 ADS는 최고 제출과 같아야 한다. CPS가 상승하면 presence 개선이 test에도
전이된 것이고, 예상과 달리 CPS가 하락하더라도 ADS는 보존된다. 이후에는 leaderboard
결과에 따라 Voice/Music presence 중 어느 쪽이 병목인지 진단 제출을 한 번만 사용한다.
