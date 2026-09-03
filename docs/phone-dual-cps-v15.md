# Phone-only dual-domain ADS + decoupled CPS v15

## 결론

`phone_dual_cps_v15`은 실제 최고점인 `lme_spear_v1`의 비전화 ADS 경로와 모든
Voice fake 점수를 그대로 유지한다. 파일 내부의 대역폭·코덱 특징만으로 전화
채널이라고 판단된 경우에만, 원본 mixture EAT+SPEAR dual-domain ensemble을
`FILE_FAKE_PROB`와 `MUSIC_FAKE_PROB`에 결합한다. Presence는 v13의 독립 CPS
보정을 사용하며 fake 확률의 gate로 쓰지 않는다.

실제 최고 제출은 Total `0.7444743492`, ADS `0.7172857143`, CPS
`0.9891720635`이다. 아래 수치는 비공개 점수의 보장이 아니라, 제출 전 방향성과
회귀를 차단하기 위한 source-disjoint 로컬 결과이다.

## 전화 평가 프로토콜

`phone_factorial_1200_v1`은 서로 다른 원천의 voice-only 100개, music-only
100개, mixed 100개를 다음 네 전화채널로 각각 변환한 1,200개 고정 audit이다.

- 8 kHz resampling
- G.711 mu-law
- G.726 24 kbit/s
- narrowband Opus 8 kHz

원본 300개가 네 채널에 반복되므로 유효 source 수는 300개이고, 이를 1,200개의
독립 표본처럼 취급하지 않는다. 이 bank는 dual-domain head 학습이나 checkpoint
선택에 사용하지 않았다.

라우터는 phone factorial 1,200개를 모두 전화로 인식했다. 일반 factorial
1,200개에서는 의도한 telephone 200개와 false positive 한 개만 라우팅했다.

## 오류 분해

기존 10% SPEAR 결합의 phone factorial 결과는 다음과 같다.

| 범위 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| 전체 전화 | 0.3083 | 0.1825 | 0.4000 | 0.6894 |
| G.711 | 0.2469 | 0.0900 | 0.3400 | 0.7566 |
| G.726 | 0.2303 | 0.0900 | 0.3500 | 0.7619 |
| Opus NB | 0.4497 | 0.2600 | 0.3900 | 0.6061 |
| 8 kHz resample | 0.2469 | 0.0500 | 0.4200 | 0.7406 |
| mixed only | 0.4000 | 0.2350 | 0.4750 | 0.6105 |

Opus에서는 real voice+real music의 File fake 중앙값도 `0.9775`까지 상승했다.
따라서 전화 여부 자체보다 코덱을 fake artifact로 오인하는 현상과 mixed Music
분리가 핵심 병목이다.

## 기각한 실험

### Paired codec score debias

같은 원본의 clean/전화 SPEAR 임베딩 쌍에서 `clean logit - phone logit`만
회귀하는 label-free 보정기를 학습했다. 독립 phone factorial에서 ADS는
`0.68936 -> 0.69282`, File EER은 `0.3083 -> 0.3059`, Music EER은
`0.4000 -> 0.3925`로 개선됐다. 그러나 G.711 Music과 music-only EER은 각각
`0.340 -> 0.350`, `0.335 -> 0.345`로 악화되어 제출에는 넣지 않았다.

### Phone-augmented MERT/SPEAR heads

Frozen MERT embedding에 학습한 세 종류 linear head의 phone ADS는
`0.5961~0.6056`이었다. channel/component consistency loss도 개선하지 못했다.
전화 증강을 더 강하게 넣은 단일 dual-domain head도 phone ADS `0.6916`으로,
기존 세 seed 범용 dual-domain ensemble `0.7155`보다 낮았다. 전화 데이터에
직접 맞춘 head보다 여러 원천에서 학습한 범용 representation의 전이가 나았다.

## 채택한 방법

1. LME+SPEAR 실제 최고 경로를 먼저 실행한다.
2. EAT presence 단계에서 전화 ID를 함께 기록한다.
3. EAT와 SPEAR의 원본 mixture start/middle/end latent statistics를 계산한다.
4. 세 seed dual-domain ensemble을 실행한다.
5. 전화 ID의 File과 Music에만 dual score를 logit 공간에서 50% 결합한다.
6. 비전화 행과 모든 Voice fake 점수는 수정하지 않는다.
7. v13 EAT/telephone presence는 CPS 두 열에만 반영한다.

소스 분리 출력은 dual expert에 입력하지 않으므로 분리 모델이 생성 artifact를
덮는 문제를 피한다. 기존 anchor 분리는 유지해 실제 리더보드에서 검증된 음성
성능도 보존한다.

## 고정 평가 결과

| 평가군 | 기존 ADS | phone-only dual | 변화 |
|---|---:|---:|---:|
| factorial holdout 전체 | 0.67618 | 0.68343 | +0.00725 |
| factorial holdout 전화 subset | 0.64054 | 0.75572 | +0.11519 |
| phone factorial 전체 | 0.68936 | 0.73214 | +0.04279 |

factorial holdout의 전화 subset에서 dual Voice는 EER `0.2374 -> 0.2713`으로
악화됐지만 다른 전화 audit에서는 개선됐다. 방향이 일관되지 않아 Voice는
anchor 그대로 유지했다. File/Music만 결합하면 비전화 1,000개는 정확히
동일하면서 전체 holdout도 개선됐다.

## 제출 검증

- 2개 파일 전체 entrypoint smoke: 통과
- 라우터 결과: phone 1, non-phone 1
- 1,200행 routing 회귀검사: non-phone 최대 변화 `0.0`, Voice 최대 변화 `0.0`
- Python tests: 35 passed
- ZIP root: `model/`, `script.py`, `requirements.txt`만 존재
- ZIP 크기: 7,514,789,052 bytes
- 압축 해제 크기: 7,514,771,952 bytes
- 최대 member: 2,387,980,808 bytes
- 중복 member: 0
- CRC: 전체 통과
- SHA-256: `ffb79be83158125ae04452cc2bd47beeff67ca88dc2bbb153b7a073f8e7aa8bc`
- 파일: `phone_dual_cps_v15.zip`

실제 test의 전화 비율이 낮으면 ADS 개선 폭도 작다. 따라서 이 후보는 전화
가설의 검증용이며, 실제 목표 ADS `0.80354` 달성 여부는 리더보드 결과로만
판단한다.
