# Dual-domain ADS + 독립형 presence v8

## 결론

`lme_spear_dual_presence_v8.zip`은 실제 최고 제출인 `lme_spear_v1`을 기준으로
두 가지를 동시에 추가한다.

1. 원본 mixture의 EAT와 SPEAR latent를 함께 보는 dual-domain authenticity
   ensemble로 ADS를 보강한다.
2. PANNs, EAT AudioSet evidence, EAT latent probe를 결합해 CPS를 보강한다.

presence 결과는 fake 확률이나 File score를 gate하지 않는다. 따라서 presence
오판이 ADS로 전파되는 기존 hard-routing 문제를 차단했다. ADS와 CPS는 마지막
CSV의 서로 다른 열에서 독립적으로 갱신된다.

## ADS 방법

source separation을 거치지 않은 원본 오디오에서 시작·중간·끝 view를 만든다.
각 view에 대해 EAT patch token과 SPEAR 13개 layer hidden state의 다음 통계를
계산한다.

- mean, standard deviation
- mean absolute temporal difference
- log Teager-Kaiser energy
- view mean과 view max

이 통계로 File, Voice, Music 및 `RR/RF/FR/FF` 조합을 함께 학습한 작은
task-specific attentive head를 사용했다. generator/source가 train과 dev에
겹치지 않도록 분할했고, MixFake 8,000개 train과 1,600개 dev를 추가했다.
서로 다른 seed 3개의 확률을 평균해 단일 seed 변동을 줄였다.

최종 점수는 검증된 LME+SPEAR anchor와 logit 공간에서 결합한다.

| 출력 | dual-domain 비중 |
|---|---:|
| File fake | 0.50 |
| Voice fake | 0.30 |
| Music fake | 0.50 |

가중치는 4개 development bank에서 한 bank라도 EER이 크게 악화되는 후보를
제외하는 방식으로 한 번 선택한 뒤 holdout, phone, YuE에는 고정했다.

## CPS 방법과 error propagation 차단

Voice presence는 `0.65 * PANNs + 0.35 * EAT`이다. Music presence는 먼저
`0.10 * PANNs + 0.90 * EAT`를 계산한 다음, 원본 mixture의 EAT latent music
probe를 0.40 비중으로 결합한다. Voice latent probe는 source-disjoint에서는
좋았지만 factorial holdout에서 일반화가 나빠 사용하지 않았다.

중요한 구현 원칙은 다음과 같다.

- `update_file_score=False`: 개선된 presence로 File fake를 다시 만들지 않는다.
- Voice/Music fake도 presence threshold로 지우거나 축소하지 않는다.
- 각 파일의 view만 묶어 처리하며 다른 평가 파일의 점수나 통계는 사용하지 않는다.

따라서 CPS 개선이 실패해도 그 오류가 ADS에 직접 전파되지 않는다.

## 고정 가중치 평가 결과

### ADS

| 평가군 | anchor ADS | v8 ADS | 변화 |
|---|---:|---:|---:|
| external mixed dev | 0.6448 | 0.8392 | +0.1943 |
| factorial dev | 0.6339 | 0.7871 | +0.1532 |
| source-disjoint mixed equal | 0.7257 | 0.8343 | +0.1087 |
| source-disjoint mixed | 0.6740 | 0.8370 | +0.1630 |
| factorial holdout | 0.6779 | 0.7272 | +0.0493 |
| phone factorial holdout | 0.6894 | 0.7401 | +0.0508 |
| locked YuE audit | 0.7409 | 0.8569 | +0.1160 |

factorial holdout에서는 File EER `0.3276→0.2953`, Voice EER
`0.2686→0.2400`, Music EER `0.3543→0.2571`로 세 항목이 모두 개선됐다.
phone holdout에서도 File `0.3083→0.2617`, Voice `0.1825→0.1425`, Music
`0.4000→0.3350`로 모두 개선됐다.

### CPS

| 평가군 | Voice AUC | Music AUC | CPS |
|---|---:|---:|---:|
| factorial dev | 0.998514 | 1.000000 | 0.999257 |
| factorial holdout | 0.998800 | 1.000000 | 0.999400 |
| phone factorial | 0.982047 | 1.000000 | 0.991023 |

전화 품질에서 Voice presence가 여전히 가장 큰 CPS 병목이다. 그러나 전화
학습 데이터를 authenticity head에 강하게 넣은 실험은 clean ADS를
`0.7752→0.7210`으로 떨어뜨렸기 때문에 제출 모델에는 넣지 않았다.

## 제출 코드 재현성과 실행 제한

1,200개 union을 제출 엔트리포인트 전체로 실행한 결과 로컬 GPU에서 817초
(13분 37초)가 걸렸다. 네 development bank의 EER/ADS는 오프라인 결과와
동일했다. offline prediction과의 최대 확률 차이는 File 0.00351, Voice
0.00246, Music 0.00301이었으나 순위 지표에는 차이가 없었다.

- Python 단위 테스트: 9개 통과
- 압축 해제 크기: 7,514,688,086 bytes
- 가장 큰 ZIP member: 2,387,980,808 bytes
- requirements: `onnxruntime-gpu==1.23.2`만 추가 설치
- 최상위 구조: `model/`, `script.py`, `requirements.txt`
- 인터넷 다운로드 없음
- 제출 파일: `lme_spear_dual_presence_v8.zip`
- SHA-256: `cff4e47f273d8dbe9771b36885f66cde7406be79897fef396e934d96bb34e8da`
- ZIP CRC 검사: 전체 69개 파일 통과, 중복 member 0개

실제 리더보드 성능은 분포 차이 때문에 로컬 수치로 보장할 수 없다. 특히 실제
CPS는 현재 `0.98917`이므로 local factorial처럼 전이되는지 제출 결과로 확인해야
한다. CPS가 1등 수준인 `0.99707`까지 오르더라도 총점 이득은 약 `0.00079`이고,
나머지 큰 격차는 ADS이므로 이후 우선순위는 여전히 ADS 오류군 분석이다.
