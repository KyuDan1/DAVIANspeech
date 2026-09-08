# v70: 긴 음성·음악 및 부분 생성 스트레스 테스트

## 범위와 역할

기존 개발셋 4,577개의 실제 길이는 최대 20.2초이고, 1,232개는 4초 미만이었다.
v70은 빠져 있던 긴 파일·짧은 가짜 구간의 실패 원인을 확인하는 **stress_eval**이다.
원천은 이미 개발용으로 분리해 둔 `codec_mixed_dev_v4`의 original voice/music다.
새로운 source/generator 일반화나 자연스러운 실제 통화 데이터라고 주장하지 않는다.

**학습, checkpoint 선택, ensemble weight/threshold/router fitting에 사용하지 않는다.**
사전에 고정된 모델을 비교하는 메커니즘 진단이다. 미지 원천의 최종 검증은 별도로 한다.

## 구성

| 축 | 조건 |
|---|---|
| 길이 | 10 / 30 / 60초 |
| 채널 | clean / G.711 μ-law / Opus narrowband 8 kbps |
| 혼합 | 동시, 음성→음악, 음악→음성, 2초 부분 가짜 음성, 2초 부분 가짜 음악 |
| 성분 진위 | RR / RF / FR / FF (첫 글자는 음성, 두 번째는 음악) |
| 순수 성분 | 진짜/가짜 음성, 진짜/가짜 음악 |
| 원천 그룹 | 10개; 각 그룹에 VR/VF/MR/MF pool별 원천 6개 |

계획 파일 수는 `10 × 3 × 3 × (5 × 4 + 4) = 2,160`이다.
같은 원천 그룹을 조건별로 재사용하므로 2,160개의 독립 원천 표본이 아니다.
신뢰구간을 계산한다면 source group 전체를 묶어 재표집해야 한다.

## 합성 방법과 한계

- waveform separation이나 생성 모델을 사용하지 않는다.
- 음성 원천을 4초 piece, 음악 원천을 10초 piece로 정리한다. 짧으면 반복하며,
  20 ms crossfade로 경계를 완화한다. 고정 piece 순서를 반복하여 60초 stream을 만든다.
- Real/Fake 모두 같은 piece 길이·stitch·gain 규칙을 적용한다. 그러나 원천 자체의
  길이에 따른 반복 빈도와 인공적인 연결은 남으므로 절대 정확도 해석에는 한계가 있다.
- sparse 조건에서는 실제 배경 stream에 2초만 가짜 성분을 삽입한다. Real 대조군에도
  같은 위치·길이·fade로 실제 성분을 삽입하여 편집 경계 자체가 정답이 되지 않게 한다.
- 삽입 위치는 seed/group/duration의 hash로 고정하며 label을 입력받지 않는다.
- 그룹별 SNR은 -10/0/+10 dB cycle로 고정한다. 같은 그룹의 counterfactual은 동일 SNR이다.
- 음성·음악을 먼저 합친 뒤 전체 waveform에 codec을 적용한다.
- 출력은 16 kHz mono FLAC/PCM16. 파일별 실제 header 길이와 저장 SHA를 기록한다.
- 성분과 fake 구간의 시간 범위를 truth에 기록한다. 이 정보는 정답 분석용이며
  모델 입력이나 router에 제공하지 않는다.

## 누수 및 원천 검사

구성 전에 parent source/speaker/group identity를 등록된 train, router_train,
training_validation 모두와 대조했다. plan 단계 overlap 0을 확인했다.
렌더링 때 raw source가 기존 source manifest의 SHA와 일치하는지 검사한다.
선택한 원천의 decoded PCM 중복도 확인한다. instrumental 의미는 기존 개발 소스의
스크리닝을 상속한다. 새로 음악의 보컬 유무가 완벽하게 검증됐다고 주장하지 않는다.

`RESERVED_SOURCE_IDS`는 그룹에 예약된 원천 목록이다. 짧은 출력에는 그 중 일부만
실제 사용될 수 있으므로, 모든 예약 소리가 모든 파일에 들어간다는 뜻은 아니다.

## 실행 상태

설정: `configs/long_component_stress_v70.yaml`.
계획: `reports/long_component_stress_v70_plan/`.
생성 위치: `data/eval/long_component_stress_v70/` (생성·전체 검증 완료).
렌더링/기존 predictor/adapter 관련 테스트 35개 통과.
전 2,160파일의 SHA/header/구간 label 검증을 통과했고 stress 역할에 등록했다.
길이별 각 720개, 합계 오디오 72,000초(20시간)다. 등록 후 repository data guard도
12개 train/router train manifest 모두 통과했다.
가중치를 먼저 고정한 EAT control/adapted 비교를 완료했다.
동결 기록은 `reports/long_component_stress_v70_comparison/frozen.json`에 저장한다.

## 고정된 두 모델의 결과

전체 합성 스트레스 ADS는 control 0.726841, adapted 0.737491이었다.
각 모델은 같은 GPU 4에서 동시에 실행됐고, 모델 로딩 후 2,160파일 추론·집계는
각각 약 129초였다. 이는 B200에서의 소형 연구 경로 시간이지, 제출 pipeline의
L4 시간 검증이나 co-location 속도 향상의 대조 실험이 아니다.

학습된 EAT의 조건별 EER (채널 3종 합산, 각 행 120개 파생 파일):

| 조건 | 길이 | File EER | Voice EER | Music EER |
|---|---:|---:|---:|---:|
| 동시 혼합 | 10초 | 19.44% | 20.00% | 21.67% |
| 동시 혼합 | 60초 | 17.22% | 21.67% | 20.00% |
| 음악→음성 순차 | 60초 | 16.67% | 6.67% | 23.33% |
| 음성→음악 순차 | 60초 | 13.89% | 10.00% | 26.67% |
| 음성 성분 중 2초만 fake | 10초 | 30.56% | **48.33%** | 21.67% |
| 음성 성분 중 2초만 fake | 60초 | 30.00% | **48.33%** | 13.33% |
| 음악 성분 중 2초만 fake | 10초 | 26.67% | 25.00% | **43.33%** |
| 음악 성분 중 2초만 fake | 60초 | 36.67% | 25.00% | **48.33%** |

현재 global readout은 **짧은 fake 구간의 존재 여부**를 놓치는 약점이 뚜렷하다.
이 bank에서는 단순히 길어지는 것보다 sparse 조작이 어렵다. 하지만 인공 반복,
10개 원천 그룹, 개발 부모 재사용이라는 제약이 있으므로 실제 test 구성이나
공식 점수 하락 원인을 이 결과만으로 역추정하지 않는다.

전화 채널을 빼도 남는 문제다. clean 60초 조건에서 sparse Voice EER은 50.0%,
sparse Music EER은 45.0%였다. 따라서 이 결과를 "전화 codec만 고치면 해결"이라고
해석할 수 없다. 짧은 조작 구간의 감지와 codec 강건성은 분리해서 다뤄야 한다.

추가로 삽입에 쓰인 20개 fake 구간의 혼합 전 파형을 다시 확인했다.
2초 구간의 최소 RMS는 Voice 0.05830, Music 0.02010이었다. `abs(sample)>1e-4`
비율의 최소값도 각각 94.93%, 99.37%였다. 따라서 전부 무음인 fake 구간을
잘못 삽입해서 나온 결과는 아니다. 이 수치는 가청성이나 semantic 판정을 대신하지 않는다.

이 bank에서 window/temperature/head/ensemble weight를 탐색하지 않는다.
다음 단계의 구간 감독·multi-instance 학습은 새 train 데이터에서 수행하고,
허용된 development에서 선택한다. v70 결과는 현 모델의 메커니즘 진단으로 남긴다.

전체 및 조건별 결과: `reports/long_component_stress_v70_comparison/`.
