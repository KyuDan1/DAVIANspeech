# 실험 기록

## 리더보드 결과 역산

| 버전 | ADS | CPS | Score | 가중 EER (`1-ADS`) |
| --- | ---: | ---: | ---: | ---: |
| 이전 | 0.68653 | 0.98932 | 0.71681 | 0.31347 |
| XLS-R 75% + SONICS 25%, gate 0.3 | 0.69010 | 0.98917 | 0.72001 | 0.30990 |

ADS만으로 File/Voice/Music EER 각각을 복원할 수는 없다. 알 수 있는 값은 다음
가중합뿐이다.

```text
0.5 × File EER + 0.2 × Voice EER + 0.3 × Music EER = 0.30990
```

이전 제출보다 가중 EER이 `0.00357` 감소해 방향은 맞았지만 개선 폭은 작았다.

## competition_v2

### 구성

- 총 1,200개
- 원본 380개: 음성 200, 의미쌍 음악 100, 혼합 80
- 채널: clean FLAC, 64kbps MP3, 전화 대역 WAV, 일부 15dB 잡음 OGG
- 동일 원본의 모든 변형을 같은 calibration/validation/holdout split에 배치

### 현재 제출 방식 재평가

| Split | ADS | File EER | Voice EER | Music EER |
| --- | ---: | ---: | ---: | ---: |
| calibration | 0.80162 | 0.16986 | 0.03030 | 0.35798 |
| validation | 0.79886 | 0.13930 | 0.01104 | 0.43095 |
| holdout | 0.79487 | 0.17657 | 0.05790 | 0.35088 |
| 전체 | 0.80177 | 0.16330 | 0.03624 | 0.36444 |

로컬 ADS가 리더보드보다 약 0.11 높아 기존 데이터가 실제 평가 분포보다 쉽다는
것을 확인했다.

### 기각한 방법

파일 단위 로지스틱 회귀 및 histogram gradient boosting 결합기를 calibration에
학습했다. 트리 모델은 calibration File EER을 크게 낮췄지만 validation에서
악화됐다. 서로 독립적인 음악 원본이 부족한 상태에서 점수 결합기를 학습하면
과적합된다고 판단해 채택하지 않았다.

### ArtifactNet 실험

ArtifactNet v9.4는 44.1kHz, 4초 구간에서 생성 코덱 잔차를 분석하는 17MB ONNX
모델이다. 최초 음악 의미쌍 100개에서는 결측 출력을 중립값 0.5로 처리한 뒤
Music EER `0.18`을 기록했다. competition_v2에서의 단독 Music EER은 다음과
같다.

| Detector | calibration | validation | holdout |
| --- | ---: | ---: | ---: |
| XLS-R music stem | 0.3521 | 0.4137 | 0.3684 |
| SONICS full audio | 0.4349 | 0.4310 | 0.4737 |
| ArtifactNet full audio | 0.3639 | 0.3190 | 0.3509 |

ArtifactNet은 calibration에서는 XLS-R보다 약간 나쁘지만 validation과 holdout에서
더 좋아 상보적인 일반화 신호로 판단했다.

### 채택 후보

```text
MUSIC_FAKE_PROB =
    0.70 × XLS-R(music stem)
  + 0.20 × SONICS(full audio)
  + 0.10 × ArtifactNet(full audio)

presence gate = 0.6
FILE_FAKE_PROB = max(active component fake scores)
```

| Split | 기존 ADS | 후보 ADS | 변화 |
| --- | ---: | ---: | ---: |
| calibration | 0.80162 | 0.81143 | +0.00981 |
| validation | 0.79886 | 0.81936 | +0.02050 |
| holdout | 0.79487 | 0.80951 | +0.01464 |
| 전체 | 0.80177 | 0.81053 | +0.00876 |

후보의 원본 그룹 단위 bootstrap ADS 95% 신뢰구간은 `[0.78183, 0.84262]`이다.
리더보드 ADS `0.69010`은 구간 밖이므로, 로컬 절대 점수를 리더보드 예상값으로
사용하지 않고 방법 간 상대 비교에만 사용한다.

## competition_v3 외부 재검증

새 시드로 MusicCaps/FakeMusicCaps 실제·가짜 의미쌍 200개를 구축했다. 이 중
기존 v1보다 독립적인 100쌍을 선택하고 음성 120개, 혼합 80개를 더해 원본
400개를 구성했다. 각 원본을 clean/MP3/telephone으로 변환해 정확히 1,200개로
맞췄다.

| 방법 | 전체 ADS | calibration | validation | holdout |
| --- | ---: | ---: | ---: | ---: |
| 이전 제출: XLS-R 75% + SONICS 25%, gate 0.3 | 0.71575 | 0.72526 | 0.66119 | 0.74291 |
| XLS-R 단독, gate 0.6 | 0.72375 | 0.73053 | 0.67930 | 0.73323 |
| XLS-R 70% + SONICS 20% + ArtifactNet 10%, gate 0.6 | 0.73048 | 0.74272 | 0.69131 | 0.75364 |

ArtifactNet 단독 Music EER은 전체 `0.3631`로 XLS-R `0.4119`, SONICS `0.4571`
보다 좋았다.

### 최종 maximin 선택

v2와 v3의 calibration/validation/holdout 여섯 개를 동일하게 보고, 가장 낮은
split ADS를 최대화했다. holdout 한 개의 최고점으로 고르지 않았다.

```text
MUSIC_FAKE_PROB = 0.50 × XLS-R + 0.50 × ArtifactNet
presence gate = 0.7
```

| v2 calibration | v2 validation | v2 holdout | v3 calibration | v3 validation | v3 holdout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8011 | 0.8602 | 0.8190 | 0.7659 | 0.7566 | 0.7530 |

여섯 split 평균 ADS는 `0.7927`, 최악 ADS는 `0.7530`이다. SONICS는 maximin
최적 가중치가 0이어서 최종 추론 경로에서 제거했다.
