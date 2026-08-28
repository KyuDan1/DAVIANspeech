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

## 실제 평가 0.7356958 이후: 분리 전 원본 병렬 전문가

실제 평가 결과는 Score `0.7356957778`, ADS `0.707531746`, CPS
`0.9891720635`였다. CPS를 그대로 유지하면서 Score `0.8`에 도달하려면 ADS
`0.778980882`가 필요하므로 ADS를 약 `0.07145` 높여야 한다.

분리기가 생성 artifact를 지우거나 자체 artifact를 만들 수 있다는 가설을 검증하기
위해, 동일한 XLS-R detector를 HTDemucs stem과 분리 전 원본에 각각 적용했다.
competition_v3에서 music stem XLS-R의 music-only EER은 `0.557`이었지만 원본
XLS-R은 `0.413`이었다. mixed의 Music EER도 `0.458` 대 `0.433`으로 원본이
더 안정적이었다.

두 개의 독립적인 1,200-file suite와 calibration/validation/holdout 여섯 split에서
maximin 탐색한 조합은 다음과 같다.

```text
VOICE_FAKE = 0.40 × XLS-R(voice stem) + 0.60 × XLS-R(original)
MUSIC_FAKE = 0.12 × XLS-R(music stem)
           + 0.48 × XLS-R(original)
           + 0.40 × ArtifactNet(original)
presence gate = 0.7
FILE_FAKE = max(active component fake scores)
```

| 방법 | v2 전체 ADS | v3 전체 ADS | 여섯 split 평균 | 여섯 split 최저 |
| --- | ---: | ---: | ---: | ---: |
| 기존 stem/ArtifactNet 50:50 | 0.8133 | 0.7625 | 0.7927 | 0.7153 |
| 원본 병렬 전문가 | 0.8262 | 0.7747 | 0.8065 | 0.7629 |

파일 점수에 원본 XLS-R을 한 번 더 직접 섞는 방식은 일부 split을 악화시켜
기각했다. 분리는 최종 판단의 유일한 입력이 아니라 보조 전문가로만 남겼다.

## EAT 일반 오디오 전문가와 AntiDeepfake 음악 head

`xls-r-2b-anti-deepfake`는 56,000시간의 실제 음성과 18,000시간의 가짜 음성으로
post-training된 speech 모델이다. 분리 music stem에 공개 speech head를 그대로
적용하는 것보다 원본에 적용하는 편이 나았지만 Music EER은 여전히 약 `0.41`이었다.
논문이 권장하는 task-specific adaptation을 시험하기 위해 encoder를 고정하고
mean-pooled 1,920차원 representation 위의 음악 head만 학습하는 경로를 추가했다.

동시에 AT-ADD Track 2 우승 방법을 따라 general-audio SSL인 EAT-base의 원본 6초
CLS representation에 L2-regularized linear music head를 학습했다. 미사용 split의
EAT Music EER은 v2 validation/holdout `0.086/0.070`, v3
validation/holdout `0.101/0.149`였다. fake generator 하나를 통째로 학습에서
제외한 검증은 generator별 `0.128~0.308`로 더 어려웠다. 이 차이를 고려해 EAT
단독 대신 다음의 보수적 결합을 채택했다.

```text
MUSIC_FAKE = 0.60 × previous raw/stem/ArtifactNet ensemble
           + 0.40 × EAT(original 6-second center crop)
```

| EAT weight | v2 validation | v2 holdout | v3 validation | v3 holdout | 최저 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.8550 | 0.8611 | 0.7678 | 0.7629 | 0.7629 |
| 0.40 | 0.9313 | 0.9345 | 0.8847 | 0.8703 | 0.8703 |

파일별 원본만 입력하므로 비공개 평가 파일 간 통계나 적응은 사용하지 않는다.

최종 MoE에서 music-stem XLS-R의 실효 가중치는 `1.2%`에 불과했다. 이 경로를
제거하고 legacy 10% 내부를 원본 XLS-R `85%`, ArtifactNet `15%`로 재탐색하자
미사용 네 split ADS는 `0.9493/0.9577/0.9345/0.9037`로 최저점이 오히려
`0.9001 → 0.9037` 개선됐다. 따라서 최종 제출은 music stem에 2B encoder를
실행하지 않아 XLS-R 추론량을 기존 세 pass에서 두 pass로 줄인다.

### 제출 패키지 전체 실행 검증

최종 `submit_moe_v3` 디렉터리의 FP16 XLS-R과 오프라인 모델 경로를 그대로 사용해
competition_v3 1,200개를 단일 B200에서 실행했다. 총 실행시간은 `796초`였고
누락·디코딩·VRAM 오류 없이 submission CSV 1,200행을 생성했다. 패키지 출력의
전체 성능은 ADS `0.94801`, CPS `0.97334`, Score `0.95055`였다. 이전 실제 제출은
더 무거운 파일당 3-pass XLS-R 구조로 평가 서버 시간 제한을 통과했으므로, 2-pass
XLS-R와 90M EAT를 사용하는 최종 구조는 기존 제출보다 실행시간 위험이 낮다.

AntiDeepfake 1,920차원 representation에 같은 방식의 음악 head를 학습하자 미사용
split Music EER은 `0.070~0.149`였다. 최종 세 전문가 maximin 결합은 다음과 같다.

```text
MUSIC_FAKE = 0.10 × legacy(raw/stem/ArtifactNet)
           + 0.40 × EAT music head
           + 0.50 × adapted AntiDeepfake XLS-R music head
```

이 결합의 미사용 split ADS는 v2 validation/holdout `0.9493/0.9630`, v3
validation/holdout `0.9381/0.9001`이었다. leave-one-generator-out Music EER은
`0.096~0.266`으로 legacy의 `0.257~0.326`보다 모든 다섯 generator에서 같거나
낮았다. adapted head는 기존 원본 XLS-R pass의 pooled embedding을 재사용하므로
2B encoder를 한 번 더 실행하지 않는다.

같은 representation에 voice 전용 linear head도 학습했다. 미사용 네 split의
Voice EER은 `0.000/0.0116/0.000/0.000`이었지만, 합성 평가셋과 비공개 평가셋의
차이를 고려하면 이 수치를 그대로 신뢰하기 어렵다. 따라서 공개 speech head를
대체하지 않고 다음처럼 10%만 섞었다.

```text
VOICE_FAKE = 0.90 × legacy(released head on raw + voice stem)
           + 0.10 × adapted AntiDeepfake XLS-R voice head
```

가중치 10%에서 미사용 split ADS는 `0.9493/0.9643/0.9345/0.9071`로, adapted
head를 쓰지 않은 경우의 `0.9493/0.9577/0.9345/0.9037`보다 최저 성능이
개선됐다. 이 head 역시 원본 XLS-R embedding을 재사용해 추론시간을 늘리지 않는다.

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

## SPEAR XLarge v2 원본 혼합음 전문가

`marcoyang/spear-xlarge-speech-audio-v2`(Apache-2.0)는 음성 184k시간과 일반
오디오 13k시간으로 학습된 600M Zipformer다. 분리 과정에서 생성 흔적이
손실될 가능성을 피하기 위해 stem이 아니라 16kHz 원본 오디오만 입력하고,
13개 계층의 평균 pooling 표현에 선형 music-fake head를 학습했다.

12개 음악 생성기를 하나씩 통째로 제외하는 Echoes leave-one-generator-out
평가에서 최적 중간 계층의 평균 Music EER은 `0.069`, 최악 생성기는 `0.178`로
나왔다. competition_v3로 학습한 head는 독립 competition_v2에서 Music EER
`0.007`, Echoes 전체에서 `0.130`이었다. 반면 EchoFake 미지 음성 생성기 및
재녹음 평가에서는 Voice EER `0.100`으로 기존 XLS-R MoE `0.070`보다 나빠
음성 전문가로는 채택하지 않았다.

SPEAR 단독은 외부 동시/순차 혼합셋에서 Music EER `0.355`로 약했지만 기존
MoE와 상관이 낮아 10% 결합 시 `0.220 → 0.210`으로 개선됐다. 1,200개 진단셋
두 곳에서도 ADS가 각각 `0.93356 → 0.94470`, `0.92565 → 0.94184`로 상승했다.
따라서 v6은 기존 음악 전문가 합을 90%로 축소하고 SPEAR 원본음 전문가에
10%를 배정한다. 음성 경로와 HTDemucs 보조 stem 경로는 변경하지 않는다.

평가 서버의 `transformers==4.57.6`에서 Hugging Face custom-code 캐시가 상대
import 파일을 누락하는 문제도 재현했다. EAT와 SPEAR 모두 번들 디렉터리를
직접 Python package로 로드하도록 바꿔 오프라인 캐시에 의존하지 않게 했다.

## v7 혼합 전용 MoE

v6의 가장 약한 외부 혼합셋은 File/Voice/Music EER이 각각
`0.280/0.300/0.220`, ADS가 `0.734`였다. 기존 검증 혼합에 쓰인 음성 및 음악
원본 ID를 모두 제외하고 별도의 800개 학습 혼합을 만들었다. real/fake 성분
네 조합, simultaneous/sequential, SNR -6/0/+6을 균형화했다.

SPEAR 중간 표현에 학습한 혼합 전용 voice/music head를 기존 MoE와 결합했다.
또한 이 head를 단일 성분에 잘못 적용하지 않도록, 같은 SPEAR embedding에
mixture-presence head를 학습했다. 학습·검증의 성분 원본을 완전히 분리했을 때
mixture 분류 EER은 `0.0050`, ROC-AUC는 `0.99975`였다. competition_v2/v3의
별도 도메인에서도 EER `0.0082/0.0042`를 기록했다.

```text
if SPEAR_MIXTURE_PRESENT_PROB >= 0.8:
    VOICE_FAKE = 0.40 × existing + 0.60 × SPEAR_mixed_voice
    MUSIC_FAKE = 0.80 × existing + 0.20 × SPEAR_mixed_music
```

세 진단 도메인에 대한 maximin ADS는 `0.927/0.923/0.8175`로, 이전 최악값
`0.734`보다 `0.0835` 상승했다. 외부 혼합셋의 v7 File/Voice/Music EER은
각각 `0.210/0.170/0.145`이며, ADS는 `0.8175`다. 직접 학습한 file head는
다른 도메인 과적합이 커서 사용하지 않고 component score의 논리 OR를 유지한다.

## v8 생성기 제외 감사와 보수적 혼합 결합

v7의 높은 원본-ID holdout 성능을 다시 검증하기 위해 fake 생성기를 하나씩
통째로 학습에서 제외했다. 이 엄격한 조건에서 SPEAR mixed voice/music head는
각각 평균 EER `0.457/0.498`로 거의 chance였다. 생성기 지문 과적합으로
판단해 v7의 60% voice 가중치를 그대로 제출하지 않았다. dev→open 음성 생성기,
FakeMusicCaps→Echoes 음악 생성기 교차 학습도 SPEAR `0.460/0.475`, XLS-R
`0.475/0.465`로 실패했다.

HTDemucs의 music stem도 원본 MoE Music EER `0.21`보다 나쁜 XLS-R `0.325`,
EAT 최적 결합 `0.25`를 기록했다. 이는 분리 과정이 music fake 흔적을 보존하지
못한다는 가설을 지지하므로 music stem detector는 추가하지 않았다.

최종 v8은 생성기 제외 평가에서 순위를 거의 바꾸지 않고 voice 최악 EER도
소폭 낮춘 보수적 20% 가중치만 사용한다. mixture router가 양성이면 두 성분이
존재한다는 강한 증거이므로, PANNs가 한 성분을 놓쳐도 File score는 두 component
score의 `max`로 계산한다.

```text
if SPEAR_MIXTURE_PRESENT_PROB >= 0.8:
    VOICE_FAKE = 0.80 × existing + 0.20 × SPEAR_mixed_voice
    MUSIC_FAKE = 0.80 × existing + 0.20 × SPEAR_mixed_music
    FILE_FAKE = max(VOICE_FAKE, MUSIC_FAKE)
```

재구성 검증의 ADS는 competition_v2 `0.9498`, competition_v3 `0.9503`, 외부
혼합 `0.8095`다. CPS `0.98917`을 그대로 대입한 최악 조건 예상 total은 약
`0.827`이지만, 목표 달성 여부는 실제 비공개 제출 점수로만 판정한다.

## v9 voice-stem adapted head 공유

v8의 병목은 외부 혼합 Voice EER `0.260`이었다. HTDemucs voice stem은 이미
released XLS-R head에 사용하지만, EchoFake에 적응한 head는 원본 embedding에만
적용하고 있었다. 동일한 stem XLS-R pass에서 mean embedding을 함께 회수해
적응 head를 적용하면 encoder 추론 횟수는 늘지 않는다.

순수 EchoFake open-set에서 현재 voice 결합 EER `0.045`가 stem-adapted 30%
추가 시 `0.025`로 개선됐다. 외부 혼합에서는 Voice EER `0.260 → 0.185`,
File EER `0.190 → 0.1483`, ADS `0.8095 → 0.8453`이었다. 같은 30%에서
competition_v2/v3 ADS도 `0.9423/0.9406`을 유지했다. 40% 이상은 두 로컬
도메인을 더 크게 훼손해 30%를 maximin 선택으로 사용한다.

```text
VOICE_FAKE = 0.70 × existing_voice + 0.30 × XLSR_EchoFake(voice_stem)
```

음악 stem은 앞선 실험대로 사용하지 않는다. v9 최악 진단 ADS `0.8453`에 현재
CPS를 적용한 예상 total은 약 `0.859`다.

## v10 Fourier music fakeprint 전문가

SPEAR/XLS-R/EAT가 모두 신경망 표현을 사용하므로 서로 다른 종류의 증거를
추가하기 위해 Deezer의 ISMIR 2025 논문 *A Fourier Explanation of AI-music
Artifacts*와 공식 CC BY-NC 4.0 구현을 적용했다. 대회 입력은 16kHz이므로 원
구현의 5--16kHz 분석 대역을 5--7.99kHz로 제한했다. 원본 혼합음의 평균 Fourier
스펙트럼에서 local lower hull을 제거한 고주파 잔차를 fakeprint로 사용한다.
분리 stem은 사용하지 않으므로 HTDemucs가 생성 흔적을 지울 위험도 없다.

Echoes 음악 생성기 12종을 하나씩 통째로 제외한 leave-one-generator-out
검증에서 평균 Music EER은 `0.0356`, 최악 Music EER은 `0.1188`이었다. 독립
외부 혼합셋에서 Fourier 단독 Music EER은 `0.105`였다. 여러 도메인에 대한
maximin 탐색 결과 기존 music MoE 90%와 Fourier 10%를 결합했다. 더 큰 비중은
FakeMusicCaps 계열의 성능을 훼손해 사용하지 않았다.

| 평가셋 | v9 ADS | v10 ADS |
| --- | ---: | ---: |
| competition_v2 재구성 | 0.94227 | 0.93953 |
| competition_v3 재구성 | 0.94060 | 0.93976 |
| 외부 혼합 400개 | 0.84533 | 0.86200 |

외부 혼합에서 File/Voice/Music EER은 각각 `0.130/0.185/0.120`이다. v9보다
File EER은 `0.1483 → 0.130`, Music EER은 `0.145 → 0.120`으로 개선됐고
Voice EER은 유지됐다. CPS `0.98917`을 가정한 최악 진단셋 예상 total은
`0.8747`이다. 이는 로컬 추정치이며 목표 달성은 실제 비공개 평가로 확인한다.

출처:

- 논문: <https://arxiv.org/abs/2506.19108>
- 공식 구현: <https://github.com/deezer/ismir25-ai-music-detector>

### v10 실제 비공개 평가 결과

설치 충돌을 제거한 v10 fixed의 실제 결과는 Score `0.7075886349`, ADS
`0.6763015873`, CPS `0.9891720635`였다. 이전 실제 제출의 ADS `0.707531746`보다
`0.031230159` 하락했다. CPS가 소수점 이하까지 동일하므로 presence 경로가 아니라
fake ranking MoE의 비공개 도메인 역일반화가 원인이다.

로컬 외부 혼합 ADS `0.862`와 실제 ADS의 방향이 반대였으므로 기존 합성 혼합셋을
더 이상 제출 선택의 단독 근거로 사용하지 않는다. 다음 제출은 v10에서 Fourier
10%만 제거한 v9 fixed로 정해 Fourier의 실제 기여를 한 번에 분리한다. v9도
회복하지 못하면 Echoes/SPEAR/EchoFake 적응 head를 제거한 실제 검증 제출로
거슬러 올라간다.
