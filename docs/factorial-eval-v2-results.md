# Factorial Eval v2 구축 및 1차 결과

## 결론

현재 병목은 음성 단독 탐지가 아니라 **음성과 음악이 완전히 동시에 재생될 때의
component fake ranking**이다. 특히 `가짜 음성 + 진짜 음악`에서 파일 판별이 가장
불안정하다. 음악 단독도 음성 단독보다 훨씬 어렵다.

다만 이 평가셋은 기존 모델이 학습한 Echoes 음악 생성기 계열을 포함한다. 역사적
anchor와 v10을 그대로 재실행했을 때 **실제 리더보드와 모델 순서가 반대로** 나왔다.
따라서 전체 점수뿐 아니라 아래 원자 대조 EER도 현재는 현상 분석용이며, 제출 모델
선택이나 ensemble weight 조정에는 사용하지 않는다.

## 평가셋

`data/eval/factorial_eval_1200_v2`는 dev/holdout/locked 각 400개, 총 1,200개다.

| 형태 | 전체 파일 수 |
| --- | ---: |
| 음성-only | 150 |
| 음악-only | 150 |
| 완전 동시 혼합 | 300 |
| 부분 중첩 | 300 |
| 순차(cascade) | 300 |

음성과 음악이 존재하는 1,050개에서 각 component는 real 525/fake 525다. 실제/가짜
음성과 음악의 네 조합은 혼합 방식별·split별로 각각 25개다. 64개의
`slices/*.csv`가 16개 원자 셀의 전체 및 split별 인덱스를 제공한다.

- 음성: 실제 96개와 Qwen3-TTS fine-tuned, Qwen3-TTS CustomVoice, F5-TTS,
  CosyVoice 생성 192개
- 음악: 실제 FMA 175개와 Echoes 생성 242개, 총 175개 원천 그룹
- 변형: clean FLAC, stereo WAV, MP3, OGG, 전화채널, 잡음/clipping
- 혼합: SNR -10/-5/0/5/10 dB, 부분 중첩 25/50/75%, 양쪽 순서
- 분리 모델은 데이터 생성 과정에 사용하지 않음

화자와 FMA 원곡 그룹의 split 교차는 모두 0건이다. 혼합 head 학습셋 세 곳과의
`VOICE_SOURCE_ID`, `MUSIC_SOURCE_ID` 정확 일치도 각각 0건이다. 모든 1,200개 파일은 16 kHz,
4~60초, mono/stereo 조건을 통과했다. 평가셋은 학습·calibration·ensemble weight
탐색에 넣지 않는다.

음악 쪽은 기존 Echoes 적응 head가 본 **생성기 계열**과 겹치지만, head 학습/과거
평가에 사용된 원곡 그룹과 파일은 제외했다. 따라서 source-disjoint 일반화는
측정하지만 완전한 unseen-generator 평가는 아니다. 새 음악 생성기 자료가 생기면
locked bank에 별도 generator-disjoint tier로 추가해야 한다.

## 리더보드 정렬 감사

동일한 1,200개 파일에 역사적 anchor와 제출된 v10 코드를 정확히 재실행했다.

| 모델 | 로컬 Score | 로컬 ADS | 실제 Score | 실제 ADS |
| --- | ---: | ---: | ---: | ---: |
| anchor 계열 | 0.6791 | 0.6441 | 0.7357 | 0.7075 |
| v10 fixed | 0.7295 | 0.7001 | 0.7076 | 0.6763 |

로컬에서는 v10이 anchor보다 Score `+0.0504`, ADS `+0.0560` 높지만 실제에서는
Score `-0.0281`, ADS `-0.0312` 낮다. 단순한 점수 오차가 아니라 **모델 선택 순서가
뒤집혔다**. Echoes 계열에 맞춘 EAT/XLS-R/SPEAR head가 같은 생성기 계열의 새 파일에
잘 작동한 것이 가장 유력하다. source ID disjoint는 generator disjoint를 보장하지
않는다.

또한 실험 과정에서 dev뿐 아니라 holdout/locked 결과까지 이미 여러 차례 확인했다.
그러므로 현재 `locked`도 더 이상 미관측 최종 검증셋이 아니라 audit split으로
취급한다. Echoes 밖의 생성기로 새 locked tier를 만들기 전에는 어떤 로컬 개선도
제출 개선 예상치로 변환하지 않는다.

## 현재 파이프라인 결과

현재 로컬 `src/pipeline.py`를 7 GPU shard로 실행했다. 전체 참고 결과는
File/Voice/Music EER `0.2959/0.2419/0.3410`, ADS `0.7014`, CPS `0.9941`,
Score `0.7307`이다. 위 순서 역전 감사 때문에 이 값은 실제 점수 추정치가 아니다.

| 형태 | File EER | Voice EER | Music EER | ADS |
| --- | ---: | ---: | ---: | ---: |
| 음성-only | 0.107 | 0.107 | - | - |
| 음악-only | 0.320 | - | 0.333 | - |
| 완전 동시 | 0.400 | 0.373 | 0.353 | 0.619 |
| 부분 중첩 | 0.293 | 0.220 | 0.340 | 0.707 |
| 순차 | 0.240 | 0.153 | 0.320 | 0.753 |

동일한 다른 component를 고정한 대조 결과는 다음과 같다.

| 대조 | EER |
| --- | ---: |
| 동시 RR 대 가짜 음성+진짜 음악 | 0.440 |
| 동시 RR 대 진짜 음성+가짜 음악 | 0.400 |
| 동시 RR 대 둘 다 가짜 | 0.347 |
| 부분 중첩 RR 대 가짜 음성+진짜 음악 | 0.267 |
| 부분 중첩 RR 대 진짜 음성+가짜 음악 | 0.387 |
| 순차 RR 대 가짜 음성+진짜 음악 | 0.187 |
| 순차 RR 대 진짜 음성+가짜 음악 | 0.360 |

완전 동시 조건에서는 fake voice가 real music에 가려지는 문제가 가장 크다. 반면
순차 조건의 Voice EER은 낮아, 파일 안에 두 성분이 있다는 사실 자체보다 같은
시간-주파수 영역에서의 마스킹이 핵심임을 보여준다. 음악 탐지는 순차에서도
0.32이므로 독립적인 두 번째 병목이다.

채널별 ADS는 clean `0.746`, noise `0.732`, MP3 `0.700`, stereo `0.698`, OGG
`0.677`, telephone `0.658`이었다. 전화채널과 저비트레이트 OGG도 우선 개선
대상이다.

## 생성기별 관찰

음성-only에서 현재 모델의 one-generator-vs-real EER은 CosyVoice `0.036`, F5
`0.068`, Qwen CustomVoice `0.106`, 화자별 fine-tuned Qwen `0.191`이었다.
Qwen fine-tuned가 가장 어렵지만, 전체적으로 speech-only는 충분히 강하다.

음악-only의 생성기별 샘플 수는 2~13개로 작아 순위를 확정할 수 없다. 다만 Suno
`0.533`, Brev `0.558`, ACE-Step `0.408`, DiffRhythm `0.400`이 어려웠고,
MusicGen `0.047`, SongGen `0.078`은 쉬웠다. 이는 생성기 지문에 대한 성능 편차가
매우 크다는 경고로 해석한다.

사용자가 준 보컬 포함 Suno 13곡에 대한 기존 v14 결과는 File/Music fake가 모두
13/13에서 0.5를 넘었다. 그러나 Voice presence는 0/13, Voice fake는 2/13만 0.5를
넘었다. 즉 파일 전체 fake 판정은 성공하지만, 보컬을 음성 component로 인식하는
대회 정의에는 맞지 않는다.

## 원본 XLS-R 대조 실험

분리 artifact 가설을 확인하기 위해 같은 XLS-R speech detector를 stem이 아닌 원본
혼합음에만 적용했다. 원본 XLS-R 단독은 전체 통제 Voice EER 평균/최악이
`0.267/0.467`로 현재 파이프라인의 `0.229/0.413`보다 나빴다.

dev에서 현재 score와 원본 XLS-R를 50:50으로 합치면 평균/최악이
`0.251/0.400 → 0.234/0.360`으로 좋아졌다. 그러나 holdout은 평균
`0.223 → 0.240`, locked는 `0.206 → 0.234`로 악화됐다. 따라서 이 결합은
채택하지 않는다. 이 결과는 source separation이 항상 안전하다는 뜻이 아니라,
단순히 원본 speech detector 하나로 바꾸는 것도 일반적인 해법이 아님을 뜻한다.

## 다음 우선순위

1. Echoes 밖의 YuE instrumental 8곡으로 첫 audit tier를 만들었다. 다음에는
   LeVo/HeartMuLa를 추가하고 한 family는 열어보지 않은 locked로 남긴다.
2. 새 tier를 만들기 전에는 기존 anchor를 제출 기준으로 유지하고, 현재 평가셋은
   RR/RF/FR/FF 오류 분석에만 사용한다.
3. 새 train 원천으로 완전 동시 혼합을 만들고 원본 오디오의 짧은 patch별 embedding에
   voice/music query attention을 학습한다. 분리 출력은 보조 expert로만 둔다.
4. dev에서만 weight/router를 선택하고 holdout을 한 번 확인한다. 새 locked는 최종
   후보 한 개에만 사용한다.
5. 보컬 포함 생성 음악의 voice presence를 별도 vocal detector 또는 시간 구간별
   PANNs 집계로 보완한다.

상세 결과는 `reports/factorial_v2_current/factorial_contrasts.csv`와
`factorial_cell_distributions.csv`에 저장했다.
