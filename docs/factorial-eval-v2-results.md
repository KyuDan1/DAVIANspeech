# Factorial Eval v2 구축 및 1차 결과

## 결론

현재 병목은 음성 단독 탐지가 아니라 **음성과 음악이 완전히 동시에 재생될 때의
component fake ranking**이다. 특히 `가짜 음성 + 진짜 음악`에서 파일 판별이 가장
불안정하다. 음악 단독도 음성 단독보다 훨씬 어렵다.

전체 균형 점수는 실제 비공개 데이터의 클래스 비율을 가정하지 않으므로 제출 점수
예측에 사용하지 않는다. 아래 원자 대조 EER과 split 간 재현성을 모델 선택 기준으로
사용한다.

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

화자와 FMA 원곡 그룹의 split 교차는 모두 0건이다. 모든 1,200개 파일은 16 kHz,
4~60초, mono/stereo 조건을 통과했다. 평가셋은 학습·calibration·ensemble weight
탐색에 넣지 않는다.

음악 쪽은 기존 Echoes 적응 head가 본 **생성기 계열**과 겹치지만, head 학습/과거
평가에 사용된 원곡 그룹과 파일은 제외했다. 따라서 source-disjoint 일반화는
측정하지만 완전한 unseen-generator 평가는 아니다. 새 음악 생성기 자료가 생기면
locked bank에 별도 generator-disjoint tier로 추가해야 한다.

## 현재 파이프라인 결과

현재 로컬 `src/pipeline.py`를 7 GPU shard로 실행했다. 전체 참고 결과는
File/Voice/Music EER `0.2959/0.2419/0.3410`, ADS `0.7014`, CPS `0.9941`,
Score `0.7307`이다. 실제 최고 제출의 `0.7365`와 수치가 가깝지만 한 번의 일치는
정렬의 증거가 아니며, 실제 점수 추정치로 해석하지 않는다.

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

1. 평가셋을 건드리지 않고 별도 train 원천으로 완전 동시 혼합을 만든다.
2. 원본 오디오의 짧은 시간 구간별 embedding을 유지한 채 voice/music별 attention
   pooling head를 학습한다. 분리 출력은 주 입력이 아니라 보조 expert로만 둔다.
3. dev에서만 weight/router를 선택하고 holdout 및 locked의 최악 대조 EER이 함께
   낮아질 때만 채택한다.
4. 음악은 Echoes 밖의 새 생성기 instrumental을 확보해 generator-disjoint tier를
   만든다.
5. 보컬 포함 생성 음악의 voice presence를 별도 vocal detector 또는 시간 구간별
   PANNs 집계로 보완한다.

상세 결과는 `reports/factorial_v2_current/factorial_contrasts.csv`와
`factorial_cell_distributions.csv`에 저장했다.
