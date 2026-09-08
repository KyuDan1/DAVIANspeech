# Encoder와 판별 단위 재검토: 공통 token readout 실험

2026-09-05. 목표는 실제 대회 1위 초과이며, 현재 확인된 최고점은
사용자가 전달한 v50+v57의 Total 0.7774307884 / ADS 0.7538888889다.
이 문서는 새 성능 달성 보고가 아니라, 좁은 residual 실험에서 벗어나기 위한
통제 비교의 구현·실행 기록이다. ZIP/API 제출은 하지 않는다.

## 무엇을 확인하려는가

이 대회는 단순 전화 분류나 음성 이진 탐지가 아니다. 음성/음악의 존재와
성분 진위를, 동시·순차·부분 생성 오디오에서 각각 판별한다. 전화 채널은
여러 변형 축 중 하나다. [공식 설명](https://dacon.io/competitions/official/236749/overview/description)

현재 최고 패키지는 HTDemucs stem XLS-R와 원본 EAT/SPEAR, WPT 등 여러
경로를 결합한다. 이미 temporal/component-query/adapter가 있으므로 그 이름을
다시 붙이는 것만으로 새 방법이라고 하지 않는다. 특히 기존 SPEAR probe,
EAT/SPEAR fusion은 서로 다른 window와 사전 압축된 통계를 사용했다.

이번 질문은 두 가지다.

1. 같은 파일·동일 구간·같은 판별 head에서 어떤 frozen encoder가 유리한가?
2. encoder를 고정한 채 token의 단순 평균과 task attention을 비교하면 차이가 있는가?

기존 결과를 다시 확인하니 `reports/eat_large_aasist_v56/seed00_strict/summary.json`의
실제 optimization train은 **2,908행**이다. 21,304행은 전체 입력 후보 수이며,
generator/source hypergraph의 최대 연결 성분이 96.65%여서 task-specific LOGO
구성이 크게 줄었다. 따라서 이 feasibility 결과로 "EAT-large를 충분히 학습했지만
실패했다"고 결론내릴 수 없다. 반면 EAT-base adapter v55는 17,760 train 사례의
학습 기록이 있다. 두 결과를 같은 규모/조건으로 묶지 않는다. 이번 18,738행 비교와
기존 v56의 차이는 head뿐 아니라 학습 규모·선택 조건에도 있으므로, v56 대비 이득을
하나의 구조 변경 덕분이라고 주장하지 않는다.

이후 결과에 따라 representation adaptation을 실험한다. frozen probe 실패를
해당 encoder 전체의 능력 한계라고 단정하지 않는다. 모든 encoder를 같은
학습률로 비교하는 것도 각 encoder의 최적 상한 비교는 아니다.

## 근거와 우선순위

- [XLS-R AntiDeepfake](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake)는
  speech forensic post-training 모델이다. 저자 카드의 In-the-Wild EER 1.23%와
  Deepfake-Eval-2024 27.76%의 차이는 speech-only도 미지 환경에서는 미해결임을 보여준다.
- [SPEAR v2](https://huggingface.co/marcoyang/spear-xlarge-speech-audio-v2)는 speech/general
  audio SSL과 token mixing을 사용한다. 혼합 표현의 타당한 후보지만, 공개 음향 이해
  성과를 딥페이크 탐지 성과로 읽으면 안 된다.
- [AT-ADD routed system](https://arxiv.org/abs/2608.00493)과
  [dual-domain fusion](https://arxiv.org/abs/2608.29021)은 EAT의 all-type 진위 판별
  가능성을 뒷받침한다. 해당 대회의 binary Macro-F1과 우리 성분별 EER는 다르다.
- [MixFake](https://arxiv.org/html/2605.23201v1)는 전경/배경의 진위를 교차하고
  신호 수준 priors로 SSL 표현을 적응시킨다. 이번 frozen readout은 논문 재현이 아니다.

## 사전 고정 비교

설정: `configs/common_encoder_probe.yaml`.

| 축 | 조건 |
|---|---|
| Encoder | EAT-large / SPEAR XLarge v2 / XLS-R-2B AntiDeepfake |
| 원본 입력 | 동일 mono 16 kHz 오디오, 분리 없음 |
| 구간 | 10.24초, 전체 파일을 덮는 동일 시작점; 짧은 파일은 zero padding |
| Layer | 상대 깊이 1/4, 1/2, 마지막의 3개 block |
| 특징 | 원래 차원의 token, EAT는 joint time-frequency patch 유지 |
| Head | LayerNorm → 96-D projection → task별 layer mixture → mean 또는 attention → mean/std → task별 linear |
| Output | File/Voice/Music fake, Voice/Music presence 5 logits |
| 파일 집계 | 각 window logit의 파일 내부 LME, T=2 |
| 학습 | 동일 strict train, 같은 weighted sampler seed, 6 epoch × 4,096 draws |
| 비교 | mean/attention head는 동일 초기 state에서 같은 encoder forward 결과를 공유 |
| 선택 | 0.5 pooled ADS + 0.5 domain-macro ADS, epoch 2/4/6 |

각 encoder의 native input width가 달라 input projection 파라미터 수는 다르다.
동일 encoder 내 두 pooling 조건은 파라미터 수와 state schema가 같다.
mean 조건에서는 query 파라미터가 비활성이므로 유효 자유도까지 같다고 주장하지 않는다.
Native frontend/normalization과 token 해상도도 다르며 이를 숨기지 않는다.

EAT 입력은 Kaldi 호환 fbank, XLS-R는 실제 오디오 부분의 layer normalization,
SPEAR는 native waveform frontend를 쓴다. 가짜 흔적을 삭제할 수 있는 별도
denoising/separation은 추가하지 않는다. EAT padding patch는 readout에서 제외되지만
encoder self-attention의 padding 문맥 효과까지 없다는 의미는 아니다.

## 정답과 데이터 보호

- strict train 18,738행 / development 4,577행 구성을 재사용한다. 출처 분리는 기존
  exclusion과 component/strict identity audit로 재검증한다.
- 추가로 train identity를 모든 protected role과 비교한다. protected manifest는
  identity 검사용으로만 읽고 audio/feature/prediction은 읽지 않는다.
- component가 없으면 해당 진위 loss는 비활성이다. NaN을 Real 정답으로 바꾸지 않는다.
- 파일 정답은 window를 LME 집계한 뒤 적용한다. Fake 파일의 모든 crop을 Fake로
  학습하지 않는다. 이 단계는 file-level weak supervision이며 dense localization 학습은 아니다.
- development는 source-disjoint 개발 검증이지 새 generator-heldout 증거가 아니다.
  이미 노출된 retired bank를 모델 선택에 재사용하지 않는다. Suno OOD/locked는 읽지 않는다.
- domain별 일부 성분 EER이 정의되지 않아도 pure speech/music domain을 통째로
  버리지 않는다. 각 task의 유효 domain EER를 평균한 뒤 ADS 가중치를 적용한다.
- epoch별 실제 train draw index를 저장하여 encoder 간 표본 일치를 검사할 수 있다.

## 검증과 실행 범위

`scripts/smoke_common_encoder_probe.py`는 authorized train의 8개 존재/진위 cell에서
각 1파일을 선택하고 전체 10개 window로 실제 가중치 forward/backward를 확인한다.
6 step loss는 실행 검사일 뿐 정확도/EER/일반화 증거가 아니다.

EAT-large 초기 smoke는 통과했다. SPEAR 초기 smoke는 짧은 파일의 native 출력
token 수가 batch별로 달라 concat에서 실패했다. 이는 encoder 성능 실패가 아니다.
smoke collector를 padding+valid mask 방식으로 수정했고, 기존 실패 디렉터리를 보존한다.
훈련 runner는 chunk별 head 결과만 연결하므로 원래부터 같은 token 수를 요구하지 않는다.

최종 smoke에서 세 encoder 모두 통과했고, 8파일/10window의 raw input SHA256이
정확히 일치했다. EAT-large/SPEAR/XLS-R의 token shape은 각각
`[10,3,512,1024]`, `[10,3,508,1280]`, `[10,3,511,1920]`이다.
공통 head 파라미터 수는 각각 101,908 / 126,996 / 189,716이다.
결과는 `reports/common_encoder_probe_smoke/{eat_large,spear_v2,xlsr}/report.json`.
초기 EAT runner는 concat collector 수정 전이지만 공통 encoder/head source SHA는 동일하다.

새 테스트 11개와 기존 data-guard/coverage/training 회귀 30개, 총 41개가 통과했다.
추가 preflight에서 train 18,738 / development 4,577행과 protected 44개 manifest
identity overlap 0을 확인했다. 이는 등록된 identity 검증이며 모든 미지 원천에 대한
완전한 지문 중복 탐지까지 수행했다는 의미는 아니다.

GPU 6에서 EAT-large 정식 mean/attention 비교를
`reports/common_encoder_probe/eat_large`로 시작했다. v64 학습과 평가가 모두 종료한
것을 PID와 결과로 확인한 뒤, 비어 있는 GPU 0에서 SPEAR 정식 비교도
`reports/common_encoder_probe/spear`로 시작했다. EAT-large는 실제 epoch 1의
600번째 gradient step까지 진행한 것을 확인했다. 이후 GPU 2의 183,359 MiB 중
약 1,324 MiB만 기존 작은 v67 작업이 사용 중임을 확인하고 XLS-R 정식 비교도
`reports/common_encoder_probe/xlsr`로 시작했다. 다른 사람의 GPU 작업은 변경하지
않았다. 공유 GPU의 학습 벽시계 시간을 L4 추론 성능 비교로 사용하지 않는다.
실행 검사 loss를 encoder 우열로 해석하지 않는다.

실행 예:

```bash
python -m pytest -q tests/test_common_encoder_probe.py tests/test_train_common_encoder_probe.py
CUDA_VISIBLE_DEVICES=6 python scripts/smoke_common_encoder_probe.py \
  --encoder eat_large --output reports/common_encoder_probe_smoke/eat_large_new
CUDA_VISIBLE_DEVICES=6 python scripts/train_common_encoder_probe.py \
  --encoder eat_large --output reports/common_encoder_probe/eat_large
```

실행 전 GPU를 재확인한다. `completed.json`이 없는 run은 완료로 보고하지 않는다.
모델 교체/제출 판단은 이 benchmark 하나로 하지 않는다. 최고 제출과의 정확한 비교,
새 생성기/출처 확인, L4 시간·메모리 및 패키지 검증은 이후 별도 필요하다.

## 다음 결정을 바꾸는 결과

- 같은 encoder에서 attention이 이기면 사전 요약 병목 가설이 지지된다.
- 두 pooling 모두 특정 encoder가 이기면 그 encoder의 forensic adaptation을 우선한다.
- Music-only는 좋고 혼합에서만 나쁘면 성분 교차 감독을 우선한다.
- 같은 출처 clean은 좋고 codec에서만 나쁘면 채널 대응 학습을 우선한다.
- 모든 후보가 특정 source/generator에서 실패하면 head를 계속 늘리기보다 데이터와
  representation adaptation을 재검토한다.

현재 54:05짜리 제출에 모든 후보를 추가하지 않는다. 대체 가능한 핵심 경로를 찾는
실험이며, 실제 1위 초과 여부는 미래 공식 채점으로만 확정한다.

## EAT-large 완료: 음악 보완 가능성, 전체 교체는 아님

EAT-large 6 epoch가 약 555.6초에 정상 완료됐다. 두 pooling 모두 epoch 6이
선택됐다. 아래 값은 **development 4,577행**의 결과이며 공식 점수가 아니다.

| Pooling | File EER | Voice EER | Music EER | ADS | 선택 점수 |
|---|---:|---:|---:|---:|---:|
| mean | 0.188764 | 0.211405 | 0.141751 | 0.820812 | 0.827862 |
| attention | 0.189444 | 0.202441 | 0.141281 | 0.822405 | 0.826896 |

attention의 pooled ADS는 약간 높지만 domain-macro를 포함한 사전 선택 점수는
mean이 높다. 사후에 pooled만 쓰도록 바꾸지 않는다. train 18,738은 후보 pool 수이고,
24,576회 sampling에서 실제 선택된 고유 행은 **10,165개**다. 모든 행을 한 번 이상
학습했다고 주장하지 않는다. codec 변형 행도 독립 원천 수와 같지 않다.

최고 제출과 직접 비교 가능한 codec development 600행:

| 모델 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| 실제 최고 v50+v57 pipeline | 0.233333 | 0.206667 | 0.253333 | 0.766000 |
| EAT-large mean, 단독 readout | 0.246667 | 0.270000 | 0.213333 | 0.758667 |

Music EER은 4%p 줄었지만 Voice/File이 나빠져 전체 ADS는 낮다. 따라서 이 모델로
최고 제출 전체를 교체하지 않는다. 음악 전문가 후보라는 해석은 가능하지만, 특정
가중치 결합이나 generator 일반화가 검증됐다는 뜻은 아니다.

EAT mean checkpoint를 독립적으로 다시 읽고 개발 80파일을 하나씩 추론했다.
저장된 batch 평가와 최대 확률 차이는 `2.98e-8`, 파일 순서를 뒤집은 독립 추론은
bit-exact였다. 이는 **80개 실행 동등성** 증거이고 전체 데이터/일반화 검증이 아니다.
`reports/common_encoder_probe/eat_large_independent_mean/report.json`에 기록했다.

최종 비교기 `scripts/compare_common_encoder_probe.py`는 세 run이 모두 완료되기
전에는 예측을 읽지 않는다. ordered train/dev ID, epoch별 실제 sampler draw,
source hash, checkpoint 선택과 재계산 점수를 검증한다. pure type별 EER와
동일 120개 base를 채널 전체와 함께 재표집한 paired bootstrap도 보고한다.
bootstrap은 개발 모델 선택 편향을 보정하지 않으며 실제 test 비율의 추정이 아니다.

이후 SPEAR/XLS-R/WavLM 학습을 완료했다. 배치 독립성 실패, 수정 SPEAR 재학습,
EAT 표현 학습 및 실제 파일 길이 감사 결과는
[후속 결과 문서](common-encoder-results-and-adaptation.md)를 우선 참고한다.
이 문서의 위쪽 실행 중 기록은 당시의 이력이다. 새 ZIP/API 제출과 locked 평가는 하지 않았다.
