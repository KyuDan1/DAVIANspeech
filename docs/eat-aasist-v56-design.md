# v56 설계안: EAT-large–AASIST 음악/혼합 오디오 전문가

작성일: 2026-09-04

상태: **구현 전 설계 동결 후보**

범위: 기존 v50 계열 앵커를 보존하고, 원본 혼합 오디오에서 동작하는 EAT-large–AASIST 전문가 하나를 추가한다.

## 1. 결론

v56에서 가장 기대값이 높은 한 가지 모델은 다음과 같다.

> **분리하지 않은 원본 오디오 → EAT-large의 24개 층을 학습 가중합 → AASIST 그래프 백엔드 → File/Voice/Music 세 개의 logit**을 만들고, 이 중 **File과 Music만 기존 v50 앵커에 작은 soft residual로 결합**한다. Voice와 두 Presence 출력은 기존 앵커와 byte-identical하게 유지한다.

EAT+XLS-R token fusion도 강한 후보지만 이번 v56의 1순위로 선택하지 않는다. AT-ADD 2위 논문의 공개 ablation에서 EAT+XLS-R의 가장 뚜렷한 이득은 Speech/Singing 쪽이고, Music은 EAT-only와 동등하거나 소폭 낮았다. 반면 1위 시스템의 SoundMusic-EAT는 sound/music 도메인 전용으로 학습되어 해당 두 도메인에서 각각 97.70%/98.99%의 높은 progress accuracy를 보였다. 현재 우리 앵커가 이미 XLS-R 2B와 SPEAR를 포함하고 있고, 실전 병목도 File/Music EER이므로, 제한된 용량과 시간은 EAT-large–AASIST 음악/혼합 전문가에 쓰는 편이 합리적이다.

이 설계의 핵심은 세 가지다.

1. Demucs, SAM-Audio 같은 생성형 분리기를 사용하지 않는다. 탐지 대상 artifact를 변형하거나 새로운 artifact를 넣을 가능성을 없애고, 순차·부분 중첩·희소 성분을 시간 crop과 attention으로 처리한다.
2. hard router나 presence gate를 사용하지 않는다. 잘못된 routing이 File/Music 점수까지 망가뜨리는 error propagation을 피하고, logit 공간의 작은 soft residual만 사용한다.
3. 임의 분할이 아니라 **generator/source parent-disjoint LOGO**로 선택한다. 같은 원본의 codec·전화·layout 파생본은 반드시 같은 fold에 둔다.

## 2. 공식 연구 근거와 공개 자산 검증

### 2.1 AT-ADD 1위: Hidden-Domain Routing

[Hidden-Domain Routing for Universal AI-Generated Audio Detection](https://arxiv.org/abs/2608.00493)은 AudioType-BEATs-6s router와 도메인별 전문가를 사용한다. Speech에는 XLS-R+AASIST, Sound/Music 및 Singing에는 EAT-large+AASIST를 사용한다. SoundMusic-EAT는 전체 오디오 유형을 사용하되 music sampling을 늘리고, codec augmentation과 binary objective를 적용하며, 학습 시 10초 random crop을 사용한다. 논문의 progress set에서 SoundMusic expert는 Music 98.99%, Sound 97.70%를 기록했다.

다만 해당 논문에서 학습된 **SoundMusic-EAT/AASIST checkpoint나 실행 코드의 저자 공식 공개 링크는 확인되지 않았다.** 따라서 이름만 같은 비공식 구현의 가중치를 가져오는 대신, 아래의 공식 EAT와 AASIST 구성요소를 이용해 우리 데이터로 재현해야 한다.

### 2.2 AT-ADD 2위: EAT+XLS-R token fusion

[Beyond Speech: A Unified Multimodal Framework for AI-Generated Audio Detection](https://arxiv.org/abs/2608.29021)은 EAT-large와 XLS-R 300M의 각 24개 층을 learnable scalar로 섞은 뒤, 서로 frame-align하지 않고 token/time 축으로 concatenate한다. 이후 shared SwiGLU, multi-head attentive mean/std pooling, MLP를 사용한다. 학습은 4초 random crop, 추론은 five-crop이며 RawBoost Algorithm 5를 사용한다.

공개 표에서 EAT-only+RawBoost는 Speech/Sound/Singing/Music 79.47/98.93/92.40/98.45, EAT+XLS-R concatenation은 84.38/98.18/97.39/98.25를 기록했다. 서로 다른 system row의 차이에 다른 학습 조건이 섞였을 가능성은 있지만, 적어도 **Music 개선을 직접 입증하는 결과는 아니다.** 논문의 최종 핵심도 단순 hard routing이 아니라 unified detector와 보수적인 speech refinement였다. 이 때문에 token fusion은 v57 후보로 남기고, v56에는 넣지 않는다.

### 2.3 EAT와 AASIST의 실제 공개 여부

- [EAT 공식 코드](https://github.com/cwx-worst-one/EAT)는 MIT 라이선스이며, 논문 저자의 EAT-large checkpoint 링크를 제공한다.
- [공식 EAT-large Hugging Face 모델](https://huggingface.co/worstchan/EAT-large_epoch20_finetune_AS2M)은 공개·비-gated 모델이다. 고정할 revision은 `53b1251196e3dabb991dfa7629fa89394c7f8335`이며 `model.safetensors` 크기는 1,237,667,364 bytes이다. 구조는 24 layers, hidden size 1024, 16 heads, 약 309M parameters이다.
- [AASIST 공식 코드](https://github.com/clovaai/aasist)는 MIT 라이선스이며 ASVspoof 2019 speech용 pretrained AASIST weights를 제공한다.
- 그러나 AASIST 공식 pretrained weights는 raw-wave speech AASIST용이다. **EAT-large token을 입력으로 받는 AASIST backend의 저자 공식 pretrained weights는 확인되지 않았다.** EAT–AASIST 결합부와 backend는 우리 데이터로 학습한다.
- [RawBoost 공식 코드](https://github.com/TakHemlata/RawBoost-antispoofing)는 MIT 라이선스이며, Algorithm 5는 linear/non-linear convolutive noise와 impulsive signal-dependent noise의 직렬 구성이다.

AT-ADD 공식 데이터는 접근 동의가 필요한 gated 자산이다. 이 설계에서는 다운로드하거나 학습에 포함하지 않는다. 대회 blind v8과 v5–v7 truth/prediction도 모델 선택·학습·분석에 사용하지 않는다.

## 3. 현재 레포 자산 감사

### 3.1 모델과 제출 용량

| 자산 | 로컬 크기 | 용도/판단 |
|---|---:|---|
| `models/eat-base-as2m/model.safetensors` | 361,533,396 B | 12-layer, 768-dim EAT-base. v55에서 사용했으나 v56 전문가는 large로 교체한다. |
| `models/spear-xlarge-speech-audio-v2/model.safetensors` | 2,387,980,808 B | 기존 앵커에 이미 포함. v56에서 별도 복제하지 않는다. |
| v50 제출 내 XLS-R shards | 4,325,972,284 B | 기존 voice detector. 그대로 보존한다. |
| v50 제출 내 Spectra-AASIST | 1,267,634,096 B | 기존 앵커. 그대로 보존한다. |
| v50 제출 내 EAT-base | 361,549,259 B | 기존 presence/feature 경로. 그대로 보존한다. |
| `sparse_voice_v49_base.zip` | 8,276,241,197 B | 현실적인 제출 앵커 크기 기준이다. |
| EAT-large 공식 weight | 1,237,667,364 B | v56 신규 핵심 weight. |

v49/v50 계열 앵커 ZIP에 EAT-large를 더한 단순 상한은 약 **9,513,908,561 B (9.514 GB)**이다. AASIST backend와 코드가 10 MB 안쪽이면 10 GB 제한 아래이지만 여유가 약 0.48 GB뿐이다. 따라서 v54/v55의 추가 대형 모델이나 사용하지 않는 checkpoint는 함께 넣지 않는다. 최종 빌드는 ZIP64로 만들고 CRC, 압축/해제 크기, 세 루트 항목(`model/`, `script.py`, `requirements.txt`)을 검증한다.

### 3.2 학습 데이터 generator와 구조

v56에 사용할 canonical train bank는 아래 7개이며 총 18,360개다. Voice/Music component 정답 필드를 함께 제공해 multitask 학습에 바로 쓸 수 있는 예시는 17,760개다.

| Bank | 개수 | 주요 역할 |
|---|---:|---|
| `external_mixed_train_v1` | 800 | 외부 혼합 데이터 |
| `mixed_devvoice_train_v1` | 800 | 다양한 voice 혼합 |
| `mixed_fmc_music_train_v1` | 800 | music source 확대 |
| `mixfake_music_train_v1` | 8,000 | FMA real music과 Udio, MusicLDM, Suno, Mustango, MusicGen-medium, AudioLDM2 fake music |
| `telephone_mixed_train_v1` | 2,400 | 전화/codec 변형 혼합 |
| `temporal_mixed_train_v2` | 2,560 | concurrent, partial-overlap, sequential, sparse-voice, sparse-music 각 512 |
| `channel_invariant_factorial_train_v1` | 3,000 | clean 1,000 + channel variants 2,000; voice-only 600, music-only 600, mixed 1,800 |

`temporal_mixed_train_v2`의 TTS에는 OpenAudio-S1-mini, SpeechT5, StyleTTS2, IndexTTS, LLaSA, MaskGCT, CosyVoice2, OpenVoice-V2, XTTS-v2, FireRedTTS, F5-TTS가 있다. Music generator 이름은 Suno, Udio, ACE-Step, ElevenLabs, DiffRhythm, MusicGen, AudioLDM/2, Stable Audio, MusicLDM, Mustango, MuBERT, SongGen 등으로 정규화한다. 예를 들어 `MusicGen_medium`/`musicgen`, `audioldm`/`audioldm2`, `stableaudio`/`stable_audio_open`은 각각 하나의 generator family로 묶는다.

주의할 shortcut은 명확하다. Real music source가 FMA/Echoes 등에 몰리고 fake가 generator별 source에 몰려 있으면 모델이 생성 artifact 대신 corpus를 외울 수 있다. 전화·codec·layout 파생본도 독립 샘플처럼 random split하면 누수가 생긴다. 따라서 모든 파생본을 원본 parent와 같은 fold에 유지해야 한다.

### 3.3 v55에서 가져갈 교훈

v55의 EAT-base shallow pooling은 작은 selection dev에서는 개선됐지만, 채널별 score offset과 generator overlap, clean/codec pair의 crop 불일치 때문에 일반화 근거가 약했다. v56은 다음을 바꾼다.

- EAT-base shallow pooling 대신 EAT-large의 전 층 가중합과 AASIST 시간-주파수 그래프를 사용한다.
- consistency pair에는 **동일한 시작점과 길이의 crop**을 사용한다.
- fold 평균만 보지 않고, 전체 channel을 합친 global EER와 class-conditional logit offset까지 본다.
- 학습 loss 최소점이 아니라 generator-family worst-fold 성능으로 checkpoint를 선택한다.

## 4. v56 architecture

### 4.1 입력과 EAT frontend

1. MP3/WAV/FLAC를 16 kHz float waveform으로 decode한다.
2. Stereo는 채널 평균 mono로 만들되 clipping을 피한다. 신경망 기반 source separation은 하지 않는다.
3. 공식 EAT preprocessing을 따라 128-bin log-Mel filterbank를 만들고 10.24초, 1,024 frames로 맞춘다.
4. 16×16 patch embedding으로 약 64×8=512 patch tokens를 만든다.
5. 초기 구현은 전체 24개 block을 통과하되, 계산·메모리 대비 정보 중복을 줄이기 위해 `[3, 7, 11, 15, 19, 20, 21, 22, 23]`의 9개 hidden state를 보존한다. 학습 가능한 scalar 9개에 softmax를 적용해 token별 weighted sum을 만든다. 마지막 네 block에는 zero-initialized bottleneck adapter를 둔다.
6. 1,024-d token을 128-d로 projection하고 LayerNorm을 적용한다.

### 4.2 AASIST backend

공식 AASIST의 설계 원리를 EAT token grid에 맞춰 재구현한다.

- projection token을 시간×주파수 grid로 복원한다.
- residual 2-D convolution blocks로 local artifact를 보강한다.
- spectral graph attention과 temporal graph attention을 별도로 적용한다.
- 두 graph를 heterogeneous graph attention과 master node로 결합한다.
- attention pooling으로 고정 길이 global embedding을 만든다.
- File, Voice, Music의 독립 binary logit head 세 개를 둔다.

Voice head는 representation regularization과 File consistency를 위한 보조 head다. 제출 시 `VOICE_FAKE_PROB`에는 결합하지 않는다. 직접 File logit과 component noisy-OR는 학습 시 일치시키고, 전문가 내부 File 확률은 두 값의 동일 가중 평균으로 고정한다.

### 4.3 길이별 추론과 기존 앵커 결합

최대 60분 제한을 위해 모든 파일에 five-crop을 쓰지 않는다.

- 길이 ≤12초: center 1 crop
- 12초 < 길이 ≤30초: start/end 2 crops
- 길이 >30초: start/middle/end 3 crops

crop logit은 max가 아니라 temperature 5의 log-mean-exp로 결합한다. 그 뒤 기존 앵커와 logit 공간에서 convex logit interpolation으로 결합한다.

```text
music_logit_v56 = (1 - w_music) * music_logit_anchor
                  + w_music * music_logit_expert
file_logit_v56  = (1 - w_file) * file_logit_anchor
                  + w_file * file_logit_expert
```

진단 구현의 후보 weight grid는 `w_music, w_file ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30}`이다. 이 sweep의 최고점을 곧바로 배포 weight로 사용하지 않고, component와 domain별 방향이 일치할 때만 동결한다.

다음 두 출력은 앵커와 정확히 동일하게 둔다.

```text
VOICE_FAKE_PROB    = anchor 그대로
VOICE_PRESENT_PROB = anchor 그대로
MUSIC_PRESENT_PROB = anchor 그대로
```

hard component router, phone router, presence threshold는 새로 넣지 않는다. 원본 혼합물 하나에서 모든 성분을 보고 residual만 더하므로 순차 혼합과 overlap 모두 동일한 경로를 탄다.

## 5. Train/validation split

### 5.1 누수 없는 3-fold generator LOGO

각 row에 아래 group key를 만든다.

```text
voice_generator_family
music_generator_family
real_source_family / speaker_id / artist_id
parent_audio_id
layout_recipe_id
channel_derivative_parent_id
```

split 코드는 whole generator family를 통째로 hold-out하는 세 outer fold를 지원한다. 한 원본에서 만든 clean, telephone, codec, packet-loss, sequential, overlap 파생본은 모두 같은 fold다. Real은 speaker/artist/source 단위로 분리한다. 같은 generator의 표기 alias도 같은 family로 정규화한다.

실제 21,304개 train row에서 모든 Voice/Music generator·identity·parent token을 연결한 hypergraph를 감사한 결과, 491개 connected component 중 최대 component가 20,590개(96.65%)였다. 이는 `unknown` 같은 공통 placeholder 때문이 아니다. 해당 값은 token 생성 단계에서 제거했다. 원인별 최대 component는 generator token만 사용할 때 10,160개(47.69%), identity token만 사용할 때 13,151개(61.73%), parent token만 사용할 때 4개(0.02%)였으며, parent를 완전히 빼도 전체 최대 component는 20,590개로 동일했다. 즉 여러 음성·음악 source와 generator를 교차 조합한 mixed/FF 데이터가 실제 bipartite 연결을 만든 것이 원인이다.

따라서 전 token atomic 3-fold는 유효한 train 크기를 만들 수 없어 자동으로 task-specific LOGO로 전환한다. 이 fallback도 대표 group 하나만 보지 않고, 각 task의 모든 generator/identity/parent token으로 별도 hypergraph를 만든다. Music task는 20,576개 present row에서 4,248개 component이며 최대 component가 12,748개(61.96%)다. Voice task는 20,704개 present row에서 2,225개 component이며 최대 component가 11,550개(55.79%), 두 번째가 4,000개다. 이는 task 안에서도 같은 generator와 source를 반복 교차 조합했기 때문이다.

엄격한 실제 fold 0은 optimization 2,908개, Music holdout 15,431개/1,423 connected components, Voice holdout 13,271개/722 connected components이며, 각 task의 **모든 token view** train–holdout overlap은 0이다. File checkpoint는 Music-LOGO와 Voice-LOGO 양쪽의 File EER를 모두 포함한 mean+worst criterion으로 선택한다. 이 분할은 보수적이고 train이 작다는 한계가 있지만, 한 task의 source alias가 양쪽으로 새는 것보다 안전하다. 한 task의 holdout에서 다른 task identity가 보일 수 있다는 제한은 지표에 명시하며, 이를 source-disjoint all-task 증거로 과장하지 않는다.

설계상 세 fold를 모두 통과해야 최종 후보로 승격한다. 이번 feasibility run은 fold 0 하나에서만 학습했으므로 최종 재학습 또는 제출 checkpoint가 아니다. blind v8과 v5–v7 truth/prediction은 어떤 단계에도 사용하지 않았다.

### 5.2 batch sampling

corpus 크기 대신 다음 순서로 균형 sampling한다.

```text
corpus → layout → presence pattern → RR/RF/FR/FF → generator/source family
```

batch의 목표 질량은 music-only 30%, mixed 50%, voice-only 20%다. Mixed의 RR, RF, FR, FF는 각각 전체 batch의 12.5%를 목표로 한다. 부족한 cell은 반복 sampling하되 한 parent가 한 epoch를 지배하지 않도록 parent cap을 둔다.

순차·희소 데이터는 파일 label을 그대로 crop에 붙이지 않는다. metadata의 voice/music 구간과 crop의 교집합으로 crop-level presence와 fake label을 다시 계산한다. crop 안에 없는 component의 BCE는 mask하고, crop File label도 **그 crop에 실제로 들어온 fake component가 있는지**로 재계산한다.

## 6. Loss와 학습법

최종 목표 loss는 다음과 같다.

```text
L = 0.50 L_file
  + 0.20 L_voice(masked)
  + 0.30 L_music(masked)
  + 0.10 L_rank
  + 0.05 L_or
  + 0.02 L_pair
  + 0.01 L_channel
```

- `L_file`, `L_voice`, `L_music`: generator-balanced binary cross entropy. Component loss는 해당 component가 crop에 존재할 때만 계산한다.
- `L_rank`: 같은 task의 positive/negative logit에 대한 pairwise logistic ranking loss. 대회의 EER 목표와 방향을 맞춘다.
- `L_or`: direct File probability와 `1-(1-p_voice)(1-p_music)`의 consistency loss다.
- `L_pair`: 동일 crop의 clean/codec pair logit에 대한 Smooth-L1이다.
- `L_channel`: class-conditional clean/channel logit 평균과 분산을 맞추는 약한 regularizer다. fake/real 자체의 차이를 지우지 않도록 weight를 0.01로 제한한다.

다만 첫 feasibility run은 학습 안정성과 시간 예산을 우선 확인하기 위해 EAT-large 309,409,295개 parameter를 모두 freeze하고, 마지막 네 block의 bottleneck adapter와 AASIST backend 443,736개만 학습했다. 실제 실행 loss는 masked Voice/Music/File BCE `0.20/0.30/0.50`, pairwise rank `0.10`, component-OR consistency `0.05`이다. `L_pair`와 `L_channel`은 아직 넣지 않았다.

최종 학습 후보 단계는 다음과 같다.

1. EAT를 freeze하고 adapter+AASIST/head를 학습한다.
2. feasibility gate를 통과한 경우에만 EAT top blocks의 제한적 unfreeze를 비교한다.
3. AASIST/head LR은 1e-4, layer-mixture scalar는 1e-4로 둔다.
4. AdamW, weight decay 1e-2, OneCycle schedule과 BF16을 사용한다.
5. 메모리가 허용하는 가장 큰 physical batch를 사용한다. feasibility run은 B200에서 batch 32였다.
6. generator worst-fold criterion의 early stopping으로 종료한다.

최종 제출은 ensemble이 아니라 단일 checkpoint다. 필요하면 동일 run 내부의 EMA를 하나의 state로 저장할 수 있지만, 여러 seed/checkpoint를 동시에 넣지는 않는다.

## 7. Augmentation

목표 augmentation은 label과 독립적으로 대칭 적용한다.

- 40%: clean
- 25%: 공식 RawBoost Algorithm 5(LnL + ISD)
- 35%: 아래 channel/codec 중 하나 또는 합리적인 짧은 stack
  - G.711 μ-law/A-law
  - G.726 24 kbps
  - Opus narrowband 8/12 kbps
  - 8 kHz resampling + PSTN bandpass
  - MP3/Ogg/AAC
  - G.711/Opus packet loss
  - random bandpass
  - light RIR, background noise, clipping/level change

첫 feasibility run은 확률 0.45로 companding, 280–3,600 Hz band limiting, 8 kHz down/up-sampling proxy, 20–40 dB channel noise 중 하나와 ±3 dB gain을 대칭 적용했다. RawBoost와 실제 codec stack은 아직 적용하지 않았다. clean/aug consistency pair를 추가할 때는 파형 변형 전에 crop 위치를 먼저 고정해 **정확히 같은 시간 구간**을 사용해야 한다. 실제 원천에 codec, noise removal, gain 같은 후처리를 한 것은 규칙상 Real이므로 이를 fake label로 사용하지 않는다. 신경망 source separator와 생성형 enhancement는 학습·추론 모두에서 제외한다.

## 8. Runtime, VRAM, disk budget

| 항목 | 목표/상한 |
|---|---:|
| 최종 ZIP | 목표 ≤9.70 GB, 규정 상한 <10 GB |
| 압축 해제 후 | 목표 <10.5 GB, 규정 상한 <32 GB |
| 신규 EAT-large | 1.238 GB |
| 신규 AASIST/head | <10 MB |
| 신규 branch 1,200개 추론 | ≤12분 |
| 전체 1,200개 추론 | 목표 ≤55분, 규정 상한 60분 |
| peak GPU memory | ≤20 GiB, L4 22.4 GiB 대비 여유 확보 |

기존 모델과 EAT-large는 순차 load/inference/unload하고 `inference_mode`, BF16, 길이 bucket batch를 사용한다. decoder는 현재 안정된 soundfile/ffmpeg 경로를 유지한다. v56의 EAT frontend는 `torchaudio`를 import하지 않는 순수 PyTorch Kaldi-compatible fbank를 사용한다. 이전 `torchaudio.compliance.kaldi.fbank` 결과와의 검증값은 max absolute error 2.57e-5, mean absolute error 2.84e-7, correlation 1.0이다. 제출 `requirements.txt`에는 서버 기본 버전의 torch/torchaudio를 다시 설치하지 않는다. EAT 실행에 필요한 source와 config는 라이선스와 함께 제출물 안에 넣고, 모든 weight는 오프라인에서 local path로만 읽는다.

빌드 직전에 다음을 자동 검사한다.

1. ZIP 루트가 정확히 `model/`, `script.py`, `requirements.txt`인지 확인
2. ZIP64 central directory, `unzip -t`, 모든 member CRC 확인
3. compressed/expanded byte size 확인
4. weight SHA-256와 pinned EAT revision 기록
5. 인터넷 차단 환경에서 3-format, mono/stereo, 4초/60초 smoke test
6. CUDA L4 상당 환경의 1,200-file wall-clock 및 peak VRAM 측정

## 9. Go/No-Go criteria

### 9.1 데이터와 학습 sanity gate

아래 하나라도 실패하면 fusion 평가로 가지 않는다.

- generator alias 정규화와 parent grouping unit test 통과
- train/hold-out 사이 parent, speaker/artist, generator-family intersection = 0
- crop-level label 재계산 unit test 통과
- 세 LOGO fold에서 direct expert Music EER 평균 ≤0.22, worst fold ≤0.30
- 어떤 generator fold도 v50 Music EER보다 0.005 초과 악화하지 않음
- clean/phone/codec 각 class의 logit이 collapse하지 않고 finite

### 9.2 fusion gate

허용된 train-derived LOGO dev에서 다음을 모두 만족한 weight 하나만 동결한다.

- Music EER: v50 대비 절대 0.040 이상 개선
- File EER: v50 대비 절대 0.025 이상 개선
- clean, telephone, codec, sequential, overlap 각 그룹 악화 ≤0.010
- 모든 channel을 합친 global EER도 개선
- clean 대 channel의 class-conditional normalized logit location 차이가 pooled SD의 0.15 이하이거나, v50 대비 25% 이상 감소
- Voice fake와 두 CPS column은 앵커 출력과 byte/hash identical

### 9.3 제출 gate와 목표치

현재 제출 결과에서 역산된 대략적 병목은 File EER 약 0.274, Voice EER 약 0.216, Music EER 약 0.371이다. ADS 0.82116을 넘으려면 현재보다 약 0.0848의 ADS 개선이 필요하다. Voice를 그대로 둘 때 **Music EER 0.371→0.221, File EER 0.274→0.194** 정도가 stretch target이며, 산식상 약 0.085 개선이다.

따라서 소폭 residual 개선만으로 1위를 주장해서는 안 된다. prospective 후보는 코드, weight hash, fusion weight를 먼저 동결한 뒤 한 번만 평가한다. ADS 절대 +0.020 이상, File/Music 모두 non-worse일 때만 최종 제출 후보로 승격한다. 이보다 작거나 한 component가 악화되면 v56은 폐기하고, 다음 단계로 EAT+XLS-R token fusion 또는 더 넓은 합법적 generator 데이터를 검토한다.

공식 리더보드의 현재 기준 0.837980/ADS 0.821160을 실제로 넘기기 전에는 SOTA라고 표현하지 않는다.

## 10. 구현 경계

이 문서는 설계만 동결한다. 구현 시에도 기존 v55 파일과 `artifactnet_detector.py`를 수정하지 않는다. 별도 v56 namespace에 다음처럼 둔다.

```text
src/eat_large_aasist.py
src/eat_large_aasist_inference.py
scripts/train_eat_large_aasist_v56.py
scripts/evaluate_eat_large_aasist_v56.py
scripts/build_eat_large_aasist_v56_submission.py
configs/eat_large_aasist_v56.yaml
tests/test_eat_large_aasist_v56.py
```

첫 구현 순서는 (1) EAT-large pinned weight와 preprocessing parity 검증, (2) 단일 10초 crop overfit test, (3) 세 LOGO fold, (4) 고정 weight fusion, (5) size/runtime smoke, (6) prospective 평가다. 이 순서에서 go gate를 넘기지 못하면 제출 ZIP을 만들지 않는다.

## 11. 2026-09-04 feasibility run 결과

### 11.1 실행 사양과 학습 이력

실제 구현은 원본 mixture만 입력하는 EAT-large 9-layer tap `[3, 7, 11, 15, 19, 20, 21, 22, 23]`, block 20–23 adapter, spectral/temporal GAT, 두 개의 heterogeneous master-node branch로 구성했다. EAT-large 309,409,295개 parameter는 freeze했고 신규 443,736개 parameter만 학습했다. 첫 feasibility 학습은 당시 대표 group 하나만 사용한 fold-0 optimization 8,686개 중 epoch당 balanced replacement sample 8,000개, batch 32, BF16, augmentation probability 0.45로 실행했다. 이후 all-view 감사를 통과하지 못했음을 발견했으므로, 아래 수치는 architecture feasibility와 후속 expert 결합 진단으로만 사용하고 엄격한 split의 checkpoint 선택 증거로 사용하지 않는다.

Music-LOGO와 Voice-LOGO에서 각각 1,200개만 고정 sampling해 checkpoint를 선택했다. epoch 4가 선택됐고, patience 3으로 epoch 7 뒤 종료했다. 전체 학습과 5,377개 authorized development 1-view 추론은 B200에서 458.77초가 걸렸다. epoch별 핵심 EER는 다음과 같다.

| Epoch | Loss | Music-LOGO File / Music | Voice-LOGO File / Voice | Selection |
|---:|---:|---:|---:|---:|
| 0 | 0.6493 | 0.3756 / 0.3685 | 0.5421 / 0.5653 | 0.4937 |
| 1 | 0.4252 | 0.2878 / 0.2308 | 0.4110 / 0.4305 | 0.6196 |
| 2 | 0.2929 | 0.2756 / 0.1931 | 0.4299 / 0.4660 | 0.6063 |
| 3 | 0.2281 | 0.2610 / 0.1818 | 0.3927 / 0.4184 | 0.6404 |
| **4** | **0.1845** | **0.2488 / 0.1691** | **0.3366 / 0.3985** | **0.6754** |
| 5 | 0.1640 | 0.2537 / 0.1739 | 0.3873 / 0.4354 | 0.6401 |
| 6 | 0.1329 | 0.2445 / 0.1685 | 0.3549 / 0.4508 | 0.6528 |
| 7 | 0.1336 | 0.2366 / 0.1591 | 0.3544 / 0.4455 | 0.6567 |

checkpoint는 1.8 MB이며 SHA-256은 `fc16d358f174141de40e0e4795097462ed6cd8c8efe390edd737dd3cc1f51ee9`이다. base EAT SHA-256은 `2c7660b3678817e1c4aec04826a1a0f573aa7d1eb82355fca92747cb2b11bce2`이다. 매 epoch 이력은 이제 atomic `history.partial.csv`로 flush하도록 구현했다.

### 11.2 direct expert 결과

아래는 어떤 앵커도 섞지 않은 direct v56 EER다. Voice는 representation 학습을 위한 보조 출력일 뿐 배포 대상이 아니다.

| Authorized development | File | Voice | Music | ADS |
|---|---:|---:|---:|---:|
| mixfake music | 0.2750 | 0.3581 | 0.2413 | 0.7185 |
| factorial mixed | 0.2571 | 0.3600 | 0.2286 | 0.7309 |
| external mixed | 0.2400 | 0.2400 | 0.2300 | 0.7630 |
| source-disjoint mixed | 0.2800 | 0.3300 | 0.2500 | 0.7190 |
| source-disjoint equal mixed | 0.2600 | 0.3500 | 0.1900 | 0.7430 |
| source-disjoint music | 0.1450 | – | 0.1450 | – |
| external mixed telephone | 0.2400 | 0.2350 | 0.2500 | 0.7580 |
| source-disjoint mixed telephone | 0.3200 | 0.3800 | 0.2800 | 0.6800 |
| telephone mixed | 0.2611 | 0.2767 | 0.2467 | 0.7401 |
| codec mixed | 0.3000 | 0.3700 | 0.2433 | 0.7030 |

codec mixed에서 exact v50은 File/Voice/Music `0.2333/0.2067/0.2733`, v55는 `0.2333/0.2067/0.2467`이었다. direct v56은 Music `0.2433`에는 경쟁력이 있지만 File `0.3000`과 Voice `0.3700`이 약하다. 따라서 direct expert와 Voice 전달은 명확한 **NO-GO**다.

### 11.3 frozen anchor residual 진단

codec mixed development에서 v50과 v56을 함께 sweep했을 때의 진단 최고점은 File weight 0.30, Music weight 0.15였다.

| Method | File | Voice | Music | ADS |
|---|---:|---:|---:|---:|
| exact v50 | 0.2333 | 0.2067 | 0.2733 | 0.7600 |
| v55 | 0.2333 | 0.2067 | 0.2467 | 0.7680 |
| v56 direct | 0.3000 | 0.3700 | 0.2433 | 0.7030 |
| v50 + v56 diagnostic | **0.2056** | **0.2067** | **0.2133** | **0.7919** |

그러나 이는 같은 codec dev에서 고른 optimistic 수치다. File의 channel별 v50→fusion 변화는 clean `0.1000→0.1333`, G.711 `0.2000→0.2333`, G.722 `0.1667→0.1333`, Opus-NB `0.3333→0.2722`, G.711→Opus transcode `0.3000→0.2667`이다. layout별로도 concurrent `0.2800→0.1800`은 좋아졌지만 partial overlap `0.2567→0.2600`, sequential `0.1433→0.1600`은 악화했다. 따라서 File weight 0.30은 배포 동결 기준을 통과하지 못한다.

RR/RF/FR/FF의 한 cell 안에서는 target이 상수라 EER를 정의할 수 없다. 다른 component를 고정한 controlled contrast EER는 다음과 같다. 표의 `fusion`도 위의 codec-dev 진단 weight일 뿐이다.

| Task / contrast | v50 | v55 | v56 direct | fusion |
|---|---:|---:|---:|---:|
| File RR vs FF | 0.1800 | 0.1800 | 0.2733 | 0.1667 |
| File RR vs FR | 0.2600 | 0.2600 | 0.3267 | 0.2600 |
| File RR vs RF | 0.2667 | 0.2667 | 0.3067 | 0.2200 |
| Music FR vs FF | 0.3067 | 0.2867 | 0.2600 | 0.2533 |
| Music RR vs RF | 0.2333 | 0.2200 | 0.2267 | 0.1800 |
| Voice RF vs FF | 0.2267 | 0.2267 | 0.4200 | 0.2267 |
| Voice RR vs FR | 0.1867 | 0.1867 | 0.3333 | 0.1867 |

이 결과는 v56이 fake-music 축인 RF/FF를 보완할 수 있지만, fake-voice/real-music인 FR의 File 판별을 해결하지 못한다는 뜻이다. 배포 함수가 Voice fake와 두 Presence column을 anchor에서 그대로 복사하고, unit test가 exact equality를 검사하므로 Voice residual의 File error propagation은 차단했다.

### 11.4 full development Music-only sweep

File weight를 0으로 고정하고, 모든 authorized development domain에서 exact-v47 Music anchor에 v56 Music만 결합했다. `w=0.10`은 pooled EER만 보면 `0.14819→0.14898`로 0.00079 악화하지만, domain mean `0.15889→0.14684`, worst `0.3100→0.2700`으로 좋아져 robust selection이 가장 높았다. `w=0.15`는 worst를 0.2600까지 낮췄지만 pooled EER가 0.15511로 더 악화했다.

| Music weight | Pooled EER | Domain mean | Domain worst | Robust selection |
|---:|---:|---:|---:|---:|
| 0.00 | **0.14819** | 0.15889 | 0.3100 | 0.80868 |
| 0.05 | 0.15155 | 0.15371 | 0.3000 | 0.81080 |
| **0.10** | 0.14898 | **0.14684** | 0.2700 | **0.82130** |
| 0.15 | 0.15511 | 0.14575 | **0.2600** | 0.82101 |
| 0.20 | 0.15689 | 0.15166 | 0.2700 | 0.81614 |
| 0.30 | 0.16795 | 0.15813 | 0.2700 | 0.80899 |

domain별 `w=0→0.10` Music EER는 codec `0.2733→0.2333`, factorial `0.2114→0.1943`, source-disjoint mixed `0.1600→0.1400`, source-disjoint phone `0.3100→0.2700`, telephone mixed `0.1867→0.1800`으로 개선했다. 반면 external mixed는 `0.1250→0.1350`으로 악화했다. 즉 낮은 Music residual은 domain-tail 보완 expert로는 유망하지만 pooled improvement와 모든-domain non-worse 기준은 통과하지 못했다.

v50+v57과의 후속 결합용 raw expert 배열은 `reports/eat_large_aasist_v56/seed00_eval_v3/raw_expert_logits.npz`에 `datasets`, `ids`, `voice_logits`, `music_logits`, `file_logits` key로 저장했다. SHA-256은 `54c31ac1d366e8d41048843b107ca9803a0c08cb9335345e716f73e6fcbc54df`이다.

### 11.5 strict all-token 재학습

위 feasibility 결과 뒤에 fallback split이 대표 group 하나만 검사한다는 문제를 발견했다. 코드를 모든 task-specific generator/identity/parent view의 connected component로 교정하고 같은 설정을 다시 학습했다. strict train은 2,908개로 줄었고, best epoch는 8이었다.

| Epoch | Selection | Music-LOGO File / Music | Voice-LOGO File / Voice |
|---:|---:|---:|---:|
| 0 | 0.56009 | 0.4030 / 0.4564 | 0.4340 / 0.4648 |
| 1 | 0.59639 | 0.3373 / 0.4561 | 0.3923 / 0.4417 |
| 2 | 0.62125 | 0.3413 / 0.4540 | 0.3619 / 0.3682 |
| 3 | 0.62752 | 0.3373 / 0.4513 | 0.3482 / 0.3522 |
| 4 | 0.61562 | 0.3567 / 0.4594 | 0.3482 / 0.3578 |
| 5 | 0.62523 | 0.3443 / 0.4470 | 0.3432 / 0.3634 |
| 6 | 0.62761 | 0.3333 / 0.4645 | 0.3426 / 0.3410 |
| 7 | 0.63630 | 0.3299 / 0.4379 | 0.3396 / 0.3507 |
| **8** | **0.63803** | **0.3264 / 0.4345** | **0.3488 / 0.3442** |
| 9 | 0.63555 | 0.3294 / 0.4379 | 0.3488 / 0.3459 |

Music-LOGO Music EER 0.4345는 sanity gate 0.22를 크게 넘는다. codec mixed direct 결과도 File/Voice/Music `0.3611/0.2733/0.4533`, ADS `0.62878`로 anchor보다 낮다. codec-dev fusion sweep은 File weight 0, Music weight 0.10에서 ADS 0.7620으로 exact v50 0.7600보다 소폭 높지만 v55 0.7680보다 낮았다.

full authorized development의 exact-v47 Music-only sweep에서 strict `w=0.10`은 pooled EER `0.148192→0.147797`, domain mean `0.158889→0.146995`, worst `0.3100→0.2667`이었다. 숫자상 작은 tail 이득은 있으나 다음 안전성 실패가 있다.

- external mixed telephone: `0.1250→0.1400`
- codec G.711: `0.2500→0.2667`
- codec Opus-NB: `0.3500→0.3667`
- mixfake music: `0.07625→0.07875`
- telephone mixed: `0.18667→0.1900`

strict checkpoint SHA-256은 `ddaca13848706329d9601ea7d73bd125ecd8d016bed9532db1725f069eb92adc`이다. 재현용 raw strict logit은 `reports/eat_large_aasist_v56/seed00_strict_eval_v1/raw_expert_logits.npz`, SHA-256 `3691fb1923e27bc59168d7a2f2d548391168a7bab3a8504e600eb5105c0eac84`에 저장했다. 이 파일은 rejected 연구 artifact이며 후보 결합 선택에는 사용하지 않는다.

### 11.6 최종 판정

- v56 direct model: **REJECT**.
- v56 File residual: **weight 0으로 고정**.
- v56 Music residual: **weight 0으로 고정**. pooled +0.0004 수준의 이득보다 domain/channel 악화가 크다.
- v56 Voice와 CPS: 새 모델을 전달하지 않고 frozen anchor를 유지한다.
- v50+v57 최종 후보에 v56을 포함하지 않는다.
- ZIP/API submission은 만들거나 실행하지 않았다.

공식 SoundMusic-EAT 방향의 구조는 구현 가능하고 일부 Music tail에 독립 정보가 있었지만, 엄격한 all-view split에서는 unseen Music generator/source에 일반화하지 못했다. 추가 학습 없이 이 branch를 종료한다.
