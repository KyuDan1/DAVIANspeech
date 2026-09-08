# All-type 학습 데이터 인벤토리 v56

작성 시점: 2026-09-04. 이 문서는 `configs/data_partitions.yaml`에 등록된 역할과
로컬의 기존 v47 feature cache만 읽어서 작성했다. `codec_mixed_blind_v8` 및 모든
`retrospective_diagnostic`, `invalid_eval` 데이터는 집계·학습·모델 선택에서
제외했다. 이 조사에서는 학습, detector inference, ZIP 생성, 대용량 복사를 하지
않았다.

## 결론

오늘 바로 시작할 수 있는 가장 큰 **완전 캐시된** 학습 풀은 train 21,304행이다.
development 5,377행도 같은 세 feature stream이 모두 캐시되어 있다. 캐시는 총
26,681행이며 truth ID와 feature ID가 22개 manifest 모두 정확히 일치한다.

다만 행 수를 그대로 독립 샘플 수로 보면 안 된다.

- train 21,304행은 exact audio hash 기준 21,267개다.
- development 5,377행은 4,768개다. `telephone_mixed_dev_v1` 600행은 다른 두
  telephone development manifest의 합집합과 exact-audio 중복이다.
- router_train까지 포함한 최대 허용 train은 24,904행, exact-audio 24,825개다.
  단, router 3,600행은 아직 v47 세-stream cache가 없다.
- train과 development/OOD 사이의 등록 identity overlap은 0이고, 캐시된
  train/development 사이 exact-audio SHA256 overlap도 0이다.
- 반대로 train 내부에는 같은 원천으로 만든 layout/codec 파생본이 많다.
  무작위 row split은 누수다. source/parent/audio-hash group 단위로 묶어야 한다.

따라서 첫 실제 실행은 메모리와 중복을 고려해 train 18,744행
(`temporal_mixed_train_v1`만 제외), development 4,577행
(`source_disjoint_mixed_equal_v1`, 중복 합집합인 `telephone_mixed_dev_v1` 제외)을
권장한다. 두 temporal 버전은 동일 ID 집합과 3,853개 identity token을 공유한다.
v2가 같은 5-layout 구조에 `PAIR_GROUP`까지 제공하므로 v2를 우선하는 편이 낫다.

## 역할별 규모

`V/M/X`는 voice-only/music-only/mixed다. RR/RF/FR/FF는 mixed 행만 센다.

| 역할 | 총 행 | V | M | X | RR | RF | FR | FF | v47 3-stream cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 21,304 | 728 | 600 | 19,976 | 4,965 | 4,965 | 5,026 | 5,020 | 전부 있음 |
| router_train | 3,600 | 1,126 | 874 | 1,600 | 434 | 426 | 387 | 353 | 없음 |
| training_validation | 1,472 | 64 | 0 | 1,408 | 336 | 336 | 368 | 368 | 없음 |
| development | 5,377 | 316 | 561 | 4,500 | 1,125 | 1,125 | 1,125 | 1,125 | 전부 있음 |
| ood_holdout | 644 | 140 | 191 | 313 | 75 | 75 | 75 | 88 | 없음 |
| stress_eval | 841 | 176 | 400 | 265 | 50 | 50 | 50 | 115 | 없음 |

현재 main train은 mixed가 93.8%다. all-type head의 presence를 제대로 학습하려면
router_train의 V/M 2,000행을 같은 source-group으로 묶어 추가하는 것이 가장 큰
보완점이다. `training_validation`은 train source pool에서 나온 row-disjoint
자료이므로 최적화 상태 확인에는 쓸 수 있지만 source 일반화 근거로 쓰면 안 된다.

## Train 및 router_train 상세

| dataset | 행 | V/M/X | mixed RR/RF/FR/FF | 시간·채널 다양성 | cache |
|---|---:|---:|---:|---|---|
| forensic_call_train_v1 | 384 | 128/0/256 | 32/32/96/96 | 통화 대화·순차 음악 metadata | O |
| temporal_mixed_train_v2 | 2,560 | 0/0/2,560 | 각 640 | concurrent/overlap/sequential/sparse-V/sparse-M 각 512 | O |
| temporal_mixed_train_v1 | 2,560 | 0/0/2,560 | 각 640 | 위와 동일, v2와 source 중복 큼 | O |
| external_mixed_train_v1 | 800 | 0/0/800 | 각 200 | FLAC 16 kHz | O |
| mixed_devvoice_train_v1 | 800 | 0/0/800 | 각 200 | FLAC 16 kHz | O |
| mixed_fmc_music_train_v1 | 800 | 0/0/800 | 각 200 | FLAC 16 kHz | O |
| mixfake_music_train_v1 | 8,000 | 0/0/8,000 | 각 2,000 | WAV 16 kHz | O |
| telephone_mixed_train_v1 | 2,400 | 0/0/2,400 | 각 600 | telephone 8 kHz 계열 | O |
| channel_invariant_factorial_train_v1 | 3,000 | 600/600/1,800 | 453/453/450/444 | clean 1,000 + 12 channel 계열 2,000 | O |
| phone_router_voice_train_v1 | 726 | 726/0/0 | - | 원음 voice; 6 fake family | X |
| multigen_music_presence_train_v1 | 474 | 0/474/0 | - | 원음 music; Suno/Udio | X |
| phone_presence_factorial_train_v1 | 2,400 | 400/400/1,600 | 434/426/387/353 | 5 phone channel × 480 | X |

`channel_invariant_factorial_train_v1`의 3,000행은 1,000 parent에서 나온 paired
view다. router_train도 main train 원천을 의도적으로 재사용한다. 이들은 데이터
증가분이 아니라 invariance supervision으로 취급하고 parent/source별 총 weight를
동일하게 해야 한다.

## Development 상세

| dataset | 행 | V/M/X | mixed cell | 핵심 역할 |
|---|---:|---:|---:|---|
| mixfake_music_dev_v1 | 1,600 | 0/0/1,600 | 각 400 | 6 fake-music family mixed 검증 |
| factorial_eval_1200_v2 dev | 400 | 50/50/300 | 각 75 | all-type, 3 layout, 6 codec |
| external_mixed_v1 | 400 | 0/0/400 | 각 100 | clean mixed domain |
| source_disjoint_mixed_v1 | 200 | 0/0/200 | 각 50 | source-disjoint mixed |
| source_disjoint_mixed_equal_v1 | 200 | 0/0/200 | 각 50 | 위와 source 중복, 다른 mixture |
| source_disjoint_music_v1 | 400 | 0/400/0 | - | music-only presence/authenticity |
| asvspoof_voice_wav_dev_v1 | 176 | 176/0/0 | - | voice-only, codec/attack 다양성 |
| external_mixed_v1_telephone_v1 | 400 | 0/0/400 | 각 100 | external mixed의 phone paired view |
| source_disjoint_mixed_v1_telephone_v1 | 200 | 0/0/200 | 각 50 | source-disjoint mixed phone view |
| telephone_mixed_dev_v1 | 600 | 0/0/600 | 각 150 | 위 두 phone set과 exact-audio 전부 중복 |
| multigen_voice_v2 dev | 90 | 90/0/0 | - | 4 modern TTS, factorial과 source 중복 |
| echoes_fma_paired_v3 dev | 111 | 0/111/0 | - | 12 music family, factorial과 source 중복 |
| codec_mixed_dev_v4 | 600 | 0/0/600 | 각 150 | 3 layout × 5 paired codec |

권장 4,577행 development는 위에서 `source_disjoint_mixed_equal_v1`과
`telephone_mixed_dev_v1`만 제외한다. exact-audio는 4,569개이며, 남은 8개 중복은
factorial과 multigen voice 사이에 있다. 평가 시 pooled row score 하나만 보지 말고
dataset-normalized mean, worst-domain ADS, 유형별/구성별 EER를 같이 사용한다.

## Generator별 명시 annotation

아래 수는 generator가 truth에 명시된 fake component/sample occurrence 수다.
두 화자 통화 데이터의 `FIRST/SECOND_GENERATOR`는 두 component occurrence로 센다.
generator 열이 없는 기존 mixed corpus는 `unknown`으로 남겨야 하며 추측 라벨을
붙이면 안 된다.

### Train voice

| Generator | 수 | Generator | 수 |
|---|---:|---|---:|
| ASVspoof2019-LA | 4,000 | StyleTTS2 | 373 |
| OpenAudio-S1-mini | 344 | SpeechT5 | 309 |
| IndexTTS | 284 | LLaSA | 267 |
| OpenVoice-V2 | 249 | CosyVoice2 | 248 |
| MaskGCT | 247 | F5-TTS | 231 |
| XTTS-v2 | 202 | FireRedTTS | 190 |

명시된 voice occurrence는 총 6,944, 12개 family다.

### Train music

| Generator | 수 | Generator | 수 |
|---|---:|---|---:|
| suno | 838 | udio | 828 |
| audioldm2 | 807 | MusicGen_medium | 796 |
| musicldm | 781 | mustango | 763 |
| stableaudio | 210 | mubert | 203 |
| producer | 201 | elevenlabs | 181 |
| brev | 180 | audioldm | 175 |
| acestep | 161 | stable_audio_open | 145 |
| songgen | 143 | musicgen | 139 |
| diffrhythm | 137 |  |  |

명시된 music occurrence는 총 6,688, 17개 family다.

### Development generator coverage

- Voice 1,421 occurrence: ASVspoof2019-LA 800, Qwen3-TTS/CosyVoice3/F5-TTS
  4계열 235, A07–A19 attack 13계열 311, HUB/SPO/Task attack 45계열 75.
- Music 1,530 occurrence: musicldm 194, MusicGen_medium/audioldm2/mustango 각
  192, udio 152, suno 150, chirp-v3.5 100, udio-120s 100,
  stable_audio_open 60, diffrhythm 30, audioldm 28, musicgen 26, producer와
  stableaudio 각 19, acestep 17, mubert와 songgen 각 16, elevenlabs 14, brev 13.

## Codec/channel별 행 수

| 역할 | channel/codec별 행 수 |
|---|---|
| train | unspecified 5,504; FLAC16k 2,400; WAV16k 8,000; telephone8k 2,400; clean 1,000; resample8k 180; PSTN bandpass 178; G.726 177; random bandpass 169; packet-loss G.711 169; μ-law 166; packet-loss Opus 164; A-law 163; FFT narrowband 163; G.711 μ-law 160; Opus NB 8k 158; Opus NB 12k 153 |
| router_train | unspecified voice 726; common FLAC16k 474; Opus NB/resample8k/PSTN/G.711/G.726 각 480 |
| development | WAV16k 1,600; FLAC16k 800; common FLAC16k 511; telephone8k 1,200; ASV WAV 176; factorial 6 codec 66–67씩; codec-v4 clean/G.711/G.722/Opus/double-transcode 각 120; unspecified multigen voice 90 |
| OOD | factorial 6 codec 66–67씩; multigen voice 90; common FLAC16k 154 |
| stress | telephone8k 789; Suno vocal wav/flac/mp3/ogg 각 13 |

`CODEC=flac16k`인 telephone 파생 manifest도 있으므로 `CODEC`만 사용하지 말고
`STRESS_VARIANT`, `CHANNEL`, `PARENT_ID`를 우선 결합해 channel environment를
만들어야 한다.

## 기존 feature/cache 상태

Canonical root는 `reports/v47_anchor_cache_train_dev_v2/cache/`다. `anchor/`,
`xlsr/`, `eat/`, `spear/` 아래 symlink를 그대로 trainer에 넘길 수 있어 복사가
필요 없다.

| stream | schema | 현재 coverage |
|---|---|---:|
| exact v47 anchor | 5 probability columns | 26,681 |
| XLS-R | `[N,W,1920]` float32, W=2/3/5/6/8/9 + mask/start | 26,681 |
| SPEAR | `[N,3,8,4,4,64]` float16 + mask | 26,681 |
| EAT temporal | `[N,3,6,2,38,128]` float16 | 26,681 |
| EAT spectral | `[N,3,6,2,8,128]` float16 | 26,681 |

22개 dataset에서 세 cache의 ID/order를 truth와 대조했고 불일치는 0이었다.
merged dataset cache는 약 7.2 GiB, shard까지 포함한 전체 작업 디렉터리는 약
13 GiB다. EAT+SPEAR 고정 payload만 전 train+권장-dev에서 약 11.5 GiB이며,
현재 trainer가 dataset block과 concatenated block을 동시에 잡기 때문에 28 GiB
RAM에서는 full 21,304행 실행이 빠듯하다. 첫 실행에서 temporal-v1을 제외하면
train+권장-dev 고정 payload가 약 10.3 GiB로 내려가 안전 여유가 생긴다.

## Leakage-safe split 판정

현재 `src/data_guard.py`를 전체 train/router manifest에 다시 실행해 12/12 PASS를
확인했다.

- train/router ↔ development identity overlap: 0
- train/router ↔ OOD identity overlap: 0
- cached train ↔ development exact-audio hash overlap: 0
- expanded train+router: 24,904 rows / 24,825 unique exact-audio hashes
- development 내부: 5,377 rows / 4,768 unique exact-audio hashes

그러나 이는 등록된 ID column과 exact rendered-audio hash에 대한 보장이다.
namespace가 다른 동일 원천을 자동으로 알아내지는 못한다. 새 split/build에서는
반드시 아래 group을 union하여 한쪽 역할에만 둔다.

1. `VOICE_SOURCE_ID`, `VOICE_SPEAKER`, `FIRST/SECOND_SOURCE_ID`,
   `FIRST/SECOND_GROUP`, `VOICE_GROUP`
2. `MUSIC_SOURCE_ID`, `MUSIC_GROUP(_ID)`, canonical song ID
3. `PAIR_GROUP`, `MIXTURE_ID`, `PARENT_ID`, codec parent
4. 원천 및 rendered audio SHA256

## 오늘 실행할 학습 설계

### 1. 첫 실행: 가장 큰 안정적 cached set

- Train 18,744: 등록 train 전체에서 `temporal_mixed_train_v1`만 제외.
- Dev 4,577: development에서 `source_disjoint_mixed_equal_v1`과 exact duplicate
  union인 `telephone_mixed_dev_v1` 제외.
- 모델은 `ThreeStreamAnchorResidualHead`를 v47 anchor의 bounded residual로 학습.
- loss는 presence-mask가 적용된 Voice/Music authenticity, File authenticity,
  `File ≈ 1-(1-Voice)(1-Music)` 일관성, paired codec latent/logit consistency를 사용.
- sampler는 dataset × type × RR/RF/FR/FF × layout × channel × generator에
  Group-DRO를 적용하고, 같은 source/parent의 모든 view는 atomic group으로 유지.
- early stopping은 pooled ADS 하나가 아니라 normalized-domain mean과 worst-domain
  ADS를 함께 사용. OOD/stress/retrospective는 early stopping에 사용하지 않는다.

기존 실행기는 `scripts/train_three_stream_anchor_residual.py`이며 먼저
`--validate-only --generator-aware`로 모든 cache/atomic-group contract를 검증한다.
Cache root는 각각 다음을 쓴다.

```text
anchor: reports/v47_anchor_cache_train_dev_v2/cache/anchor
xlsr:   reports/v47_anchor_cache_train_dev_v2/cache/xlsr
eat:    reports/v47_anchor_cache_train_dev_v2/cache/eat
spear:  reports/v47_anchor_cache_train_dev_v2/cache/spear
```

### 2. 같은 날 확장

router_train 3,600행에 동일 cache를 한 번 추출한 뒤 main train에 합친다. 우선순위는
`phone_presence_factorial_train_v1` → voice-only → music-only다. 새 독립 데이터로
세지 말고 기존 source와 묶어 presence/channel auxiliary loss에만 높은 비중을 준다.
Full 24,904행을 쓰려면 trainer를 dataset-streaming 또는 disk-backed tensor 방식으로
바꿔 peak CPU RAM을 낮춘 후 실행한다.

### 3. Prospective all-type v9

후보와 hyperparameter를 development에서 완전히 동결한 **다음에만** 새 bank를
예약한다. 권장 크기는 120 base × 5 paired channel = 600 sample이다.

| base 구성 | 수 |
|---|---:|
| voice-only real/fake | 12 / 12 |
| music-only real/fake | 12 / 12 |
| mixed RR/RF/FR/FF × concurrent/partial/sequential | 각 cell-layout 6, 총 72 |

채널은 clean, G.711, G.722, Opus 8 kHz, double-transcode를 동일 parent에 적용한다.
train/dev/OOD/stress와 모든 retrospective(v8 포함)의 source, speaker, reference,
canonical song, source/rendered SHA256을 보호한다. Suno/Udio vocal song은 RF에 절대
넣지 않고 두 component가 모두 fake인 FF로만 표기한다. concurrent FF 일부는
원본 joint song을 그대로 써 separation artifact 없는 성능도 확인하고,
partial/sequential FF는 서로 독립적인 fake voice + fake instrumental을 합성해
separator를 사용하지 않는다.

예약→source/hash validation→semantic presence screen→render→hash lock까지 detector
출력을 0회 유지한다. 한 frozen candidate만 실행하고 즉시 retrospective로 옮긴다.

## 금지 사항

- `codec_mixed_blind_v8` 또는 다른 retrospective 결과로 weight/router/checkpoint 선택
- source 파생본을 train/dev 양쪽에 나누는 row-random split
- Suno/Udio vocal song을 music-only 또는 RF로 라벨링
- generator metadata가 없는 sample에 filename 추측으로 generator 라벨 추가
- OOD/stress를 epoch 선택이나 반복 ablation에 사용
