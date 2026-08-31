# 다중 생성기 1,200개 평가셋 설계

## 목적

새 평가셋의 목적은 로컬 점수를 높게 만드는 것이 아니라 다음 질문을 실제 대회
정의대로 분리해서 답하는 것이다.

1. 음성만 있을 때 새로운 TTS 생성기를 탐지하는가?
2. 음악만 있을 때 새로운 음악 생성기를 탐지하는가?
3. 실제/가짜 음성과 실제/가짜 음악의 네 조합을 모두 구분하는가?
4. 동시 재생, 일부 구간 중첩, 순차 재생 중 어디서 실패하는가?
5. MP3, OGG, 전화 채널, 스테레오, 잡음에서도 순위가 유지되는가?

평가 오디오를 만들 때 Demucs, SAM-Audio 같은 음원 분리를 전혀 사용하지
않는다. 따라서 분리 모델이 생성한 흔적을 정답 신호로 학습하거나, 원래의
딥페이크 흔적을 지우는 문제가 없다.

## 음성 bank

`scripts/prepare_multigen_voice_eval.py`가 AIHub 원화자 16명을 화자 단위로
고정 분할한다.

| split | 화자 수 | 실제 음성 수 | 화자당 생성 문장 |
| --- | ---: | ---: | ---: |
| dev | 5 | 30 | 3 |
| holdout | 5 | 30 | 3 |
| locked | 6 | 36 | 3 |

같은 문장을 실제 음성과 모든 TTS에 공통으로 사용한다. 따라서 문장 내용만으로
real/fake를 맞힐 수 없다. 참조 음성과 평가 문장은 서로 다른 발화다.

현재 생성 계열은 다음과 같다.

- 화자별 fine-tuned Qwen3-TTS 1.7B: 48개
- 공식 Qwen3-TTS CustomVoice 0.6B: 48개
- F5-TTS v1 Base zero-shot voice cloning: 48개
- CosyVoice 3 0.5B multilingual zero-shot: 48개

Qwen fine-tuned 모델은 해당 AIHub 화자의 학습 자료로 미세조정되었으므로, 같은
원천 코퍼스에 대한 생성이라는 제한이 있다. 이 계열만으로 결론을 내리지 않고
독립 사전학습 모델인 Qwen CustomVoice, F5, CosyVoice 결과를 생성기별로 따로
보고한다.

Typecast는 현재 API 자격증명이나 내려받은 생성 파일이 없어 포함했다고
표기하지 않는다. 추후 실제 Typecast 결과가 제공되면 동일한
`generation_manifest.csv`의 `JOB_ID`에 맞춰 별도 생성기 행으로 추가한다.
`scripts/import_external_tts_eval.py`가 `JOB_ID,AUDIO_FILE,GENERATOR` CSV를 받아
동일한 16 kHz/길이 검증과 manifest 변환을 수행한다.

## 음악 bank

기존 `source_disjoint_music_v1`은 SONICS의 Suno/Udio 보컬 곡을
`VOICE_PRESENT=0`으로 기록했다. SONICS 메타데이터의 `no_vocal` 값은 실제로
전부 false이므로, 이 데이터는 순수 음악 평가에 사용하면 안 된다.

새 `scripts/build_echoes_fma_paired_eval.py`는 다음 절차를 사용한다.

1. Echoes 생성곡의 `original_audio` 제목·아티스트를 공식 FMA metadata와
   정규화해 매칭한다.
2. 실제 원곡이 `fma_small.zip`에 있는 경우만 사용한다.
3. 과거 `echoes_v1`, `source_disjoint_music_v1`, `multigen_music_v2`에 사용된
   원곡 그룹을 제외한다.
4. 실제 원곡과 생성곡 모두 원본 파형에서 frozen PANNs voice score가 0.20
   이하인 경우만 채택한다.
5. 실제/생성 양쪽에 동일한 16 kHz, mono, 12초 crop, FLAC 처리를 적용한다.

결과는 과거 평가와 겹치지 않는 source-matched FMA 원곡 25개 그룹과 추가
실제 instrumental 150개를 포함한다. 전체는 175개 원천 그룹, 실제 175개,
Echoes 생성 242개(총 417개)다.
생성기는 ACE-Step, AudioLDM, DiffRhythm, MusicGen, SongGen, Suno, Udio,
Stable Audio 등 12종이다. 같은 FMA 원곡과 그 생성 변형은 반드시 같은 split에만
존재한다. 생성 오디오에도 보컬이 섞인 경우는 순수 음악 bank에서 제외한다.

사용자가 제공한 `data/suno_music_with_vocals`의 13곡은 별도 OOD 평가에
사용한다. 이 경우 Suno가 보컬과 반주를 함께 생성했으므로 정답은
`VOICE_FAKE=1`, `MUSIC_FAKE=1`, `FILE_FAKE=1`이다. 모든 파일에서 fake 점수가
높아야 하지만, 단일 생성기·단일 클래스 자료이므로 EER 선택이나 학습에는
사용하지 않는다.

## 최종 1,200개 factorial 구성

각 split은 정확히 400개이며 전체 구성은 다음과 같다.

| 구성 | split당 | 전체 |
| --- | ---: | ---: |
| 음성 단독 | 50 | 150 |
| 음악 단독 | 50 | 150 |
| 완전 동시 혼합 | 100 | 300 |
| 25/50/75% 부분 중첩 | 100 | 300 |
| 음성→음악 또는 음악→음성 순차 | 100 | 300 |

각 혼합 방식 안에는 `(voice real/fake) × (music real/fake)` 네 조합이 정확히
25개씩 들어간다. 음성 단독과 음악 단독도 real/fake가 25개씩이다. 따라서
Voice EER과 Music EER의 양성·음성 클래스가 정확히 균형을 이룬다.

실제 비공개 데이터가 균형이라는 뜻은 아니다. 균형화는 방법 간 비교의 분산을
줄이기 위한 실험 설계이며, 전체 점수를 비공개 점수의 직접 추정치로 사용하지
않는다. `slice_manifest.csv`와 `slices/*.csv`는 다음 16개 원자 셀을 split별로
따로 가리킨다(오디오는 복제하지 않는다).

- voice-only real/fake 2개와 music-only real/fake 2개
- concurrent/partial-overlap/sequential 각각의 voice-real/music-real,
  voice-fake/music-real, voice-real/music-fake, voice-fake/music-fake 12개

단일 라벨 셀에서는 EER을 정의할 수 없으므로 점수 분포를 보고, EER은 다른
성분의 라벨을 고정한 대조군끼리 계산한다. 예를 들어 voice-fake 성능은
`music-real`을 고정한 `RR 대 FR`과 `music-fake`를 고정한 `RF 대 FF`를 따로
계산한다. File EER도 `RR 대 FR`, `RR 대 RF`, `RR 대 FF`를 분리한다. 이렇게
해야 가짜 음성+진짜 음악과 진짜 음성+가짜 음악의 실패를 평균 속에 숨기지 않는다.

동시·부분 중첩 샘플은 음성 대 음악 SNR을 `-10, -5, 0, 5, 10 dB`로
변화시킨다. 부분 중첩은 음성 우선/음악 우선과 25/50/75% 중첩을 포함하고,
순차 샘플은 양쪽 순서와 0/0.2/0.5초 간격을 포함한다.

다음 채널 조건은 라벨과 독립적으로 거의 같은 수로 순환 배치한다.

- mono 16 kHz FLAC
- 2채널 WAV
- 64 kbps MP3
- 48 kbps OGG
- 300–3400 Hz 전화 대역 및 8 kHz 왕복 resampling
- 18/24/30 dB 잡음과 약한 clipping

## 누수 방지와 사용 규칙

- 화자는 한 split에만 존재한다.
- FMA 원곡과 모든 생성 변형은 한 split에만 존재한다.
- `dev`, `holdout`, `locked`의 원천 오디오는 서로 공유하지 않는다.
- 생성 문장과 대응 실제 문장은 같은 split 안에만 존재한다.
- 이 bank와 최종 1,200개 truth는 `configs/data_partitions.yaml`에서 보호한다.
- `locked`는 후보, threshold, ensemble weight를 모두 고정한 뒤 한 번만 본다.
- 평가셋은 학습, pseudo-label, calibration에 넣지 않는다.

## 출처와 이용조건

- Qwen3-TTS: <https://github.com/QwenLM/Qwen3-TTS> (Apache-2.0)
- F5-TTS: <https://github.com/SWivid/F5-TTS> (코드 MIT, 공개 checkpoint는
  CC-BY-NC이므로 대회 이용조건을 별도로 확인)
- CosyVoice: <https://github.com/FunAudioLLM/CosyVoice> (Apache-2.0)
- FMA: <https://github.com/mdeff/fma> (각 음원의 원작자별 라이선스)
- AIHub 한국어 음성: AIHub 데이터 이용약관 적용

Echoes와 사용자가 제공한 Suno 파일도 원 배포처의 데이터·생성 서비스
이용조건을 2차 보고서에 개별 기록해야 한다. 이 문서는 기술적 provenance를
기록하며, 각 음원의 재배포 권한을 대신 보증하지 않는다.

## 재현 명령

음성 원본과 생성 job을 준비한 뒤 각 생성기 스크립트를 실행하고 manifest를
합친다.

```bash
python scripts/prepare_multigen_voice_eval.py \
  --voices-json /home/nas_main/kyudanjung/DATASET/speech/models_release/VOICES.json \
  --speech-root /home/nas_main/kyudanjung/DATASET/speech \
  --model-root /home/nas_main/kyudanjung/DATASET/speech/models_release \
  --output-dir data/eval/multigen_tts_pool_v1

python scripts/finalize_multigen_voice_eval.py \
  --pool-dir data/eval/multigen_tts_pool_v1 \
  --output-dir data/eval/multigen_voice_v2 \
  --generator-manifest \
    data/eval/multigen_tts_pool_v1/qwen_finetuned.csv \
    data/eval/multigen_tts_pool_v1/qwen_custom.csv \
    data/eval/multigen_tts_pool_v1/f5.csv \
    data/eval/multigen_tts_pool_v1/cosy.csv
```

음악 bank와 최종 평가셋은 다음과 같이 만든다.

```bash
python scripts/build_echoes_fma_paired_eval.py \
  --echoes-zip data/external/echoes/Echoes.zip \
  --fma-zip data/external/sonics_eval/fma_small.zip \
  --fma-metadata-zip /home/nas_main/kyudanjung/tts_eval_tools/fma_metadata.zip \
  --output-dir data/eval/echoes_fma_paired_v3 \
  --exclude-truth \
    data/eval/echoes_v1/truth.csv \
    data/eval/source_disjoint_music_v1/truth.csv \
    data/eval/multigen_music_v2/truth.csv \
  --panns-dir models/panns

python scripts/build_factorial_eval_1200.py \
  --voice-bank data/eval/multigen_voice_v2 \
  --music-bank data/eval/echoes_fma_paired_v3 \
  --output-dir data/eval/factorial_eval_1200_v2
```

## 평가 원칙

전체 점수 외에 반드시 `MIX_MODE`, `CHANNEL`, `VOICE_GENERATOR`,
`MUSIC_GENERATOR`, SNR, 중첩률별 File/Voice/Music EER을 함께 본다. 개선안은 dev에서
선택하고 holdout에서 방향만 확인한 뒤, locked에서 기존 리더보드 순위
`twin > musicorig > routed-v14`가 재현되는지 검사한다. 이 순서가 재현되지 않으면
로컬 절대 점수를 제출 점수의 추정치로 사용하지 않는다.

예측을 만든 뒤 원자 셀과 통제 대조 EER은 다음 명령으로 저장한다.

```bash
python scripts/evaluate_factorial_slices.py \
  predictions.csv data/eval/factorial_eval_1200_v2/truth.csv \
  --output-dir reports/factorial_eval
```

`factorial_contrasts.csv`가 RR 대 FR/RF/FF 및 조건부 Voice/Music EER을,
`factorial_cell_distributions.csv`가 16개 셀별 확률의 10/50/90 백분위수를
담는다. 제출 후보 선택에서는 dev의 평균만 최적화하지 않고 대조 EER 중
최악값도 함께 낮춘다.
