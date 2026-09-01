# 대회 규정 (원문 기준)

로컬 평가셋과 제출 패키지가 반드시 만족해야 하는 조건. 추정이 아니라 주최측
명시 사항이며, 여기서 벗어나면 로컬 측정이 대회를 재현하지 못한다.

## 1. 평가 데이터

- 총 **1,200개** 오디오 파일
- 길이 **4초 이상 1분 이하**
- **16 kHz** 샘플링레이트로 표준화
- **채널은 샘플별로 모노/스테레오 모두 존재**
- **MP3, WAV, FLAC 등 다양한 확장자**로 구성되며 모델이 전부 처리해야 함
- 일부 샘플에 **전화채널 오디오** 포함

## 2. 오디오 유형

| 유형 | 정의 | VOICE_PRESENT | MUSIC_PRESENT |
| --- | --- | :---: | :---: |
| 음성 | 사람의 발화 **또는 보컬만** 포함 | 1 | 0 |
| 음악 | **보컬이 없는** 반주·악기음만 포함 | 0 | 1 |
| 혼합 | 음성과 음악이 동시에 또는 순차적으로 포함 | 1 | 1 |

> **보컬은 음성 성분으로 분류합니다. 따라서 보컬과 반주가 함께 포함된 노래는
> 혼합 오디오에 해당합니다.**

이 한 줄이 실수를 부르는 지점이다. AI 노래는 "음악만"이 아니라 **혼합**이고,
보컬과 반주가 모두 생성됐으므로 `VOICE_FAKE = 1`, `MUSIC_FAKE = 1`이다. 노래를
music-only로 매기면 voice 부분집합에서 통째로 빠져 VOICE_EER과 MUSIC_EER이 모두
잘못된 모집단에서 계산된다. `complike_v1`이 정확히 그 오류를 갖고 있었다.

## 3. Real(0) / Fake(1) 기준

- AI로 **생성된** 음성 또는 음악 성분은 FAKE
- AI로 생성되지 않은 실제 원천은 REAL
- 음성과 음악 중 **하나라도 FAKE이면 파일 전체가 FAKE**
- **실제 원천에 품질 개선, 잡음 제거, 음량 조정 등 성분 자체를 새로 생성하지 않는
  후처리만 적용된 경우는 REAL**

마지막 항목은 미검증 위험이다. 스푸핑 탐지기가 enhancement 아티팩트를 생성
아티팩트로 오인하는 것은 알려진 실패 모드이고, 우리는 아직 이 조건을 시험하지
않았다.

## 4. 예측 항목

파일마다 0~1 확률 5개: `FILE_FAKE_PROB`, `VOICE_FAKE_PROB`, `MUSIC_FAKE_PROB`,
`VOICE_PRESENT_PROB`, `MUSIC_PRESENT_PROB`.

## 5. 코드 제출 제약

| 항목 | 한도 | 현재 |
| --- | --- | --- |
| 전체 추론 시간 | ≤ 60분 (1,200개) | 31~33분 |
| 패키지 설치 시간 | ≤ 10분 | `onnxruntime-gpu` 한 줄 |
| 제출 파일 용량 | ≤ 10GB (해제 후 ≤ 32GB) | 4.43 GiB |
| 실행 환경 | 오프라인 (설치 외 인터넷 불가) | 검증 완료 |
| 하드웨어 | 6 vCPU, 28GB RAM, **L4 22.4GiB VRAM** | XLS-R fp32 로드 시 9GB |

L4 22.4GiB 제약 때문에 SAM-Audio는 배제된다. 가중치 상주만 24~31GB다.

## 6. 입력 포맷 검증 결과

대회가 명시한 조건을 실제로 만들어 오프라인으로 통과시켰다.

| 파일 | 포맷 | 채널 | 결과 |
| --- | --- | :---: | --- |
| `A_mp3_mono.mp3` | MP3 | 1 | 통과 |
| `B_mp3_stereo.mp3` | MP3 | 2 | 통과 |
| `C_flac_mono.flac` | FLAC | 1 | 통과 |
| `D_flac_stereo.flac` | FLAC | 2 | 통과 |
| `E_wav_stereo.wav` | WAV | 2 | 통과 |
| `F_phone.wav` | WAV (8k 대역 왕복) | 1 | 통과 |
| `G_ogg.ogg` | OGG | 1 | 통과 |

그때까지 WAV 모노로만 시험하고 있었으므로 이는 미검증 위험이었다. 재현:

```bash
ffmpeg -i data/test/TEST_0000.wav -ar 16000 -ac 2 -b:a 64k fmt_test/test/B_mp3_stereo.mp3
# ... 나머지 포맷도 동일하게 만든 뒤
python src/pipeline.py --test-dir fmt_test/test --sample-submission fmt_test/sample_submission.csv ...
```

## 7. 평가셋이 이 규정을 어떻게 반영하는가

`scripts/build_eval_competition_like.py`가 만드는 `complike_v2` 구성이다.

| 범주 | n | V_PRES | M_PRES | V_FAKE | M_FAKE | 유형 |
| --- | ---: | :---: | :---: | :---: | :---: | --- |
| speech_real | 150 | 1 | 0 | 0 | NA | 음성 |
| speech_fake | 150 | 1 | 0 | 1 | NA | 음성 |
| mix_real_real | 80 | 1 | 1 | 0 | 0 | 혼합 |
| mix_fake_real | 80 | 1 | 1 | 1 | 0 | 혼합 |
| mix_real_fake | 80 | 1 | 1 | 0 | 1 | 혼합 |
| mix_fake_fake | 80 | 1 | 1 | 1 | 1 | 혼합 |
| music_real | 120 | 0 | 1 | NA | 0 | 음악 |
| music_fake | 120 | 0 | 1 | NA | 1 | 음악 |
| song_ai | 200 | 1 | 1 | 1 | 1 | 혼합 |

총 1,060개. voice-present 820 (가짜 510), music-present 760 (가짜 480).

"음악만" 범주에는 보컬 없는 트랙만 들어가야 하므로 GTZAN은 PANNs `voice < 0.2`로
거른 533곡만 쓰고, 가짜 쪽은 MusicGen 생성 instrumental을 쓴다. 생성 클립도
PANNs로 보컬을 검사해 걸리면 버린다.
