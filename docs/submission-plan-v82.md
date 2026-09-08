# v82: 최고 제출에서 성분별 변경을 분리

**해석 정정:** 현재 Music EER 31.43% 가정은 철회한다. v18→v50 사이 Music 변경이
있어 과거 probe를 연결할 수 없다. 최신 공식 결과/정확한 식별 범위는
`docs/official-v82-v83-results-v84.md`를 우선한다.

## 공식 API 접수 완료 (2026-09-06 UTC)

팀 `DAVIANspeech`, 대회 `236749`로 다음 두 파일을 제출했고, 각각
`{"isSubmitted": true, "detail": "Success"}` 응답을 받았다.

- `best_voice_probe_v82.zip`: 현재 최고 anchor의 Voice EER 진단용.
- `best_eat_music_v83.zip`: 원본 오디오 EAT Music-only 개선 후보, 중복 XLS-R pass 제거.

실제 응답/시각/해시는 `reports/api_submissions_v83/`에 저장했다.
이는 접수 성공이며 채점 성공 또는 점수 향상의 증거는 아니다.
시작 시 quota 3, 두 번째 제출 직전 quota 2였다. 이번 실행에서는 2개만 제출했으며
세 번째 제출은 채점 결과를 보고 결정한다.

**최신 Music 후보는 `best_eat_music_v83.zip`이다.** v82의 출력은 유지하면서
폐기되는 Music residual과 그 전용 XLS-R 추가 pass를 생략했다.
3개 혼합 파일에서 5개 확률 전체가 문자열까지 동일했고 ZIP CRC도 통과했다.
`docs/music-deadpass-v83.md` 참조. 두 Music ZIP을 각각 제출하지 않는다.

공식 기준은 `v50_v57m_challenger_v2.zip`이다.
총점 0.7774307884, ADS 0.7538888889, CPS 0.9893078836.
고정 archive SHA-256:
`a5730fde4e0c49100ee9aca5cc373eb550bf02948517072f23afe02fd62b91d6`.

사용자가 최대 3회 제출을 허용했다. 이 문서는 계획이며 API 성공 기록이 아니다.

## 1. 현재 최고 모델의 Voice 진단

`best_voice_probe_v82.zip`: 원래 ZIP의 모든 member를 복사하고 `script.py`에만
마지막 CSV 처리 단계를 추가한다. `VOICE_FAKE_PROB=0.5`, 나머지 4개 출력과 ID는
문자열까지 보존한다. 모델 추론과 파일 확률 계산은 전부 원래대로 수행한다.
점수 향상 목적이 아니므로 진단 제출 점수가 낮아지는 것은 예상되는 결과다.

동일 평가 집합/파이프라인 조건에서:

`Voice EER = 0.5 - (0.7538888889 - probe ADS) / 0.2`.

과거 v18 Music probe는 현재 Music 절대 EER을 알려주지 않는다.
v18→v50에서 Music을 바꾸는 추가 변경이 있어 이전에 적었던 연결 가정은 성립하지 않는다.
현재 Music 상수 probe를 별도로 측정해야
`File EER = (1 - ADS - 0.2*Voice EER - 0.3*Music EER) / 0.5`도 식별할 수 있다.
오래된 Voice/Music probe를 현재 성분 EER로 취급하지 않는다.

## 2. Music-only 후보

`best_eat_music_v82.zip`: 동일 최고 anchor를 먼저 실행한 후 새 EAT-large adapter의
원본 오디오 Music 확률만 적용한다. File/Voice/CPS를 함께 바꾸지 않는다.
학습은 기존 TRAIN에서 완료된 checkpoint를 사용하며 평가 데이터 추가 학습은 없다.
이 후보가 공식 ADS를 개선하면 해당 차이는 Music EER 변화로 직접 해석할 수 있다.

추가 모델은 약 1.24 GB이며 별도 namespace로 묶어 기존 패키지 코드를 덮어쓰지 않는다.
80개 개발 오디오의 기존 native 추론 결과 재현과 나머지 열 보존을 검사한다.
전체 파이프라인 smoke와 ZIP CRC/크기 검증은 별도로 필요하다.
첫 독립 smoke에서 EAT의 timm shim이 transformers보다 먼저 초기화되는 import-order
문제를 발견했다. 새 Music wrapper가 transformers dependency discovery를 먼저
실행하도록 수정하고 별도 staged_v2/smoke_v2에서 재검증한다.
기존 checkpoint나 다른 실험의 frozen 소스는 수정하지 않았다.

독립 staged_v2 검사 완료: 개발 80개, Music 확률 최대 차이 `1.11e-16`,
다른 4개 확률과 ID 문자열 동일. 새 환경에서 encoder를 로드한 시간을 포함해
6.77초, peak CUDA 1,302 MiB였다. 전체 anchor 또는 L4 시간 측정은 아니다.

### 최고 anchor를 다시 실행한 600개 codec-mixed 개발 비교

| 조건 | 기존 Music EER | Music-only v82 |
|---|---:|---:|
| 전체 | 25.33% | 21.33% |
| clean | 11.67% | 6.67% |
| G.711 | 20.00% | 11.67% |
| G.722 | 8.33% | 8.33% |
| Opus narrowband | 38.33% | 28.33% |
| G.711 → Opus | 40.00% | 33.33% |
| 실제 음성과 혼합 | 25.33% | 22.00% |
| 가짜 음성과 혼합 | 26.00% | 20.67% |
| 동시 혼합 | 23.00% | 18.00% |
| 부분 overlap | 32.00% | 26.00% |
| 순차 혼합 | 20.00% | 20.00% |

File/Voice EER은 교체하지 않았으므로 각각 23.33%, 20.33%로 동일하다.
ADS만 0.766667 → 0.778667. 이 600개는 모두 두 성분이 있어 CPS/총점은
정의할 수 없다. 공식 점수 예상 또는 새로운 blind 결과가 아니다.
기존 cache와 재실행 anchor 사이의 확률 차이가 있었으므로, 이번에는 hash 검증된
**재실행 anchor**를 사용했다. 결과: `reports/music_only_submission_v82/codec_audit/`.

## 3. 남은 슬롯

앞의 결과에 따라 조합 후보 또는 File 개선 진단을 고른다. 검증이 끝나지 않은 후보로
슬롯을 소진하지 않는다. v80 local Voice는 긴 부분 가짜 파일에서 크게 개선되지만
일반 개발셋에서는 악화되므로 지금 전면 교체하지 않는다.

사용자가 팀 이름 `DAVIANspeech`를 확인해 주었고 API validate에서 quota 3을 확인했다.
현재 Voice 진단 → Music v83 순서로 제출을 진행한다. 세 번째 슬롯은 결과를 본 뒤 결정한다.
실제 접수 여부는 `reports/api_submissions_v83/*.json`의 `result.isSubmitted`와
`status=accepted`로 확인한다. `attempt_started`는 접수 성공이 아니다.
토큰은 대화에서 제공된 값을 비표시 입력으로 전달하며 코드·문서·명령줄·결과 파일에
저장하지 않는다. 로컬 SDK import에서 누락된 aiohttp를 발견해 제공 wheel의 의존성을
설치한 뒤 다시 시작했다. 이 import 오류는 API 호출 전 발생했으며 제출 횟수를 쓰지 않았다.

## 완성된 ZIP 무결성 기록

- `best_voice_probe_v82.zip`: 110개 member, 압축 해제 9,185,965,963 bytes.
  SHA-256 `e9e35a18e720b72bf26f33f78e055bc5341a3c2f64762b525aecd0cf02fd58a9`.
- `best_eat_music_v82.zip`: 130개 member, ZIP 9,012,561,325 bytes,
  압축 해제 10,425,245,822 bytes.
  SHA-256 `1267660ca5b4dc5168988bec6df4114ebcc81c04bd3c9f200fad38fdf5202082`.

두 ZIP 모두 전체 member CRC, 크기 제한, 4GB 이상 단일 member 없음,
최상위 구조, 중복/unsafe/symlink 없음, 기존 비-entrypoint member의 CRC/크기 보존을
검사했다. 무결성 통과는 전체 추론 성공이나 공식 점수 개선을 뜻하지 않는다.
Music 전체 패키지 smoke도 완료했다. ZIP을 실제로 압축 해제하고 동일한 개발 혼합
오디오 3개(RR, FR, RF)를 기존 entrypoint와 새 entrypoint에 각각 넣었다.
두 번 모두 `output/submission.csv` 생성, ID/확률 범위 검사 통과.
Music 이외 4개 확률의 실행 간 최대 차이는 **0.0**이었다.
합계 실행 시간은 154.79초(B200, 두 파이프라인 실행 및 모델 로드 포함)다.
이는 L4에서 1,200개를 60분 내 처리했다는 증거가 아니다.
결과: `reports/music_only_submission_v82/full_package_smoke/report.json`.

신규 환경에 로컬 검증용 `configs/requirements-grader-audio-v82.txt`를 추가 설치했고
`pip check`도 통과했다. torch/torchaudio 2.7.1+cu128, numpy 1.26.4가 유지됐다.
제출 ZIP의 requirements는 기존과 동일한 `onnxruntime-gpu==1.23.2` 한 줄이다.
로컬 전체 환경 파일을 제출 requirements로 사용하지 않는다.
