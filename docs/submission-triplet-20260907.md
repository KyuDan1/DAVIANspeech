# 2026-09-07 성능 후보 3개

사용자 요청에 따라 현재 유망한 세 후보를 제출한다. 공통 기준은 공식 최고
v83(총점 0.7774307884)이며, 세 후보 모두 joint prompt로 File/Voice를 보완한다.
세 후보 간 모델 파일 차이는 Music checkpoint 하나다. 각 후보는 실제 ZIP을
새 디렉터리에 풀어 평가 서버의 핵심 패키지 버전으로 전체 실행을 검증한다.

| 제출 파일 | Music expert | Music weight | broad Music EER | codec mixed EER |
|---|---|---:|---:|---:|
| `v101_prompt_codec.zip` | seed35 exact-codec | 0.28 | 0.1202 | 0.2033 |
| `v108_generator_codec.zip` | ArtifactBench + 낮은 학습률 codec 적응 | 0.37 | 0.1164 | 0.2100 |
| `v109_generator_broad.zip` | ArtifactBench broad, 추가 codec 적응 없음 | 0.50 | 0.1131 | 0.2433 |
| 기존 v83 | EAT Music | — | 0.1291 | 0.2133 |

v108을 우선 제출하고, v101로 전화/코덱 성능을, v109로 최신 생성기 대응의 이득을
확인한다. v109는 pooled EER와 최신 생성기 검출이 좋지만 codec mixed가 악화되어
가장 공격적인 후보이다. 세 제출 모두 성능 후보이며 상수 확률 진단 제출은 없다.

공통 설정은 joint File 0.40, Voice 0.12, joint 3-view, Music 1-view,
최종 File과 component OR 결합 0.15이다. Music이 달라지면 이 OR를 통해 최종 File도
달라진다. 따라서 세 후보를 “Music 출력만 다른 ablation”이라고 해석하면 안 된다.
Voice와 Presence 두 출력의 수치 보존은 동일 입력 smoke 5개에서 세 후보 모두
CSV 문자열까지 동일함을 확인했다. 이는 해당 시험 입력에서의 확인이다.

v101은 고정 ZIP 그대로 batch 12, v108/v109는 batch 48이다. batch 변경은
bfloat16 반올림 차이를 유발할 수 있다. 이전 broad Music batch 감사에서 EER는
동일했으나, batch 속도/메모리 측정은 B200 결과이며 L4 1,200파일 실행시간을
직접 증명하지 않는다.

ArtifactBench fake-only 결과는 expert score > 0.5 비율이지 EER나 최종 pipeline
정확도가 아니다. 또한 이 OOD 자료의 관찰이 후속 codec 재적응 방향에 영향을
주었으므로, 반복 비교 결과를 독립적인 최종 holdout 성능이라고 부르지 않는다.
학습에 넣지는 않았지만 향후 model/weight 선택 후의 새 독립 검증을 대체할 수 없다.

ZIP은 각각 약 9.07GB, 압축 해제 약 10.48GB다. 실제 제출은 30자 이하의 별도
파일명으로 한다. `plan.json`의 SHA와 전체 실행 report를 확인한 파일만 SDK로
전송하며, 매 시도는 별도의 ledger에 기록한다. 접수 성공은 채점 성공 및 목표 점수
달성과 구분한다.

- 제출 계획: `reports/submission_triplet_20260907/plan.json`
- 실제 ZIP 실행 검증: `reports/submission_triplet_20260907/v101_smoke/`, `v108_smoke/`, `v109_smoke/`
- API 접수 기록: `reports/api_candidates_20260907/`
- 제출 도구: `scripts/submit_verified_candidates_v109.py`

공식 0.85 이상 달성 여부는 세 제출의 채점 결과로 확인해야 하며 로컬 EER만으로
보장할 수 없다.

## 제출 전 실행 검사 완료

세 ZIP 모두 전체 CRC/SHA 확인 후 실제 압축 해제한 `script.py`를 실행해 통과했다.
시험 입력은 WAV/MP3/FLAC 5개이며 60초 스테레오를 포함한다. 인터넷 socket
연결을 차단했고 입력 파일이 수정되지 않았으며 출력 5열의 확률 범위를 확인했다.
평가 서버 핵심 버전(torch/torchaudio 2.7.1+cu128, numpy 1.26.4, pandas 2.0.3)
환경에서 검사했다. ffmpeg는 서버 기본 설치 항목이며 로컬 PATH를 명시했다.

| 후보 | 5개 입력 추론 시간 | 최대 CUDA allocated | 최대 RSS |
|---|---:|---:|---:|
| v108 | 114.21초 | 12.09GiB | 13.88GiB |
| v101 | 115.42초 | 12.09GiB | 13.87GiB |
| v109 | 78.21초 | 12.09GiB | 13.86GiB |

B200 GPU와 CPU affinity 6개로 측정했다. L4 전체 1,200파일 60분 제한 통과를
보장하는 벤치마크는 아니다. 신규 prompt expert는 원본 오디오를 사용하지만,
공유 기준 파이프라인에는 분리 단계가 남아 있으므로 전체를 separation-free라고
표현하지 않는다.

## 실제 API 제출 결과

2026-09-07 아래 세 파일을 대회 `236749`, 팀 `DAVIANspeech`로 순차 제출했다.
모두 SDK가 `{"isSubmitted": true, "detail": "Success"}`를 반환했다.

| 순서 | 파일 | 접수 완료 UTC | 제출 직전 잔여 횟수 |
|---|---|---|---:|
| 1 | `v108_generator_codec.zip` | 11:59:57 | 3 |
| 2 | `v101_prompt_codec.zip` | 12:01:34 | 2 |
| 3 | `v109_generator_broad.zip` | 12:03:10 | 1 |

오늘 요청한 3회 제출을 완료했으므로 추가 실험/제출을 멈춘다. 이 기록은 **접수
성공**이며, 평가 서버의 추론 성공·공식 점수·순위는 아직 확인하지 못했다.
각 파일의 정확한 SHA-256, 접수 응답 및 타임스탬프는
`reports/api_candidates_20260907/<파일명>.json`에 저장되어 있다.
