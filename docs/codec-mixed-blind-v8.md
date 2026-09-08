# Codec mixed blind v8

`codec_mixed_blind_v8`은 frozen v50과 frozen v54를 한 번 비교한 뒤 즉시 퇴역한
prospective 평가셋이다. 데이터 예약, 의미 검사, 렌더, 잠금까지 authenticity
detector 추론, prediction 열람, EER/ADS/CPS 계산은 모두 0회였다. 두 패키지의
추론이 모두 끝나고 CSV 구조·범위·ID를 검증한 다음 truth를 단 한 번 열었다.

## 일회성 결과와 퇴역

scorer의 기존 별칭 `v47`과 `v48`은 각각 **frozen v50**과 **frozen v54**를 뜻한다.
v54는 Voice EER를 `0.34444→0.31667`로 개선했지만 File EER가
`0.25741→0.27593`으로 더 크게 악화되었다. Music은 의도대로 완전히 같았다.

| 후보 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| frozen v50 | **0.25741** | 0.34444 | 0.24444 | **0.72907** |
| frozen v54 | 0.27593 | **0.31667** | 0.24444 | 0.72537 |

따라서 v54는 전체 후보로 기각한다. 이 결과를 이용해 residual weight, File 결합식,
threshold를 다시 맞추지 않는다. v8은 점수 직후 `locked_eval`에서
`retrospective_diagnostic`으로 옮겼으며 이후 모델 선택·튜닝·stopping에 사용할 수
없다. 상세 산출물은 `reports/v54_blind_v8_one_shot/paired_score/`에 있다.

## 설계

- 72개 base recipe를 RR/RF/FR/FF와 concurrent/partial-overlap/sequential의
  12개 cell에 6개씩 고정했다.
- 동일 base를 clean, G.711, G.722, Opus 8 kHz, G.711→Opus 이중 변환으로
  렌더해 360개 paired sample을 만들었다.
- instrumental FR의 fake voice는 EchoFake replay attack이며 11개 TTS 계열과
  서로 다른 target/reference speaker를 사용한다.
- FF는 SONICS의 Suno/Udio joint AI song을 HTDemucs로 동기화된 vocal과
  accompaniment stem으로 나눠 사용한다. 이 보컬은 항상 fake voice로 표기하며
  RF에 Suno/Udio 보컬 곡이 들어가는 경우는 없다.
- instrumental RF는 보컬이 없는 FakeMusicCaps의 5개 생성기 계열을 사용한다.

## 의미 검사와 selection-bias 통제

검사는 진위를 판단하지 않고 component label이 맞는지만 확인한다. 음악 stem은
PANNs any-voice ≤ 0.20과 Demucs vocal/mixture ≤ -1.5 dB를 모두 만족해야 한다.
SONICS vocal stem은 별도로 PANNs any-voice ≥ 0.20이어야 한다. 임계값과 교체
순서는 첫 검사 전에 고정했다.

실패 시 같은 label·같은 generator에서 stable-hash 순서의 다음 source만 골랐다.
이전에 실패했거나 이미 검사한 source는 다시 후보가 될 수 없다. 7회의 full
screen에서 기록된 38개 실패의 source ID, 원인 수치, 교체 이력을 provenance에
누적했으며 최종 72개가 모두 통과했다. 따라서 최종 분포가 이 고정 의미 gate에
조건부라는 한계도 provenance에 명시한다.

## 누수 및 무결성

- 기존 truth/source/speaker identity overlap: 0
- 기존 canonical song overlap: 0
- 알려진 source audio SHA256 overlap: 0
- source validation: PASS
- 완성 bank validation: 18/18 PASS
- reservation SHA256:
  `6532bf244444d868fef351c1b24d35c70152a1ba0460d0040ea496cafb82301b`
- truth SHA256:
  `7c299e957541c8bb7431f256e43c2ae9576a968366d62dcd0519e3141abcb342`

## 재현 경로

- builder: `scripts/build_codec_mixed_blind_v8.py`
- 최종 예약: `reports/codec_mixed_blind_v8_design/reservation_final/`
- source 검증: `reports/codec_mixed_blind_v8_design/source_validation/`
- bank 검증: `reports/codec_mixed_blind_v8_design/bank_validation/`
- truth: `data/eval/codec_mixed_blind_v8/truth.csv` (일회 평가 후 회고 전용)
- 동결 설정: `configs/codec_mixed_blind_v8.yaml`
