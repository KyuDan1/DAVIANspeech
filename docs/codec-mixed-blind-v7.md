# Codec mixed blind v7

`codec_mixed_blind_v7`은 최종 후보 한 개만 확인하기 위해 locked 상태로 만든 뒤
한 번 채점하고 즉시 퇴역한 평가셋이다. 구축·설정 동결 시점까지 authenticity
detector 추론, prediction 열람, EER/ADS/CPS 계산은 모두 0회였다.

## 무엇이 달라졌나

- 기존 v5/v6를 포함한 `data/**/truth*.csv` 83개를 역할 설정과 무관하게 모두
  보호했다. 예약 시 각 파일의 경로·행 수·SHA256을 snapshot으로 저장했고,
  source 검증과 렌더 직전에 다시 확인했다.
- 음성은 기존에 화자 풀이 소진된 LibriSpeech/ASVspoof 대신 EchoFake의 새
  source speaker를 사용했다. real/fake 각각 36개이며 target speaker는 72개
  모두 다르다. fake voice는 11개 TTS 계열에 3~4개씩 분산했다.
- fake music은 FakeMusicCaps의 5개 생성기 계열에 7~8개씩 분산했다.
  보컬이 포함된 SONICS의 Suno/Udio는 0개다. 따라서 Suno vocal을
  `voice real + music fake`로 잘못 표기하는 RF 오염이 없다.
- fake music뿐 아니라 real FMA 음악도 모두 동일한 고정 의미 검사를 거쳤다.
  PANNs speech/vocal 최대값 0.20, Demucs vocals-to-mixture -1.5 dB 중 하나라도
  넘으면 같은 라벨·같은 생성기에서 사전 해시 순서의 다음 소스로 교체했다.
  총 33회 교체 후 최종 72개 음악이 모두 통과했다. 이 검사는 음악에 음성이
  존재하는지만 확인하며 생성 진위 점수는 만들지 않는다.

## 균형

72개 base recipe는 아래 12개 cell에 정확히 6개씩 배정된다.

- 구성요소 진위: RR, RF, FR, FF
- 배치: concurrent, partial overlap, sequential

각 base를 `clean`, `g711_ulaw`, `g722_wb`, `opus_nb_8k`,
`transcode_g711_opus`의 다섯 paired view로 렌더해 총 360개 파일을 만든다.
각 채널은 72개씩이며, 채널 사이의 차이는 source 선택이 아니라 동일 base에
적용한 통신 변환뿐이다.

## 누수 및 무결성 결과

- 기존 truth source/speaker identity overlap: 0
- 기존 canonical music-song overlap: 0
- 알려진 source audio SHA256 overlap: 0
- within-bank source, target speaker, reference speaker 재사용: 0
- source validation: PASS
- 완성 bank validation: 16/16 PASS
- reservation SHA256:
  `950c57ece9132dfb77b7a6c7299872dcfb82160db774a711a35b35f65b393874`
- truth SHA256:
  `75fe464b32b4b9f11f14c334e5fba38c871d0c11bf5d73b7b0d0f9bd0c303ac5`

## 재현 경로

- builder: `scripts/build_codec_mixed_blind_v7.py`
- 최종 예약: `reports/codec_mixed_blind_v7_design/reservation_final/`
- source 검증: `reports/codec_mixed_blind_v7_design/source_validation_complete/`
- bank 검증: `reports/codec_mixed_blind_v7_design/bank_validation_complete/`
- truth: `data/eval/codec_mixed_blind_v7/truth.csv`
- 동결 설정: `configs/codec_mixed_blind_v7.yaml`

## 단 한 번의 결과와 퇴역

사전에 동결한 v50과 v52를 함께 추론한 뒤 정답을 한 번만 열었다.

| 모델 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| v50 | 0.21111 | 0.20000 | 0.30556 | **0.76278** |
| v52 | 0.21296 | 0.20000 | 0.30556 | 0.76185 |

결과 디렉터리는 `reports/sota_candidate_v52_blind_v7_one_shot/paired_score/`이며,
scoring fingerprint는
`3336737e6b5f76cd3f4a1cca6298c3b531e03d7ce6906e996689f657b92d59eb`이다.
채점 직후 v7을 `retrospective_diagnostic`으로 옮겼다. 이 결과로 weight, threshold,
router, checkpoint 또는 stopping point를 고르는 것은 금지한다.
