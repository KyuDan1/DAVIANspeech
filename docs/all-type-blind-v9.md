# All-type prospective blind v9

`all_type_blind_v9`은 다음 단일 후보를 동결한 뒤 한 번만 확인하기 위한 prospective
평가셋이다. 데이터 선택·의미 검사·렌더·검증·등록 과정에서 authenticity detector,
기존 후보 prediction, EER/ADS/CPS는 실행하거나 열지 않았다. `truth_open_count`와
`score_open_count`는 모두 0이며, 후보 동결 전에는 이 bank를 절대 추론하지 않는다.

## 구성

- 총 120개 독립 base와 5개 paired channel로 600개 파일을 만들었다.
- Voice-only 24개와 Music-only 24개는 각각 real/fake를 12개씩 포함한다.
- Mixed 72개는 RR/RF/FR/FF × concurrent/partial-overlap/sequential 12개 cell에
  6개씩 고정했다.
- 각 base는 clean, G.711 μ-law, G.722 wideband, Opus narrowband 8 kHz,
  G.711→Opus 이중 변환으로 동일하게 렌더했다.
- 따라서 File/Voice/Music authenticity와 Voice/Music presence가 모두 양·음성
  class를 가지며 ADS와 CPS의 모든 항이 정의된다.

## 원본과 라벨 정책

Real voice는 미사용 EchoFake bonafide speaker, fake voice는 미사용 EchoFake
replay attack의 11개 TTS family를 중심으로 구성했다. Real music은 미사용 FMA,
fake instrumental은 FakeMusicCaps의 5개 generator family를 사용했다.

Concurrent FF의 6개 joint 곡은 SONICS의 미사용 Suno/Udio 원본 곡을 분리하지 않고
그대로 사용했다. 이 경우 보컬과 음악이 모두 AI 생성 성분이므로 Voice와 Music을
모두 fake로 라벨했다. Suno/Udio 보컬 곡을 RF 또는 music-only fake로 잘못 넣지
않았다. Partial/sequential FF에는 서로 독립적인 fake voice와 fake instrumental을
사용해 레이아웃을 정확히 통제했다.

## 진위와 독립적인 의미 검사

검사 기준은 첫 source를 보기 전에 고정했으며 오직 성분 존재 여부만 확인한다.
Instrumental은 PANNs any-voice ≤ 0.20이면서 HTDemucs vocal/mixture ≤ -1.5 dB,
joint song은 PANNs any-voice ≥ 0.20이면서 vocal/mixture가 -20~-1.5 dB여야 한다.
HTDemucs는 이 presence 검사에만 사용했고 최종 joint 오디오를 분리·재합성하지 않았다.

실패하면 동일 kind·label·generator 안에서 사전에 고정된 stable-hash 순서의 다음
미사용 source로만 교체했다. 총 42개 실패의 ID, presence 수치, 실패 규칙과 교체
이력을 provenance에 누적했고 최종 96개 music-present source가 통과했다. 이
presence gate에 조건부인 평가셋이라는 한계도 함께 기록한다.

## 누수 및 무결성

- 보호 범위: 빌드 전에 존재하던 85개 truth와 16개 source/audio hash manifest
- 기존 train/development/retrospective/v8 source identity overlap: 0
- target/reference speaker overlap: 0
- canonical song overlap: 0
- 알려진 source 및 rendered audio SHA256 overlap: 0
- source validation: PASS
- bank validation: 14/14 PASS
- reservation SHA256:
  `ccf94ab671d5d21505cea01346c4f7ebd61710ee68492aa94c35ca4fb003b519`
- truth SHA256:
  `b11b0fecb01ac976d0b38842700d1574918012e0b53d84286737ebe183754bff`
- audio manifest SHA256:
  `b15a0264c3a74f5fff63089ea8e0fa984bb36afd63b360992f6f0c055eff86bc`

## 사용 규칙과 재현 경로

이 bank는 `locked_eval`이다. 학습, threshold/ensemble/router 조정, early stopping에
사용할 수 없다. train/development만으로 후보 하나를 완전히 동결한 뒤 단 한 번만
추론·채점하고 즉시 retrospective로 퇴역해야 한다.

- builder: `scripts/build_all_type_blind_v9.py`
- frozen config: `configs/all_type_blind_v9.yaml`
- 최종 예약: `reports/all_type_blind_v9_design/reservation_final/`
- source 검증: `reports/all_type_blind_v9_design/source_validation/`
- bank 검증: `reports/all_type_blind_v9_design/bank_validation/`
- locked truth: `data/eval/all_type_blind_v9/truth.csv`
