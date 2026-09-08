# v61: 긴 파일의 짧은 가짜 음성 구간 검증

**설계 보완(탐지 점수 확인 전):** 고정 앞/중간/뒤 위치는 기존 5-window가 모두 보는
위치였다. 아래 원래 2,160행은 covered-position 대조 조건으로 유지한다. 별도의
uniform-position 720행을 생성하며, 새 긴 파일 비교의 주 평가는 이 uniform bank이다.
둘을 합쳐 임의 비중의 하나의 EER로 판단하지 않는다.

상태: 원천 100개, 고정 위치 대조군 2,160행, uniform 720행의 생성·무결성 검증 완료.
모두 보호된 평가 역할에 등록했다. 탐지 성능 평가는 아직 수행하지 않았다.
현재 v60 학습·checkpoint 선택은 변경하지 않는다.

## 왜 별도로 만드는가

v60 개발 4,577행은 최장 20.2초이며 30초 이상이 없다. 따라서 4초 창을 5개만 보는
모델과 전체 시간을 보는 모델의 차이를 긴 파일에서 검증하지 못한다. 이 bank는
**30/45/60초의 진짜 배경 음성 속 2초의 가짜 음성**을 찾는지 검사한다.
음성만 있는 시간적 혼합 검사이며, RF/FR 음악 혼합이나 자연스러운 실제 통화까지
대표하는 평가로 해석하지 않는다. 전체 목표를 이 검사 통과로 대체하지 않는다.

## 대조군과 한계

- 그룹마다 서로 다른 원천의 진짜 배경 음성, 진짜 삽입 음성, 가짜 삽입 음성,
  재녹음 진짜 삽입 음성, 재녹음 가짜 삽입 음성을 예약한다.
- 재녹음된 진짜 음성은 REAL이다. 재녹음 여부를 authenticity와 교차하여, 녹음 품질
  자체가 Fake의 단서가 되는 설계를 피한다. 직접/재녹음은 같은 내용의 짝은 아니며
  각각 다른 화자·원천을 사용한다.
- 진짜와 가짜 모두 같은 2초 길이, 앞/중간/뒤 위치, RMS/peak gain 정책과 10 ms
  crossfade로 삽입한다. 클래스 label을 renderer에 넘겨 편집을 바꾸지 않는다.
- 삽입 crop은 authenticity 점수가 아닌 에너지로 고른다. 2초보다 짧은 원천은
  payload header에서 제외한다. silent/nonfinite 원천은 추가 검증에서 실패시킨다.
- 배경을 연결·반복해서 길이를 맞추므로 **자연적인 60초 대화가 아니다**. 모든
  클래스에 동일한 반복 방식을 적용하지만, 반복이라는 도메인 차이는 남는다.
- 최종 파일 전체에 clean/G.711/Opus 조건을 적용한다. 창마다 codec 상태를 초기화하는
  v60 학습 augmentation과 달리 실제 연속 파일 codec roundtrip을 검사한다.
- fake 시간 구간 annotation은 codec 적용 전의 삽입 구간이다. Codec 지연·잔향까지
  sample-exact한 생성 기원을 뜻하지 않는다.

20개 독립 source group × 3길이 × 3위치 × 2녹음조건 × 2정답 × 3채널 = 2,160행을
계획한다. 2,160행을 독립적인 원천 2,160개로 세지 않는다. 불확실성을 추정할 때는
변형 파일별이 아닌 20개 source group을 묶어 재표본화해야 한다.

## 누수 방지와 출처

`scripts/reserve_long_voice_v61.py`는 모든 이전 `data/**/truth*.csv`와 data/reports의
reservation.csv를 보호한다. source/content/speaker/reference speaker를 제외하며,
한 bank 안에서도 역할 간 동일 identity를 재사용하지 않는다. 현재 v60 학습 대상으로
추가하지 않으며, 추후에도 학습·threshold/weight/router 선택용으로 사용하지 않는다.

[EchoFake 공식 데이터 카드](https://huggingface.co/datasets/EchoFake/EchoFake)는 MIT로
표시되어 있다(2026-09-05 확인). 로컬 open_set_eval 3개 parquet에는 direct/replay 각각
real 6,400개와 fake 6,400개가 있으며, fake의 생성기는 CosyVoice2, FireRedTTS, IndexTTS,
MaskGCT, OpenVoice-V2의 5종이다. 이 결과를 Qwen3-TTS/F5-TTS 등 모든 생성기에 대한
일반화로 주장하지 않는다. 원천 파일의 권리·출처는 최종 보고서에 계속 명시해야 한다.

공식 저장소에 [과거 중복 데이터 신고와 수정 답변](https://huggingface.co/datasets/EchoFake/EchoFake/discussions/3)이
있으므로 ID만 신뢰하지 않는다. Materialization은 source bytes hash, decoded PCM hash,
저장된 오디오 hash를 기록하고 동일 decoded PCM을 bank 내에서 허용하지 않는다.
이는 미지의 모든 재인코딩·근접 중복까지 배제한다는 보증은 아니다.

원천/예약 metadata만 선정한 초기 디렉터리 `long_voice_v61`은 hash 함수 인자 오류로
미완료였다. `long_voice_v61_v2`는 metadata 예약까지 완료했지만 2초 길이 조건을
header에서 검증하지 않아 렌더링하지 않았다. 두 예약은 삭제하거나 재사용하지 않고,
그 원천까지 보호한 `long_voice_v61_v3`에서 길이 검증을 포함해 다시 예약한다.
어느 단계에서도 authenticity detector 출력으로 원천을 고르지 않았다.

v3 예약은 기존 truth/예약 126개와 hash manifest 18개를 보호했다. 최종 source 100개의
길이는 2.133~12.353초이며, direct/replay fake 각각 20개에서 5개 생성기가 4개씩이다.
Materialization은 100개 모두 완료했고 bank 내 decoded PCM 중복은 0이었다.
`configs/data_partitions.yaml`의 `locked_eval`에 **원천 예약 manifest**를 추가하여
향후 학습에서도 보호한다. 이 추가는 이미 실행 중인 v60의 train/dev 내용이나
checkpoint 선택 기준을 바꾸지 않는다. 완성된 파생 파일의 truth 등록은 무결성
검증 이후에 별도로 한다.

## 재현과 사용 조건

```bash
python scripts/reserve_long_voice_v61.py \
  --output data/reservations/long_voice_v61_v3
python scripts/build_long_voice_v61.py materialize \
  --reservation data/reservations/long_voice_v61_v3 \
  --output data/sources/long_voice_v61_v3
python scripts/build_long_voice_v61.py render \
  --reservation data/reservations/long_voice_v61_v3 \
  --sources data/sources/long_voice_v61_v3 \
  --output data/eval/long_voice_sparse_v61
python scripts/validate_long_voice_v61.py \
  --reservation data/reservations/long_voice_v61_v3 \
  --bank data/eval/long_voice_sparse_v61 \
  --output reports/long_voice_v61/construction_validation.json
```

이미 있는 출력은 덮어쓰지 않는다. `*.partial`은 완료 artifact가 아니다. 완성 후에도
header/count/hash/라벨·구간/역할 분리 검증 및 partition 등록 전에는 score하지 않는다.
후보는 v60의 기존 development 관문을 먼저 통과한 하나로 고정하며, 같은 bank에서
중간 checkpoint들을 반복 평가하지 않는다. 기존 최고 제출과 File EER을 비교하고
길이·위치·녹음·채널별 결과를 별도로 보고한다. 제출 승격을 위해 pooled File EER
최소 1%p 개선, 길이/채널별 최대 악화 2.5%p 이하를 요구한다. 이 voice-only 관문이
통과해도 음악/혼합 prospective 및 L4 전체 runtime 관문을 대신하지 않는다.

## 구현 검증 기록

- `tests/test_long_voice_v61.py`: **17 passed**. 재녹음 REAL 라벨, reference speaker
  보호, 역할 간 identity 분리, 30/45/60초 × 위치별 편집, 가짜 구간 annotation 및
  factorial 누락/라벨 손상 거부를 검사했다. 이 tests는 실제 전체 bank 검증을 대신하지 않는다.
- 원천 예약을 보호 역할에 등록한 뒤 `assert_development_eval_separation`이 통과했다.
  모든 기존 train/router_train manifest와 원천 예약의 `identity_tokens` 교집합은 0이었다.
- 렌더러는 첫 4개 source group 432행까지 생성한 상태를 확인했다. 완료된 `truth.csv`와
  위 construction validation이 나오기 전까지 전체 구축 완료로 표기하지 않는다.
- v60 두 학습 프로세스는 epoch 4에 진행 중이었다. 새 bank의 authenticity 추론은
  수행하지 않았고, 새 ZIP 생성/API 제출도 하지 않았다.

## 시간 위치 검증으로 발견한 설계 오류와 보완

`scripts/audit_sparse_interval_coverage_v61.py`로 fake 삽입 구간과 기존 WPT의
4.0375초 창 3개/5개의 교집합을 계산한다. 이는 waveform 시간 좌표만 이용하며
탐지 모델이나 authenticity 점수를 읽지 않는다. 원래 early=1초,
middle=(길이-2)/2초, late=길이-3초는 모든 길이에서 기존 창과 겹친다.
따라서 원래 bank만으로는 창 사이 구간을 놓치는 가설을 검사할 수 없었다.

원천 선택을 바꾸지 않고, 각 source group·길이에 고정 SHA-256 난수로
`[0.25초, 파일 길이-2.25초]`에서 삽입 시작점을 뽑는 **uniform 조건**을 추가한다.
같은 group/길이에서는 real/fake·direct/replay·codec 모두 동일한 시작점을 쓴다.
시간 위치를 모델 점수에 맞추거나 5-window의 빈 곳만 골라 배치하지 않는다.
이 조건은 20그룹 × 3길이 × 2녹음 × 2라벨 × 3채널 = 720행이다.
원래 2,160행과 같은 원천이므로 독립적인 두 평가셋이라고 주장하지 않는다.

주 비교는 uniform 720행에 File EER 1%p 개선 및 길이/채널별 악화 2.5%p 이하 조건을
적용한다. 고정 위치 2,160행은 별도 대조 결과로 보고하고 pooled/길이/채널 EER의
악화가 2.5%p를 넘으면 승격하지 않는다. 개발 관문을 통과한 고정 후보 하나만 비교한다.
기준 변경은 bank의 authenticity 추론 **이전**, 순수 시간 기하 감사에서 이뤄졌다.

```bash
python scripts/build_long_voice_v61.py render --uniform-position \
  --reservation data/reservations/long_voice_v61_v3 \
  --sources data/sources/long_voice_v61_v3 \
  --output data/eval/long_voice_uniform_v61
python scripts/validate_long_voice_v61.py \
  --reservation data/reservations/long_voice_v61_v3 \
  --bank data/eval/long_voice_uniform_v61 \
  --output reports/long_voice_v61/uniform_construction_validation.json
```

원래 2,160개의 최종 metadata 저장에서 `dict(**plan, stage=...)`의 중복 keyword 오류가
발생했다. `publish-render` 복구는 모든 생성 파일의 해시·recipe를 확인한 뒤 metadata만
저장했다. 오디오는 재생성하지 않았고 `publication_only_repair=true`를 기록했다.
이 오류를 재현하는 metadata 직렬화 회귀 테스트를 추가했다.

고정 위치 bank는 `reports/long_voice_v61/construction_validation.json`에서 2,160개
전 파일의 디코딩/길이/해시/라벨·구간 및 알려진 보호 원천 해시 중복 0 검사를 통과했다.
그 truth도 `locked_eval`에 등록했다. 최대 peak 1.0인 codec 출력이 포함되며,
이 검증은 무클리핑·무잡음이나 자연 통화 품질을 보증하는 검사가 아니다.
고정 위치의 180개 group×길이×위치 조건은 3-window/5-window 모두 삽입 구간을
100% 본다는 시간 기하 결과도 저장했다. v60 완료 판정/long-bank 테스트 합계
25개가 통과했다. 이는 탐지 성능 검증을 대신하지 않는다.

Uniform bank 720개도 `uniform_construction_validation.json`에서 전수 검증을 통과했다.
5-window가 2초 삽입 구간을 전혀 보지 못하는 조건은 30초 1/20, 45초 4/20,
60초 13/20이다. `uniform_positions_coverage.json`에 3-window 결과와 부분 coverage도
기록했다. 이 비율은 bank의 시간 배치 결과이며 실제 제출 test 비율/탐지 실패율이 아니다.

고정 위치 bank의 PCM 포화 sample 비율은 clean/G.711에서는 real/fake 모두 0,
Opus에서는 real 300/259,200,000, fake 144/259,200,000이었다. 드문 codec 포화가
있음을 기록하며, 이를 보고 파일을 제거하거나 waveform을 재생성하지 않았다.

최종 관련 테스트 **46 passed**, `git diff --check` 통과. 두 파생 bank 등록 후에도
development/locked 역할 분리가 유효하며, 새 locked 원천·파생 파일과 모든 기존
train/router_train의 identity 교집합은 0이었다. 이 근거를 확인한 후 bank provenance를
`locked_unscored_integrity_verified`, `full_bank_complete=true`로 갱신했다.
여기서 complete는 데이터 구축만을 의미하며, 모델 성능·대회 목표 달성을 의미하지 않는다.
