# v81: 실제 보컬 원본 확보 — 아직 TRAIN 편입 아님

## 필요한 이유와 범위

현재 생성 보컬의 존재/진위 약점은 발화 중심 학습과 보컬 주석 부족의 영향일 수 있다.
이를 encoder 크기 문제로만 간주하지 않는다. 실제 보컬의 음색·발성 범위를 넓히기 위해
VocalSet 원본을 확보한다. **이 자료는 실제 보컬이며, 가짜 보컬 학습 예제를 제공하지 않는다.**
합성 보컬과의 EER을 이 자료만으로 계산할 수도 없다.

[공식 VocalSet 기록](https://zenodo.org/records/1442513)은 전문 가수 20명의 단선율 보컬
약 10.1시간을 설명한다. 2026-09-05 [공식 API](https://zenodo.org/api/records/1442513)에서
`cc-by-4.0`을 확인했다. 원래 저자 Wilkins, Seetharaman, Wahl, Pardo와 DOI를 보존한다.

## 원본과 기존 제3자 사본을 구분

공식 `VocalSet11.zip`은 2,077,243,579 bytes,
MD5 `0c09396242f946e7111ad7d8fc649b81`이다. 압축 파일 다운로드를 시작했으며,
`data/external/vocalset_original_v81/report.json`의 완료 상태, 공식 hash와 전체 ZIP CRC를
검증하기 전에는 확보 완료라고 간주하지 않는다. 원본 음원을 아직 추출/학습하지 않는다.

기존 `vocalset_presence_subset_v1/`에는 제3자가 편집했다고 설명하는 60파일이 있다.
그 README의 MIT 표기를 원본 라이선스로 대신하지 않으며, 출처 설명만으로 원본과
동일한 오디오라고 확정하지 않는다. 다음을 순서대로 확인한다.

1. 등록된 모든 TRAIN/개발/보호 역할과 최근 실제 학습 inventory에 기존 사본 참조가 있는지 조사.
2. 공식 원본과 기존 사본의 녹음/가수 및 waveform 관계 확인. 파일명 일치만으로 중복을 판정하지 않음.
3. 원곡·녹음·가수 기준 분할을 먼저 보호하고, 그 후 TRAIN 편입. 기존 보호 eval은 계속 제외.
4. 실제 보컬의 존재 학습과 실제 보컬 오탐 감소를 분리 평가. 가짜 보컬 일반화는 다른 적법한 양성 원천 필요.

`scripts/audit_vocalset_usage_v81.py`의 문자열/참조 조사 결과만으로 decoded-PCM 중복 없음이나
모든 과거 비등록 실험에서 미사용이었다는 결론을 내리지 않는다.

## CtrSVDD를 즉시 넣지 않은 이유

[공식 CtrSVDD 안내](https://github.com/SVDDChallenge/CtrSVDD_Utils)는 일부 실제 원천 음원이
배포 ZIP에 없고 별도 확보해야 한다고 명시한다. 전체 메타데이터의 개수는 확보된 음원의
개수와 다르다. 가짜 음원만 받아 원천/진위가 서로 얽힌 분류기를 학습하지 않는다.

CtrSVDD 전체 조건은 CC BY-NC-ND 4.0이고, 원천별 추가 조건도 확인해야 한다.
예를 들어 [M4Singer의 이용 조건](https://github.com/M4Singer/M4Singer/blob/master/dataset_license.md)은
CC BY-NC-SA 4.0 외에 책임·기관 권한에 관한 진술을 포함한다. 이를 대신 확약하거나,
가중치 제출까지 자동 허용된다고 단정하지 않는다. 이번 작업에서는 CtrSVDD 오디오를
내려받거나 학습에 넣지 않았다. 이것이 모든 CtrSVDD 연구 이용이 금지된다는 뜻은 아니다.

## 실행과 상태

다운로더: `scripts/fetch_vocalset_v81.py`.
공식 record/license/size/hash 변경 시 중단하고, 중단된 `.part`는 남겨 `--resume`으로
정확한 HTTP Range를 확인한 뒤 이어 받는다. 기존 파일 덮어쓰기, 음원 추출, 학습 등록,
모델 학습, API 제출은 하지 않는다. metadata/안전한 ZIP 경로 검사 테스트 2개가 통과했다.

학습 중인 v80의 데이터 분할 및 고정 source/config는 변경하지 않는다.

참조 조사 완료: 등록 역할의 manifest 및 최근 실제 TRAIN/개발 inventory를 합쳐
59개 항목을 검사했으며, `vocalset`/`squillo`/기존 60파일 stem 문자열 참조는 0건이었다.
60개 사본의 SHA256 및 README상 가수 20명 목록도 기록했다. 이는 문자열 참조 조사이며
원본/가공본의 waveform 중복이나 과거 비등록 사용까지 부정하는 증거는 아니다.
결과: `reports/vocalset_usage_v81/report.json`.
