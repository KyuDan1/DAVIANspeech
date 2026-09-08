# v72: 짧은 가짜 구간의 정답으로 표현까지 조정하기

## 질문

v71은 시간별 탐지 head를 학습하지만 EAT 표현은 완전히 고정한다.
v71의 첫 예정 평가(2/6 epoch)에서는 file_only의 로컬 development ADS가
0.826643, dense_supervised가 0.818489였다. **중간 결과이며 최종 우열이 아니다.**
전체 발화 위주로 학습한 표현이 짧은 삽입 구간 정보를 충분히 드러내지 못한다면,
head의 시간별 정답만 늘리는 것으로는 부족할 수 있다는 가설을 별도로 검사한다.

## 통제된 변경

v71 dense_supervised와 같은 원천 catalog, strict train/development 분할,
seed·온라인 합성 규칙·epoch별 4,096 draws·6 epochs·head 초기화·손실·집계 온도를
사용한다. 기존 274,564개 EAT adapter 파라미터만 추가로 업데이트한다.
원래 EAT backbone은 계속 고정한다. adapter learning rate는 사전 고정한 1e-4,
head는 v71과 같은 1e-3이다.

첫 학습 파일에서 gradient를 켠 경로와 v71의 고정 경로가 같은 초기 확률을 내는지
검사한다. 학습 중에는 adapter에 실제로 gradient가 흐르는지, 원래 EAT 파라미터에는
흐르지 않는지 매 step 확인한다. 각 epoch의 실제 원천 SHA256과 draw trace를
저장하므로, 완료 후 v71과 입력 순서가 정말 같았는지 비교해야 한다.

v71을 중단하거나 그 설정을 수정하지 않는다. 두 실험은 같은 GPU 0에 함께 올릴
수 있는 별도 작업이다. v72는 추가 학습 메모리가 들기 때문에 CUDA smoke로 먼저
확인한다. 데이터·모델을 읽는 NAS I/O 대기는 GPU 메모리 확대만으로 해결되지 않는다.

## 평가와 주의점

2/4/6 epoch의 기존 development에서 고정된 pooled/domain-macro ADS 기준으로
checkpoint를 선택한다. v70 stress, locked, Suno를 선택이나 학습에 쓰지 않는다.
선택된 checkpoint는 파일별 독립 추론으로 다시 평가해야 한다. 로컬 숫자를 대회
점수로 해석하거나, 초기 개선만으로 자동 제출하지 않는다.

v71과 비교할 때 변경은 표현 adapter의 학습 허용이지만, 이는 추가 학습 파라미터와
연산량을 동반한다. 동일 파라미터 수/연산량 실험이라고 주장하지 않는다.
v71의 최종 선택 epoch와 v72의 최종 선택 epoch가 다를 수 있으므로 같은 epoch의
진행 지표와 사전 규칙으로 선택된 최종 결과를 모두 구분해 제시한다.

## 실행 상태

CUDA smoke의 8 draws 학습/8파일 평가가 통과했다. native 초기 확률은 v71 고정
경로와 정확히 같았고, adapter 274,564개와 head 101,908개가 학습 대상이었다.
GPU peak allocated는 3,032.74MiB였다. 이 값은 성능 개선 증거가 아니다.
8개 파일의 개별 추론도 통과했다(저장 값과 최대 확률 차이 2.82e-8, 역순 추론
bit-exact). 첫 8개 학습 draw의 전체 trace는 v71 전체 학습의 첫 8개와 정확히 같다.
이것을 전체 6 epoch 동일성의 증거로 확대하지 않고, 완료 후 전체 trace도 비교한다.
GPU 0의 전체 학습 작업은 train 18,738 / development 4,577 / raw source 1,767개
분리 검사를 통과했다. 아직 전체 학습 완료 또는 성능 개선 결과는 없다.
코드: `scripts/train_dense_component_adapter_v72.py`
설정: `configs/dense_component_adapter_v72.yaml`
결과 위치: `reports/dense_component_adapter_v72/`

## 전체 학습 완료 및 통제 확인

6 epoch를 완료했고 선언한 개발 선택 기준으로 epoch 6이 선택됐다. 실제 24,576개
draw의 모든 원천·구간·채널·순서 trace는 v71과 bit-exact였다. 그중 합성은
12,306개, 기존 학습 파일 draw는 12,270개다. 사용한 원천은 같은 1,767개다.
adapter의 28개 state tensor가 초기 부모 모델에서 실제로 변경되었음을 확인했다.
통제 검증: `reports/dense_component_adapter_v72/matched_audit/report.json`.

선택 시 개발 ADS는 0.826645로, v71 dense_supervised의 0.826880보다 좋지 않았다.
Voice EER은 0.22659→0.21414, Music EER은 0.13753→0.13424로 개선되었으나
File EER은 0.17309→0.18051로 악화했다. 표현을 추가 학습하면 무조건 좋아지는
것은 아니며, 현재는 전체 모델 교체 근거가 부족하다. 이후 4,577개 전체를 파일별로
독립 재추론해 세 EER/ADS가 저장 평가와 같음을 확인했다.
`reports/dense_component_adapter_v72/full_file_local/`에 저장했다.
음악 출력 단독 활용의 비교는 별도 과제로 남아 있다.
