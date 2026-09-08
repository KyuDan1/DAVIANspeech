# v76: 不확실한 음성 부재 감독 신호의 짝지은 대조 실험

## 사전 고정 범위

이 실험은 학습 자료의 레이블 민감도 조사다. 보컬 여부가 확인되지 않은 음악을
자동으로 보컬로 재분류하거나 진위를 변경하지 않는다. 기존 v73 후보와 진행 중인
v74 보호 평가의 코드·가중치·판정 규칙은 수정하지 않는다.

TRAIN 18,738개와 DEVELOPMENT 4,577개는 기존 엄격한 분할을 그대로 사용한다.
보호 평가 자료는 ID/원천 분리 검사 이외에 읽지 않는다. Suno 평가 13개를 학습,
early stopping, 가중치 결정에 사용하지 않는다.

## 유일한 개입

원본 EAT와 PANNs가 모두 음성 존재 확률 0.5 이상을 출력한 **실제 TRAIN 음성 부재
44행**만 제외 대상으로 지정한다. 이미 학습된 head의 출력은 선정 기준에 넣지 않는다.
다른 채널 변형이나 비슷한 출처로 제외 범위를 확대하지 않는다.

- control: 기존 음성 존재 레이블 그대로 사용.
- mask_reviewed_negatives: 44행의 음성 존재 BCE만 제외. 양성으로 바꾸지 않음.

두 arm 모두 선택된 EAT adapted의 음성 존재 최종 출력 weight/bias에서 시작한다.
EAT backbone, adapter, 정규화, projection, layer 혼합, pooling은 모두 고정한다.
학습 가능한 매개변수는 arm당 193개다. File/Voice/Music fake와 Music Presence
가중치는 바꾸지 않는다. 이 좁은 개입의 실패는 encoder 재학습까지 불가능하다는
증거가 아니다.

## 학습과 계산

기존 corpus→File class→component cell 균형 sampler를 사용한다. seed 20260905,
epoch당 4,096 draws, 6 epochs, batch 64, AdamW learning rate 0.001,
weight decay 0.01, 파일별 logmeanexp 온도 2를 결과 확인 전에 고정한다.
draw 순서는 두 arm이 정확히 같으며 최종 6 epoch 가중치만 비교한다.
개발 점수에 따른 epoch 선택이나 early stopping은 없다.

전체 epoch의 고정 draws에 필요한 TRAIN 특징과 DEVELOPMENT 특징을 먼저 추출하되
개발 특징/정답은 optimizer 입력에 들어가지 않는다. 원본 head의 192차원 window
통계로 출력이 bit-exact 복원되는지 모든 window에서 검증한다. 파일의 모든 window를
집계하며 다른 파일과 섞어 판단하지 않는다. GPU 한 장에서 파일별 특징 추출 프로세스
4개를 실행한다. 보호 평가가 이미 포화시킨 GPU는 사용하지 않는다.

## 개발 결과 보고

학습 완료 후 unchanged/control/masked를 각각 보고한다.

1. EAT parent: Voice Presence 이외 네 열이 정확히 동일해야 한다. 따라서 ADS는
   바뀌지 않아야 하며 CPS/음성 존재 ROC-AUC와 세부 domain을 본다.
2. v73 composition: Voice/Music fake와 Music Presence는 고정하고 새 Voice Presence로
   기존 noisy-OR File을 재계산한다. 이 경우 File EER/ADS도 변할 수 있다.

mask 대 control 차이가 관심 효과다. unchanged 대비 변화만으로 레이블 개입의
효과라고 주장하지 않는다. 공통 학습 지속 효과가 섞이기 때문이다. 원래 높은 개발
CPS에는 ceiling이 있으며 기존 개발셋도 보컬 라벨 오류가 있을 수 있다.
이 개발 실험만으로 Suno 보컬이나 실제 대회 점수가 개선됐다고 주장하지 않는다.

결과가 좋더라도 자동 제출·기존 후보 덮어쓰기·보호 평가 반복 튜닝을 하지 않는다.
출력은 `reports/presence_repair_v76/` 아래 별도 경로에 생성한다.
