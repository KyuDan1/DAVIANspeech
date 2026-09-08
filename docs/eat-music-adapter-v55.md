# v55: raw-audio EAT Music adapter 실험

## 결론

v55는 기존 frozen EAT cache 위의 작은 head를 반복하지 않고, **분리하지 않은 원본
mixture를 EAT에 넣어 마지막 두 encoder block 뒤 adapter를 실제로 학습**했다. v4
development에서는 Music EER이 v50 `0.27333`에서 `0.24667`로 좋아졌지만, 고정한
후 처음 연 v5/v6/v7에서 각각 개선/중립/악화가 나왔다. 따라서 일반성 기준을
통과하지 못했고 제출 후보에서 제외한다.

- v5: Music EER `0.19167 -> 0.18333`, ADS `+0.00250`
- v6: Music EER `0.25833 -> 0.25833`, ADS 변화 없음
- v7: Music EER `0.30556 -> 0.31667`, ADS `-0.00333`
- blind v8: 접근·추론·점수화하지 않음

이 결과는 end-to-end adaptation 방향 자체가 틀렸다는 뜻은 아니다. 작은 EAT-base,
6초 crop, 제한된 pair loss만으로는 generator/source OOD와 채널 간 score offset을
동시에 제어하지 못했다는 결과다.

## 1. 기존 shallow EAT head가 실패한 이유

이전 계층형 EAT와 patch-graph 모델은 EAT backbone을 고정한 뒤 cache된 표현 위에
head만 학습했다. 로컬 factorial/phone/YuE에서는 큰 이득이 있었지만 실제 제출의
v32/v33은 v18보다 낮았다. 반복 실험에서 다음 패턴이 확인됐다.

1. EAT 표현이 presence·audio semantics에는 강하지만 생성 artifact에 맞게 바뀌지
   않았다.
2. cache head는 train loss와 로컬 dev가 좋아져도 generator identity, 장르, 채널
   shortcut을 학습했다.
3. 6초 전체를 평균/표준편차 같은 통계로 먼저 축약하면 희소한 시간·주파수 artifact
   위치가 사라졌다.
4. 큰 residual은 hidden에서 쉽게 역전됐다. v55도 이 때문에 residual을 v4에서
   미리 `0.15`로 고정했다.

## 2. v55 방법

입력은 source separation을 거치지 않은 16 kHz 원본 mixture다. 각 6초 crop을
128-bin Kaldi fbank로 변환하고 EAT-base에 넣는다.

- EAT base checkpoint는 완전히 동결하고 byte-identical하게 재사용
- block 10과 11 뒤에 `768 -> 32 -> 768` bottleneck residual adapter 삽입
- 마지막 두 layer의 patch token을 축약하지 않고 모두 유지
- head별 layer weighting 후 4-head attentive weighted mean/std pooling
- CLS는 별도 layer weighting으로 결합
- 학습 가능한 파라미터 427,341개, checkpoint 1.7 MB
- 파일의 최대 세 view는 log-mean-exp(`T=5`)로 결합

adapter 이전 block은 autograd graph 없이 실행하고, 첫 adapter 이후 frozen block은
입력 gradient만 통과시킨다. 따라서 base weight는 바뀌지 않지만 adapter가 실제 EAT
token representation을 바꾼다.

## 3. 데이터·누수 방지

학습에는 `configs/data_partitions.yaml`의 train role만 사용했다.

- external/mixed devvoice/mixed FakeMusicCaps train
- MixFake train
- telephone mixed train
- temporal mixed v2의 `truth_train.csv`만 사용
- channel-invariant factorial train

Music-present 17,760개를 대상으로 corpus, RR/RF/FR/FF, layout, channel, music
generator/source group의 sampling mass를 균형화했다. channel-invariant bank의
2,400개 행은 같은 mixture의 codec 파트너를 연결해 logit consistency를 적용했다.
학습 source와 v4의 `VOICE_SOURCE_ID`, `MUSIC_SOURCE_ID` 교집합은 각각 0개였다.

checkpoint 선택은 `codec_mixed_dev_v4`의 Music EER만 사용했다. v5/v6/v7은 epoch,
가중치, threshold 선택에 사용하지 않고 checkpoint와 residual `0.15`를 고정한 뒤 한
프로세스에서 각각 한 번만 열었다. v8은 scorer에서 명시적으로 거부한다.

## 4. 재현 설정과 hash

```bash
CUDA_VISIBLE_DEVICES=7 \
python -u scripts/train_eat_music_adapter_v55.py \
  --output-dir reports/eat_music_adapter_v55/seed00 \
  --epochs 8 --samples-per-epoch 6144 --batch-size 32 \
  --eval-batch-size 24 --workers 4 --learning-rate 1e-4 \
  --patience 3 --seed 20260904
```

- base EAT SHA-256: `5fa59385650eeb7b6c74414461306e055d943523cd7206c181e2d329969fff1b`
- adapter SHA-256: `d544ca01efc625e3a7f3cd85770dd753da81e3b600f4226f24ca19030d91d58c`
- 선택 epoch: 5
- train loss: epoch 1 `0.43963`, epoch 5 `0.14433`, epoch 8 `0.11182`
- v4 direct Music EER: epoch 1 `0.26333`, best `0.25000`, epoch 8 `0.27000`

train loss가 계속 감소하는데 OOD EER은 다시 나빠진 것이 source/channel shortcut의
정량 증거다.

## 5. 고정 후 평가

| bank | v50 Music EER | v55 15% Music EER | 변화 | v50 ADS | v55 ADS |
|---|---:|---:|---:|---:|---:|
| v4 development | 0.27333 | **0.24667** | -0.02667 | - | - |
| v5 retrospective | 0.19167 | **0.18333** | -0.00833 | 0.81083 | **0.81333** |
| v6 retrospective | 0.25833 | 0.25833 | 0 | 0.78917 | 0.78917 |
| v7 retrospective | **0.30556** | 0.31667 | +0.01111 | **0.76278** | 0.75944 |

v7의 channel별 EER은 G.711과 transcode에서 오히려 좋아졌지만 전체 EER은
나빠졌다. 이는 각 channel 내부 순위보다 **channel 사이의 score offset**이 달라져
전체 positive/negative 순서가 뒤집혔다는 뜻이다. 즉 단순 channel별 EER만 보고
invariance를 주장하면 안 된다.

직접 expert EER도 v5/v6 `0.35`, v7 `0.37778`로 anchor보다 낮았다. 낮은 상관 때문에
v4 residual에는 도움이 됐지만 독립 expert로 쓸 수준은 아니다.

## 6. 다음 실험에서 고쳐야 할 것

1. 같은 mixture의 clean/codec pair에 동일한 시간 crop을 적용한다. v55는 두 파일의
   random crop 위치가 달라 consistency loss에 content 차이도 섞였다.
2. pair 비중과 invariant objective를 높이되, EER을 각 channel 평균으로만 고르지
   말고 모든 channel을 합친 global EER와 channel 간 logit location 차이를 함께
   최소화한다.
3. EAT-large + AASIST 계열처럼 10초 crop, music oversampling, 실제 codec augmentation,
   frontend/backend 분리 learning rate로 학습한다. EAT-base 두 adapter만으로는
   representation capacity와 artifact locality가 제한됐다.
4. generator 단위 leave-one-group-out를 epoch selection에 포함한다. 현재 v4는 source
   identity는 분리됐지만 music generator family는 train과 일부 겹친다.
5. 다음 candidate는 이미 노출된 v5-v7로 다시 선택하지 않고, 새 prospective blind
   bank에서 한 번만 검증한다.

현재 의사결정은 명확하다: v55 weight를 키우거나 ZIP에 넣지 않는다. 남길 가치는
raw EAT adapter 학습 코드, token-level pooling 구현, 그리고 hidden 전이를 망치는
channel-global calibration 문제의 발견이다.
