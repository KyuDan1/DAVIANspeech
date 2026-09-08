# v80: XLS-R의 짧은 가짜 음성 구간 학습

## 질문

v74에서 전체 대체 후보 v73은 30–60초 파일의 일부에만 가짜 음성이 있는 조건을
안정적으로 처리하지 못했다. v71/v72에서는 이미 EAT에 긴 합성 파일과 구간 정답을
학습했지만 전체 성능이 개선되지 않았다. 따라서 **긴 합성 데이터 자체가 새로운 방법은
아니다.** 이번에는 음성 진위에 강했던 XLS-R의 forensic 표현과 Voice 한 과제에 집중한다.

이것은 아직 검증되지 않은 연구 후보이며 제출/1등 성능을 주장하지 않는다.

## 통제된 비교

- XLS-R encoder 및 기존 mean head의 정규화·96차원 projection·layer 결합은 고정한다.
- 세 깊이의 token을 기존 Voice layer 가중치로 결합한다. 전체 mean/std로 요약하면
  기존 Voice head와 동일한 logit이 나오는지 실제 forward마다 검사한다.
- 51 token(중심 간 약 1.02초)의 이동 mean/std를 10 token(0.2초) 간격으로 계산한다.
  10.24초 window는 전체 파일을 덮고, 겹친 구간은 window 중심에 가까운 쪽만 집계한다.
- 같은 초기값의 193-parameter linear readout 두 개를 학습한다.
  `file_only`는 파일 Voice BCE만, `dense`는 같은 BCE에 가짜 구간 위치 BCE를 0.5배 더한다.
  파일 점수는 유효 시간 bin의 logmeanexp(온도 5)다.
- encoder는 두 head가 공유한다. 같은 파일을 두 번 encoder에 넣지 않는다.
  서로 다른 파일을 encoder batch로 섞지도 않는다.

**주의:** XLS-R의 attention은 10.24초 window 전체 문맥을 본다. 이동 pooling을 사용한다고
encoder의 receptive field까지 1초가 되는 것은 아니며, 파형 분리 방법도 아니다.

## TRAIN 생성과 누출 방지

기존 v78 strict TRAIN inventory와 v71의 raw source catalog를 hash·source/group 기준으로
다시 검사한다. 보호 eval 원천은 학습하지 않는다. 별도 등록되지 않은 새 데이터는 쓰지 않는다.

각 pair는 실제 배경 음성에 **다른 group의 실제 음성 / 가짜 음성**을 같은 위치·길이로
삽입한다. 두 class 모두 source change와 같은 fade 처리를 받아 편집 흔적 자체가 정답이
되지 않게 한다. 절반은 음성만, 나머지는 같은 실제 음악을 두 class에 함께 넣는다.
가짜 음악은 이번 Voice 국소화 학습에 넣지 않는다. 따라서 RF/FF 전체 문제를 해결한
학습이라고 해석해서는 안 된다.

- 길이 4/8/16/32/60초, 삽입 0.5/1/2/4초(파일 절반 이하), 위치 균등 추출.
- 실제/가짜 pair에 같은 채널 및 난수 key: clean, G.711, G.722, Opus NB, G.711→Opus.
- 혼합 SNR −10/−5/0/5/10dB, 가짜 음성 generator 균등 선택.
- raw payload hash, source/group, 위치, 길이, SNR, 채널을 학습 draw별 기록.
- 구간 경계 ±0.08초는 dense loss에서 제외한다. 시간 정답은 알려진 원천 배치이며
  모든 프레임에 실제 발화가 있다는 oracle VAD annotation은 아니다.

두 head는 매 epoch 같은 512 pair, 4 epoch, LR 0.001로 학습한다. 마지막 epoch 가중치를
먼저 저장하고 기존 development의 음성 존재 4,016개를 한 번 평가한다. 조기 종료나
보호 v74/Suno의 성능을 보고 checkpoint를 선택하지 않는다.

## 판단 기준과 한계

같은 실행에서 기존 global Voice head도 development에 재계산한다. 전체 EER,
RR 대 FR, RF 대 FF, 순차/중첩, 채널별 결과를 함께 본다. RF에는 배경 가짜 보컬 진위
주석이 불명확한 데이터가 포함될 수 있어 이를 별도 한계로 기록한다.

기존 짧은 development 개선만으로 승격하지 않는다. 긴 파일과 새로운 원천에서의
전향적 검증이 추가로 필요하다. 이미 노출된 v74를 새 blind set으로 부르거나 반복해서
가중치를 맞추지 않는다. 이번 출력은 Voice 하나이며 File/Music/CPS를 자동 교체하지 않는다.

실행: `scripts/train_local_voice_v80.py`, 설정: `configs/local_voice_v80.yaml`.
`--smoke`는 TRAIN pair 2개만 사용하고 development/eval을 읽어 점수를 계산하지 않는다.
완료 판단은 출력 디렉터리의 `report.json`을 기준으로 한다.

## 구현 검증과 시작 기록

기본 테스트 5개가 통과했다. 실제 strict TRAIN pair 2개(4 waveform)의 CUDA smoke에서는
두 head의 초기 출력이 bit-exact였고, 둘 다 학습 후 갱신됐으며 encoder에는 gradient가
없었다. 기존 global head와 projection 재구성의 최대 logit 차이는 7.16e-7이었다.
peak CUDA 8,486MiB, 모델 로딩 이후 약 26.5초이며 정확도 평가는 아니다.

첫 smoke는 새 환경의 PATH에 ffmpeg가 없어 학습 전에 종료됐다. 기존 환경에 설치된
ffmpeg의 절대 경로를 PATH에 추가하여 두 번째 smoke가 완료됐다. 실패 기록을 덮어쓰지
않았다. 전체 실행에는 완료 smoke의 config/code/weights hash 일치를 요구한다.

```bash
PATH=/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin:$PATH \
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONWARNINGS=ignore::FutureWarning HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/tmp/davian_grader_v77.o2Sm4v/venv/bin/python scripts/train_local_voice_v80.py \
  --smoke-report reports/local_voice_v80/smoke_v2/report.json \
  --output reports/local_voice_v80/full
```

두 학습 head는 GPU 1의 같은 encoder 출력을 공유한다. full의 최종 완료 및 EER은 아직
확인되지 않았으며, TRAIN loss가 감소하는 것만으로 성능 향상을 주장하지 않는다.
