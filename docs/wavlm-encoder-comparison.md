# WavLM 확장 비교 및 GPU 동시 실행 — 2026-09-05

## 목적

기존 제출의 작은 residual을 계속 추가하기 전에, 같은 오디오·학습량·판별 head로
encoder 자체의 음성/음악/혼합 판별 가능성을 비교한다. WavLM은 아직 제출 모델이
아니며, 이 실험의 개발 점수를 실제 리더보드 점수로 해석하지 않는다.

## 왜 WavLM인가

- [WavLM 원 논문](https://arxiv.org/abs/2110.13900)은 masked prediction과
  denoising, 발화 혼합을 이용한 음성 표현 학습을 제안한다. 혼합 조건에서 음성
  단서를 읽는 후보라는 가설은 합리적이지만 음악 생성 탐지 성능을 보장하지 않는다.
- 여기서는 원본 파형을 입력하여 내부 특징을 읽는다. 사전학습의 denoising 목적은
  추론 중 diffusion으로 파형을 분리·재생성한다는 뜻이 아니다.
- [공식 모델 카드](https://huggingface.co/microsoft/wavlm-large)의 WavLM Large는
  사전학습 encoder이지 학습 완료된 진위 판별기가 아니다. 한국어 전화음성과
  AI 음악에서의 우수성은 별도로 검증해야 한다.
- 후속 후보는 UniSpeech-SAT, HuBERT이다. 모든 encoder를 무조건 합치는 대신
  WavLM부터 동일 조건으로 비교하고, 유망한 표현의 미세조정을 다음 단계로 둔다.

## 고정된 비교 조건

기존 `configs/common_encoder_probe.yaml`은 보존한다. 추가 설정
`configs/common_encoder_probe_wavlm.yaml`은 checkpoint 목록에 WavLM만 추가한다.
seed, train/development, batch 4, 4096 draws/epoch, 6 epochs, 10.24초 전체 구간
coverage, mean/attention 두 head, 손실과 선택 기준은 동일하다.

WavLM의 24개 block 중 6/12/24번째 출력을 사용한다. 공식 feature extractor로
실제 오디오 구간만 정규화하고 뒤를 0으로 채운다. padding mask는 encoder와 head에
전달한다. encoder는 동결하고 5개 출력을 내는 공통 head만 학습한다.
모든 파일의 예측은 독립적이다. 모델 파일은 local-only로 읽는다.

기존 3개 실험의 source snapshot은 변경하지 않는다. 비교 도구는 WavLM 확장 시에도
설정, source manifest, 실제 train/development ID 순서와 매 epoch 샘플링 순서를
검사한다. **완료된 실험만** 최종 비교에 포함한다.

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 /home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python scripts/train_common_encoder_probe.py --encoder wavlm --config configs/common_encoder_probe_wavlm.yaml --output reports/common_encoder_probe/wavlm

/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python scripts/compare_common_encoder_probe.py --include-wavlm --output reports/common_encoder_probe/comparison_with_wavlm
```

출처: `microsoft/wavlm-large`, revision
`c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`.
모델 카드가 연결하는 [라이선스 원문](https://raw.githubusercontent.com/microsoft/UniSpeech/main/LICENSE)은
CC BY-SA 3.0이다. MIT라고 기록하지 않는다. 향후 배포 시 출처·라이선스 및
변경사항 고지를 포함해야 하며, fine-tuned 파생물 배포 조건도 확인해야 한다.

## GPU 배치

사용자 요청에 따라 GPU 하나당 한 작업으로 제한하지 않는다.

| GPU | 동시 배치 |
|---|---|
| 2 | 진행 중인 XLS-R 공통 비교 + 저대역 음악 magnitude 실험 |
| 4 | 진행 중인 저대역 음악 magnitude/phase 실험 + WavLM smoke, 통과 후 본 실험 |
| 0 | 완료된 SPEAR의 파일 단위 독립 추론 검증 |

GPU 메모리뿐 아니라 연산 사용률, CPU·스토리지 병목도 확인한다. 다른 팀원의
작업은 중단·이동하지 않는다. 비교의 공정성을 위해 기존 학습 batch는 변경하지
않는다. 동시 실행 자체를 속도 개선의 실측 증거로 주장하지 않는다.

## 해석 범위

개발셋과 train은 등록된 원천 ID 기준으로 분리되어 있다. 그러나 생성기 종류의
완전한 미지 조건 실험은 아니며, 개발 결과만으로 일반화나 공식 1등을 주장할 수 없다.
전체 점수 외에 음성 전용/음악 전용/혼합, RR/RF/FR/FF, 순차/겹침, 채널별 EER을
분리해 본다. WavLM 결과는 실험 완료 후 비교 산출물에 기록한다.

## 현재 실행 검증 결과

- WavLM 실제 가중치 smoke 통과: 8 train 파일, 10 windows,
  token shape `[10, 3, 511, 1024]`, head별 101,908 parameters.
  이는 실행/gradient 검사이며 탐지 정확도 평가가 아니다.
- 공통 코드 및 WavLM/비교기/독립 추론 관련 단위 테스트 26개 통과.
- WavLM 본 학습 실행을 GPU 4에 시작했다. 완료 점수는 아직 없다.
- 별도로 SPEAR mean의 80파일 독립 추론 검증은 **실패**했다.
  저장된 batch 예측 대비 최대 확률 차이 0.197727, 파일 순서를 뒤집은 독립 추론은
  bit-exact였다. batch 처리와 파일 단독 처리의 차이 원인은 추가 조사 대상이다.
  head-only smoke 통과만으로 encoder까지 독립적이라고 볼 수 없으므로, SPEAR의
  기존 개발 점수는 잠정치로 취급하고 모델 승격을 보류한다. 이 새 실험의 실패를
  과거 제출 SPEAR 경로에도 그대로 일반화하지 않는다.
  증거: `reports/common_encoder_probe/spear_independent_mean/report.json`.
