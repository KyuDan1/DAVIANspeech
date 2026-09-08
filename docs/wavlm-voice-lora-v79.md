# v79: WavLM의 제한적 Voice 학습 대조 실험

## 질문과 현재 상태

WavLM의 frozen readout이 약했던 것이 학습 가능한 음성 진위 단서의 부재인지,
다운스트림 적응 부족인지 구분한다. WavLM을 곧바로 제출 모델에 추가하는 실험이 아니다.
기본 테스트 2개와 실제 TRAIN 음원 4개의 CUDA gradient 검증을 완료했고, 전체 paired
학습을 시작했다. 완료 결과 전에는 EER 개선을 주장하지 않는다.

## 방법

- control: 고정 WavLM Large + 기존 mean head의 Voice 학습 지속.
- adapted: 같은 초기값·같은 draws + 마지막 4개 층의 Q/V projection에 rank 8 LoRA.

원래 WavLM 파라미터는 양쪽에서 고정한다. LoRA 파라미터는 131,072개다.
3개 깊이의 hidden token과 기존 mean/std head를 쓰며, 두 arm의 초기 출력이 같은지
확인한다. 이는 논문의 모든 층 SLS/SEA 구현이나 전체 backbone fine-tuning과 다르다.
모든 encoder forward는 **한 window씩** 실행해 이웃 파일의 길이·값에 의존하지 않게 한다.
개발 점수로 window 수나 threshold를 고르지 않는다.

6 epoch × 4,096 draws, batch 8, head LR 0.001, LoRA LR 0.0003을 설정 파일에서
학습 전에 고정했다. 마지막 epoch의 두 가중치를 먼저 저장하고 개발 평가를 한 번 수행한다.
Voice 외 출력 head 값은 제출 확률로 사용하지 않는다.

## 데이터와 중요한 제한

엄격한 기존 TRAIN 18,738개 중 Voice 존재·진위 감독을 쓸 수 있는 13,814개를 사용한다.
실제 음성 4,655개, 가짜 음성 9,159개이며 sampler는 corpus/class/cell을 고려한다.
음성 부재 600개와 실제 전경 음성 + 가짜 배경 음악 4,324개는 Voice loss에서 제외한다.
후자는 배경의 가짜 보컬 존재 여부가 불명확할 수 있기 때문이다. 정답을 뒤집지는 않는다.

**이 필터에는 trade-off가 있다.** 실제 음성 + 가짜 음악의 음성 REAL 예제가 줄어들어,
모델이 음악의 가짜 단서를 음성의 가짜 단서로 오인할 위험을 학습 중 충분히 막지 못할 수 있다.
따라서 전체 Voice EER만으로 승격하지 않고, 가짜 음악이 있는 조건의 RF 대 FF와
실제 음악이 있는 RR 대 FR을 별도 확인해야 한다. 이 필터가 최적이라고 확정한 것이 아니다.
이 실험은 별도 보컬용 학습 데이터를 새로 확보한 실험도 아니다.

개발 평가 대상은 기존 development 중 음성이 존재하는 4,016개다. 보호 Suno, 긴
음성 v74 bank, 비공개 평가 자료는 학습·checkpoint 선택에 사용하지 않는다.
v74가 드러낸 긴 파일의 짧은 가짜 구간 문제를 이번 encoder 변경만으로 해결했다고
주장하지 않는다. 그 시간축 학습은 별도 우선 과제다.

## 확인한 구현 동작

TRAIN 음원 4개에서 초기 control/adapted 출력은 bit-exact였고, 실제 backward/step 뒤
LoRA가 갱신됐다. 원래 backbone에는 gradient가 생기지 않았으며 eval 반복 출력도
같았다. 작은 검증의 GPU peak는 1,951.85MiB, 로딩 이후 4.53초였다.
이 수치가 전체 학습 메모리·시간이나 탐지 정확도를 나타내지는 않는다.

- 설정: `configs/wavlm_voice_lora_v79.yaml`
- 학습: `scripts/train_wavlm_voice_lora_v79.py`
- 작은 검증: `reports/wavlm_voice_lora_v79/smoke/report.json`
- 전체 실행: `reports/wavlm_voice_lora_v79/full/`

전체 실행의 `frozen.json`이나 epoch 로그는 완료 표시가 아니다.
`report.json`의 완료 상태와 실제 결과를 확인해야 하며 자동 제출하지 않는다.
