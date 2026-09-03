# 전화/협대역 품질 라우터 v1

## 결론

기존의 `4.2 kHz 이상 에너지 < 3e-6` 규칙을 443차원 음향 특징과 작은 선형
분류기로 교체했다. 최종 모델은 **6.1 KiB**, CPU 추론 비용은 오디오 로딩을
포함해 평균 **65 ms/file**이며, 1,200개 기준 약 **78초**다.

여기서 라우터가 답하는 질문은 “이 파일이 전화에서 왔는가?”가 아니다.
정확한 질문은 **“협대역 열화 때문에 전용 딥페이크 탐지 expert를 실행해야
하는가?”**다. 따라서 G.711/G.726/8 kHz Opus/PSTN은 양성이고, 7 kHz 이상을
보존하는 G.722와 wideband Opus는 일반 경로로 보내는 음성적 의미의 음성이다.

최종 v6은 원본 혼합파형에서 직접 판단한다. 음성·음악 분리나 diffusion
separator를 전혀 거치지 않으므로 분리기가 새로운 생성 artifact를 만드는
문제도 없다.

## 왜 이렇게 설계했나

전화 코덱 식별 연구는 LPC/잔차, spectral flatness·centroid·dynamics, ZCR,
MFCC와 통계량을 결합하고 화자·잡음·코덱을 분리해 평가하는 방식을 사용한다
([Sharma et al., non-intrusive codec identification](https://www.commsp.ee.ic.ac.uk/~sap/uploads/publications/Sharma2012.pdf)).
ASVspoof 2021도 TTS/VC 음성을 전화·VoIP 채널에 통과시키는 조건을 별도 평가했고
([evaluation plan](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)),
실제 참가 시스템들은 codec/channel augmentation이 채널 강건성에 도움이 된다고
보고했다
([UR-AIR](https://www.isca-archive.org/asvspoof_2021/chen21_asvspoof.html),
[CRIM](https://www.isca-archive.org/asvspoof_2021/kang21b_asvspoof.html)).
최근 ADD-C도 codec과 packet loss 조합에서 탐지기가 크게 약해지고 targeted
augmentation이 효과적임을 확인했다
([Shi et al., 2025](https://arxiv.org/abs/2504.12423)).

또한 Opus 규격은 8 kHz narrowband와 16 kHz wideband를 명시적으로 구분한다
([RFC 7587](https://datatracker.ietf.org/doc/html/rfc7587)). 그래서 단순히
“VoIP codec이면 전화”라고 라벨링하지 않고, downstream expert 선택에 실제로
필요한 대역폭을 기준으로 라벨을 정의했다.

## 평가 데이터 프로토콜

모든 변형은 **최종 음성+음악 mixture 전체에 적용**한다. 실제 통화에서도
마이크에 들어온 전체 신호가 코덱을 통과하기 때문이다. 한 원본에서 나온 여러
codec 버전은 항상 같은 split에만 있어 source leakage가 없다.

| 역할 | 특징 행 수 | 협대역/일반 | 오디오 유형 | 사용 방법 |
|---|---:|---:|---|---|
| train | 21,200 | 12,700 / 8,500 | voice, music, mixed | 모델 fitting |
| development | 18,624 | 9,312 / 9,312 | voice, music, mixed | 모델·threshold 선택 |
| 기존 locked OOD | 4,109 | 2,429 / 1,680 | voice, music, mixed | 동결 후 1회 평가 |
| 외부 최종 평가 | 2,400 | 1,200 / 1,200 | 새로운 voice 원본 | 동결 후 최종 확인 |

Train에는 G.711 μ/A-law, G.726, 8 kHz Opus, PSTN band-pass,
packet-loss concealment, transcoding, 후단 잡음·clipping을 넣었다. 일반 경로의
hard negative로 clean, MP3/AAC/Ogg, audio-mode Opus, 5/6 kHz low-pass,
G.722/wideband Opus를 넣었다. 특히 5 kHz low-pass는 “고주파가 적다”만으로
협대역이라고 판단하는 지름길을 막는다.

초기 학습은 mixed audio에 편향돼 voice OOD 검출률이 낮았다. 이를 고치기 위해
기존에 학습 사용이 허가된 source component만 다시 노출한
`phone_router_voice_train_v1`과 `multigen_music_v2`를 추가했다.
`src/data_guard.py --training` 검사를 통과시킨 뒤 특징을 추출했으며, 평가 원본은
train에 넣지 않았다.

외부 최종 평가는 모델을 동결한 다음, 학습에 사용하지 않은 OpenSLR의
[Mini LibriSpeech dev-clean-2](https://www.openslr.org/31/)에서 결정론적으로
고른 300개 음성으로 만들었다. clean/5 kHz low-pass/audio Opus/G.722를 음성,
FFT 협대역/G.726+후단잡음/협대역 Opus+clipping/G.726→G.711 재인코딩을 양성으로
평가했다. 이 결과를 보고 모델이나 임계값을 다시 바꾸지 않았다.

## 모델

각 파일에서 다음을 계산한다.

- 32개 선형 주파수 band의 평균·표준편차·10/50/90 분위수
- band의 시간 변화량과 16개 cepstral envelope 계수
- centroid, spread, flatness, 85/95% roll-off, ZCR, RMS
- 250 Hz~7 kHz의 15개 cutoff에 대한 frame 단위와 전체 파일 에너지 비율

총 443차원이다. StandardScaler 뒤에 class-balanced logistic regression
(`C=0.01`)을 붙였다. ExtraTrees와 histogram boosting도 비교했지만, 보지 못한
후단 잡음·clipping에서 선형 모델이 가장 안정적이었다. 임계값 `0.5157917`은
development false-positive 목표 0.1%로 고정했다.

## 결과

| 평가 | 방법 | AUC | EER ↓ | 전화 TPR ↑ | 일반 FPR ↓ | 최악 변형 TPR ↑ |
|---|---|---:|---:|---:|---:|---:|
| development OOD | 기존 단일 high-band 규칙 | 0.9434 | 16.97% | 8.61% | 0.097% | 0% |
| development OOD | v5, mixed 중심 | 0.999802 | 0.311% | 98.30% | 0.097% | 87.50% |
| development OOD | **v6, 단일 성분 보강** | **0.999842** | **0.247%** | **99.46%** | **0.097%** | **95.36%** |
| 기존 locked OOD | v5 | 0.991976 | 2.728% | 94.44% | 0.893% | 82.28% |
| 기존 locked OOD | **v6** | **0.993447** | **2.506%** | **95.31%** | **0.833%** | **83.65%** |
| 외부 LibriSpeech | v5 | 0.998160 | 0.917% | 98.25% | 0% | 93.00% |
| 외부 LibriSpeech | **v6** | **0.998394** | **0.667%** | **98.50%** | **0%** | **94.00%** |

외부 최종 세트에서 기존 hard rule의 TPR은 41.58%였고 FPR은 0%였다. v6은
동일한 FPR 0%에서 TPR 98.50%로 올랐다. 가장 남은 약점은 두 번 재인코딩한
`G.726→G.711`이며, 기존 locked OOD 83.65%, 외부 음성 94.0%다.

오디오 유형별 기존 locked OOD의 v6 결과는 mixed TPR 98.09%/FPR 0%, music
TPR 97.47%/FPR 1.84%, voice TPR 88.86%/FPR 0.80%다. voice 수치가 상대적으로
낮은 이유는 협대역 음성 중 고주파가 부분적으로 되살아난 이중 transcoding이
가장 어려웠기 때문이다. 단일 성분 보강 전 voice TPR 87.68%보다 개선됐다.

## 구현 및 사용

- 특징과 portable 추론: `src/telephone_router.py`
- 채널 시뮬레이터: `src/telephone_channel.py`
- 데이터/학습/고정 평가: `scripts/*telephone_router*.py`
- 최종 체크포인트: `model_heads/telephone-router-narrowband-v1.npz`
- `src/simple_pipeline.py`는 체크포인트가 주어지면 새 probability/threshold를
  사용하고, 없을 때만 예전 hard rule로 fallback한다.

중요한 제한은 이 라우터 자체가 ADS를 올려 주는 탐지기는 아니라는 점이다.
라우터가 정확해도 전화용 voice/music expert가 일반 expert보다 좋아야 최종
점수가 오른다. 따라서 다음 제출 ablation은 **anchor + 라우터만 교체**와
**anchor + 라우터 + 전화 expert**를 분리해야 한다. v14 하락만으로 라우터가
나쁘다고 결론 내릴 수 없는 이유도 당시에는 hard rule과 전화 expert가 동시에
바뀌었기 때문이다.
