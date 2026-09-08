# v63: WPT 음량 민감도와 peak normalization 검사

상태: 음량 민감도 및 600행 slot 정확도 비교 완료. 전체 File EER 악화로 탈락.
공식 점수 개선을 확인한 모델이 아니다.

## 먼저 정정한 가설

`Wav2Vec2Encoder` 함수의 기본 인자는 `normalize_waveform=True`지만 실제
`SpectraAASIST` 생성자는 마지막 인자로 **False**를 넘긴다. 따라서 WPT 경로가 원래
Spectra의 normalization을 누락했다는 처음의 추측은 틀렸다. 버그 수정이나 원본
전처리 복원으로 주장하지 않는다. 실제 생성자까지 확인한 뒤 그 해석을 정정했다.

이후 실험은 별개의 질문이다: **실제 내용이 같은 음성의 음량만 달라져도 WPT 점수가
크게 변하는가?** 그 변화를 줄이면 분류 정확도도 좋아지는가는 다시 평가해야 한다.

## 실제 측정

authorized codec v4에서 5채널 × real/fake 각 3개 = 30개 파일을 seed 20260905로
선택했다. 기존 제출의 동일 WPT 가중치로 gain 0.1/1/10을 적용했다. 디지털 clipping을
적용하지 않은 float 입력 진단이며, 특히 10배 입력은 실제 PCM 통화를 그대로 재현한
것이 아니다. 비교 대상은 preemphasis 직후 각 창의 최대 절댓값으로 나누는 연산이다.
다른 파일/다른 창의 통계는 사용하지 않는다.

gain=1 대비 gain 0.1/10에서의 평균 File 확률 변화:

| 채널 | 기존 WPT | peak normalization 적용 |
|---|---:|---:|
| clean | 0.1162 | 0.0010 |
| G.711 | 0.1280 | 0.0012 |
| G.722 | 0.1076 | 0.0009 |
| Opus NB | 0.1798 | 0.0020 |
| G.711→Opus | 0.1768 | 0.0012 |

기존 경로는 채널별 6개 중 4~6개에서 gain에 따른 확률 범위가 0.1을 넘었다.
정규화 경로에서는 없었다. float/BF16 연산 때문에 완전한 bit-exact gain invariance는
아니며, 정규화 후 최대 확률 차이도 약 0.00636 남았다.

**출력 안정성은 분류 정확도가 아니다.** 모두 같은 점수를 내는 모델도 안정적일 수
있으므로, 이 결과만으로 제출하거나 EER 개선이라고 해석하지 않는다.
원자료는 `reports/wpt_gain_v63/old_wpt_development_diagnostic/`에 있다.

## 정확도 비교는 별도로 고정

`configs/wpt_peak_v63.yaml`을 full 600행 정확도 비교 전에 동결했다. 기존 가중치,
5 views, temperature 2, 기존 slot 비중 0.4를 유지한다. 변경은 peak normalization
하나이며 다른 네 출력은 문자열까지 유지한다. v60의 학습/선택, v62의 slot 교체
기준은 변경하지 않는다.

File EER 최소 1%p 개선, 채널·도메인 최대 악화 2.5%p 이하 등 같은 개발 관문을
적용한다. 통과해도 prospective, 긴 음성, 파일별 독립성, 전체 runtime 검증이 필요하다.
실패한다면 normalization의 일관성만으로 억지 승격하지 않는다. normalization을
학습과 추론 양쪽에 적용하는 재학습은 별개의 후속 실험이다.

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/audit_wpt_gain_sensitivity_v63.py \
  --output reports/wpt_gain_v63/old_wpt_development_diagnostic
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_wpt_peak_v63.py \
  --output reports/wpt_gain_v63/normalized_slot_development
```

Gain/window-local normalization 및 slot 교체 관련 테스트 9개가 통과했다. 이 tests도
탐지 성능 개선의 근거로 사용하지 않는다.

## 최종 정확도 결과와 후속 질문

기존 최고 File EER **0.23333333 → 0.24000000**으로 악화했다. `pass_gate=false`이며
normalization만 켠 후보는 제출하지 않는다. 원자료는
`reports/wpt_gain_v63/normalized_slot_development/{comparison.json,decision.json}`이다.

채널 내부 결과는 다르다. clean/G.711은 동률, G.722는 0.166667 → 0.172222로 소폭
악화했지만 Opus는 0.333333 → 0.305556, G.711→Opus는 0.300000 → 0.266667로 좋아졌다.
채널별 EER 개선과 pooled EER 개선은 같지 않다. 서로 다른 채널의 점수를 함께 비교하는
순위/운영점 문제를 살펴볼 필요가 있으며, 이 표만으로 calibration이 원인이라고 확정하지 않는다.

따라서 후속 후보는 normalization을 추론에만 추가하지 않고 **학습과 추론 양쪽에
동일하게 적용**하는 것으로 분리한다. 기존 v60 paired BCE와 같은 source split/seed/
augmentation/학습량을 유지하고, normalization 하나의 차이를 검사하는 편이 타당하다.
현재 이 후속 학습을 시작한 것은 아니다. 실패한 후보를 이미 통과한 것으로 바꾸거나
긴 음성 locked bank를 열어 유리한 조건만 고르지 않는다.
