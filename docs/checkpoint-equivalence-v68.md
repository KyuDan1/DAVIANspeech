# v68: 학습 평가와 파일별 추론의 일치 검사

범위: 완료된 smoke checkpoint 2개의 실제 개발 오디오 각 8개에 대한 실행 검증.
새 모델의 성능 개선, 공식 점수, 전체 1,200개 제출 검증이 아니다.

## 방법

`scripts/verify_paired_checkpoint_inference_v68.py`는 학습 시 저장한
`dev_predictions.csv`의 ID를 등록된 development role에 대조한다. 학습/locked role의
파일은 허용하지 않는다. 완료 marker 및 smoke 명시 승인을 확인하고, checkpoint를
별도의 inference loader로 읽는다. 같은 파일들을 개별 추론한 뒤 역순으로도 반복한다.

캐시 대비 최대 확률 차이 허용값은 실행 전 고정한 0.0005다. BF16 batching의
수치 차이를 위한 허용값이며 성능이 같은 것을 보장하지 않는다. 파일 순서 반전은
그보다 엄격하게 bit-exact를 요구한다. checkpoint와 예측 CSV의 해시가 검증 도중
변하면 실패한다. 이 검사는 threshold나 model weight를 변경하지 않는다.

## 결과

| checkpoint | 표본 | 저장 평가 대비 최대 확률 차이 | 파일 순서 반전 | 판정 |
|---|---:|---:|---|---|
| v67 저대역 magnitude+phase smoke | 8 | 5.55e-17 | bit-exact | 통과 |
| v64 peak-normalized WPT smoke | 8 | 0.00030428 | bit-exact | 통과 |

WPT 결과는 bit-exact한 평가 재현이라고 표현하지 않는다. 관찰된 차이는 허용값
안이며, 두 경로의 batch 크기와 BF16 수치 연산이 다르다. 이 두 요인이 원인이라고
별도의 FP32 통제 실험 없이 확정하지 않는다.

최종 후보가 나올 경우 실제 완료 checkpoint와 전체 개발 파일에서 다시 검사해야 한다.
특히 작은 확률 차이가 순위를 바꿀 수 있으므로 허용값 통과만으로 EER 동일성을
주장하지 않는다. 배포할 파일별 추론의 예측으로 고정된 개선 기준을 재확인한다.

원자료는 `reports/checkpoint_equivalence_v68/{lowband_smoke,wpt_peak_smoke}/`의
`report.json`, `predictions.csv`다. 실제 입력은 development 오디오이며 synthetic
노이즈만 사용한 앞선 runtime 검사보다 실행 경로 검증 범위가 넓다. 여전히 작게
선택된 smoke 표본이므로 generalization 주장은 하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/verify_paired_checkpoint_inference_v68.py \
  --run reports/lowband_music_v67_smoke --allow-smoke \
  --output reports/checkpoint_equivalence_v68/lowband_smoke
CUDA_VISIBLE_DEVICES=3 python scripts/verify_paired_checkpoint_inference_v68.py \
  --run reports/paired_peak_wpt_v64_smoke --allow-smoke \
  --output reports/checkpoint_equivalence_v68/wpt_peak_smoke
python -m pytest -q tests/test_checkpoint_equivalence_v68.py
```

비교 로직 테스트 7개 통과. 아직 v64/v65/v67 전체 학습은 진행 중이며, 이 검증 때문에
중간 checkpoint를 후보로 승격하거나 locked 평가/ZIP/API 제출을 진행하지 않았다.
