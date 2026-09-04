# v45: EAT patch-graph 15% dose-response 후보

## 목적

v45는 v44와 모델·전처리·checkpoint가 완전히 같고 마지막 EAT patch-graph
File/Music logit 비중만 `0.05 → 0.15`로 올린 후보이다. Voice fake와 두 Presence
출력은 exact-v18과 동일하다. 따라서 실제 점수 차이가 생기면 patch expert의 hidden
전이 강도만 측정한다.

현재 리더보드 1위는 총점 `0.837980`, ADS `0.821160`, CPS `0.98932`다. 우리 v18의
ADS는 `0.736341`이므로 SOTA에는 `+0.084819`가 필요하다.

## 잠금 평가

| 평가 | v18 | v44 5% | v45 15% | v18 대비 v45 |
|---|---:|---:|---:|---:|
| factorial holdout ADS | 0.745506 | 0.757896 | **0.775013** | +0.029507 |
| phone factorial ADS | 0.733857 | 0.790929 | **0.847500** | +0.113643 |
| YuE cross-component ADS | 0.828448 | 0.849252 | **0.849252** | +0.020804 |
| 세 평가 단순 평균 | 0.769270 | 0.799359 | **0.823922** | +0.054652 |

로컬 단순 평균은 1위 ADS보다 높지만 평가 세트의 구성비와 생성기는 hidden test와
다르므로 예상 공식 점수로 해석하면 안 된다. 과거 v19/v32/v33에서 큰 residual이
로컬과 반대로 움직였기 때문에 권장 제출 순서는 v44 5%로 방향을 먼저 확인한 뒤,
상승했을 때 v45 15%를 제출하는 것이다.

## 패키지 검증

- 파일: `eat_patch_graph_v45.zip`
- 압축 크기: 7,909,774,213 bytes
- 압축 해제 크기: 7,909,751,677 bytes
- 최상위: `model/`, `script.py`, `requirements.txt`
- entry 112개, 중복 0개, unsafe path 0개
- CRC 오류 없음
- requirements: `onnxruntime-gpu==1.23.2`
- SHA-256: `ee74bd9d7e09747b34df53512e418c60d784796f12d7e797d3a13c62c2dc56ee`
- Suno 13개 CUDA full smoke 성공
- Suno File/Music fake 확률 13/13 모두 0.5 초과

빌더의 `--residual-weight` 인자로 0~0.30 범위의 동일 ablation을 재현할 수 있다.
