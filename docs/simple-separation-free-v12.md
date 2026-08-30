# v12: source-disjoint eval과 separation-free detector

## 결론

가장 단순한 조합이 현재 내부 평가의 최고 ADS를 기록했다.

- `VOICE_FAKE_PROB`: 원본 오디오를 NII XLS-R-2B AntiDeepfake에 직접 입력
- `MUSIC_FAKE_PROB`: 원본 오디오의 고주파 Fourier fakeprint
- `FILE_FAKE_PROB`: 두 확률의 `max` (한 성분이라도 fake이면 파일 fake)
- Presence: 기존 PANNs를 그대로 유지
- 사용하지 않음: source separation, EAT, SPEAR, 추가 mixture head

즉, 분리 결과에 생길 수 있는 artifact를 탐지 입력에 넣지 않는다. 순차 혼합과 동시 혼합도 같은 원본 파형에서 판단한다.

## 데이터 분할과 누수 방지

데이터 역할은 `configs/data_partitions.yaml`에 고정했다.

- train: 학습 전용 세트 3종
- development: 모델과 조합을 고르는 세트 5종
- locked eval: `source_disjoint_mixed_locked_v1` 한 종

locked eval은 80개, 8초 길이이며 다음처럼 균형을 맞췄다.

- sequential 40 / simultaneous 40
- 각 조건에서 `(real voice, real music)`, `(real, fake)`, `(fake, real)`, `(fake, fake)`가 각각 10개
- 기존 두 source-disjoint 개발 혼합 세트와 voice source 중복 0
- 기존 두 source-disjoint 개발 혼합 세트와 music source 중복 0
- component pair 중복 0

`src/data_guard.py`는 ID뿐 아니라 `GROUP_ID`, `VOICE_SOURCE_ID`, `MUSIC_SOURCE_ID`, `SOURCE_FILE`을 비교한다. `scripts/train_embedding_head.py`는 학습 전에 이 검사를 자동 실행하므로 locked eval 원본이 train에 들어가면 즉시 실패한다.

## 후보 선택: development만 사용

분리 stem의 음성 점수와 원본 음성 점수를 섞는 비율을 6개 개발 조건에서 비교했다. `0.9 raw + 0.1 stem`의 최악 조건 ADS가 raw-only보다 0.001 높았지만, 차이가 너무 작고 분리 artifact·실행시간·패키지 복잡도를 감수할 근거가 되지 못했다. 따라서 locked eval을 보기 전에 raw-only를 고정했다.

고정 후보의 6개 개발 조건 ADS는 다음과 같다.

`0.824, 0.769, 0.862, 0.774, 0.863, 0.824`

- 평균 ADS: 0.819
- 최악 조건 ADS: 0.769

## locked eval: 한 번만 평가

locked 결과를 본 뒤에는 가중치나 규칙을 바꾸지 않았다.

| 방법 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| 기존 v11 MoE | 0.150 | 0.125 | 0.325 | 0.8025 |
| v12 단순 separation-free | **0.150** | **0.075** | **0.300** | **0.8200** |

조건별 v12 결과:

| 조건 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| sequential | 0.200 | 0.000 | 0.350 | 0.795 |
| simultaneous | 0.100 | 0.100 | 0.150 | 0.885 |

CPS는 모든 locked 샘플에 음성과 음악이 존재하도록 만든 세트라 ROC-AUC를 계산할 수 없다. 이 세트의 목적은 혼합 상황의 ADS와 component EER 평가다. Presence는 실제 리더보드에서 이미 약 0.989이므로 변경하지 않았다.

## 해석

분리와 복잡한 routing을 제거했는데 Voice EER과 Music EER이 모두 좋아졌다. 현재 증거에서는 “분리해야 혼합을 풀 수 있다”보다 “원본 artifact를 보존한 서로 다른 전문가가 각 성분을 직접 본다”가 더 안정적이다.

남은 가장 큰 병목은 Music EER 0.300이다. 다만 locked eval은 더 이상 모델 선택에 사용하지 않는다. 다음 음악 개선은 새로운 source-disjoint development split에서만 선택하고, 별도의 새 locked eval로 최종 확인해야 한다.

## 재현

```bash
python src/data_guard.py --config configs/data_partitions.yaml

python src/simple_pipeline.py \
  --test-dir data/eval/source_disjoint_mixed_locked_v1/audio \
  --sample-submission data/eval/source_disjoint_mixed_locked_v1/sample_submission.csv \
  --output output/source_disjoint_mixed_locked_v1_simple.csv

python src/evaluate_diagnostic.py \
  output/source_disjoint_mixed_locked_v1_simple.csv \
  data/eval/source_disjoint_mixed_locked_v1/truth.csv
```
