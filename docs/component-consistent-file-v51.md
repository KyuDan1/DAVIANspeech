# Component-consistent File residual v51

2026-09-04 기준. v51은 v50의 모델을 바꾸지 않고 `FILE_FAKE_PROB`만 보정하는
후보다. 핵심은 **모든 Voice/Music specialist가 끝난 뒤** 최종 성분 확률을 읽는 것이다.
따라서 이후 Music expert를 앞에 추가해도 같은 고정식과 가중치를 재튜닝 없이 검증할
수 있다.

## 결론

선택한 고정식은 다음과 같다.

```text
mixed_gate = (VoicePresence >= 0.10) and (MusicPresence >= 0.20)
component  = max(final VoiceFake, final MusicFake)

if mixed_gate:
    FileLogit = 0.80 * v50_FileLogit + 0.20 * componentLogit
else:
    File = v50_File
```

`0.20`은 성분 오류가 File로 전파되지 않도록 제한한 작은 residual이다. Presence는
authenticity 점수를 다시 보정하지 않고 high-recall 혼합 gate로만 쓴다. 처리에는 같은
파일의 다섯 확률만 사용하며, 평가 파일 사이의 통계나 적응은 없다.

## File EER 결과

| 평가군 | 역할 | v50 File EER | v51 File EER | 변화 |
|---|---|---:|---:|---:|
| Channel-invariant train, 3,000 (exact v47 입력) | 선택 및 group audit | 0.16802 | **0.16604** | -0.00198 |
| Codec mixed dev v4, 600 | 선택 | 0.23333 | **0.22000** | -0.01333 |
| Codec mixed blind v5, 240 | 이미 열린 retrospective 선택 | 0.21667 | **0.20278** | -0.01389 |
| Codec mixed blind v6, 240 | 개봉 후 진단 전용 | 0.23333 | **0.18611** | -0.04722 |

blind v6는 식이나 가중치 선택에 사용하지 않았다. 고정된 v51을 사후 적용해 방향성만
확인했으며, 이후 이 데이터에 맞춘 조정은 금지한다.

Voice/Music EER가 변하지 않는다는 전제에서 ADS는 다음과 같이 바뀐다.

| 평가군 | v50 ADS | v51 ADS | 변화 |
|---|---:|---:|---:|
| Codec mixed dev v4 | 0.76000 | **0.76667** | +0.00667 |
| Codec mixed blind v5 | 0.81083 | **0.81778** | +0.00694 |
| Codec mixed blind v6, 진단 전용 | 0.78917 | **0.81278** | +0.02361 |

## 왜 max 20%인가

동일한 gate와 20% 가중치로 대안을 비교했다.

| 방법 | train | dev v4 | blind v5 | blind v6 진단 |
|---|---:|---:|---:|---:|
| 입력 File (train은 v47, 나머지는 v50) | 0.16802 | 0.23333 | 0.21667 | 0.23333 |
| **final component max** | **0.16604** | **0.22000** | **0.20278** | **0.18611** |
| probability noisy-OR | 0.16703 | 0.22111 | 0.21389 | 0.18611 |
| presence-logit max | 0.17197 | 0.23333 | 0.21667 | 0.20000 |
| phone 30% / phone 20% | 0.16802 | 0.21333 | 0.20278 | 0.18611 |
| learned nonnegative monotone | 0.17838 | 0.20778 | 0.20000 | 0.18333 |

- Noisy-OR는 두 성분의 상관된 오류까지 동시에 올려 blind v5 이득이 작았다.
- Presence logit을 fake logit에 더하는 방식은 calibration 오류를 증폭했다.
- 전화 router로 non-phone만 30%까지 올리면 dev v4는 더 좋지만 blind v5 이득이 없고
  group CV가 더 불안정했다. 복잡도 대비 근거가 약해 버렸다.
- 비음수 monotone logistic fusion은 외부 mixed set 숫자는 좋지만 group CV와 전체 train
  EER가 나빠졌다. 학습된 계수는 File `0.20665`, component `0.10272`로 사실상 더 공격적인
  약 33% residual이며, 일반성 근거가 부족해 버렸다.

즉, max 20%는 최고 단일 숫자가 아니라 서로 다른 개발 코퍼스에서 동시에 이득을 내는
가장 단순한 Pareto 선택이다.

## Group-disjoint 확인

Channel-invariant train의 동일 원본 codec 변형이 fold를 넘지 않도록 `MIXTURE_ID`를
묶고, 5개 seed × 5-fold로 평가했다.

| 모델 | 평균 File EER | fold 표준편차 | 최저 | 최고 |
|---|---:|---:|---:|---:|
| v50 File | 0.16951 | 0.02105 | 0.12176 | 0.21484 |
| **v51 max 20%** | **0.16807** | **0.01955** | 0.12669 | **0.20495** |
| noisy-OR 20% | 0.16709 | 0.02054 | 0.12176 | 0.20495 |
| phone 30/20% | 0.16906 | 0.01938 | 0.12669 | 0.20495 |
| learned monotone | 0.17775 | 0.01959 | 0.13033 | 0.21355 |

Voice/Music source를 bipartite connected component로 묶은 더 엄격한 5-fold에서도 v51은
`0.16796 → 0.16661`이었다. 다만 반복 조합 때문에 가장 큰 connected component 하나가
203개 원본을 포함해 fold 크기가 불균형하다. 이 수치를 독립 source benchmark처럼
과대해석하면 안 된다.

## 세부 오류와 한계

`COMPONENT_CASE`는 앞 글자가 Voice, 뒤 글자가 Music이다. 즉 RF는 진짜 음성+가짜
음악이고 FR은 가짜 음성+진짜 음악이다.

- dev v4 concurrent FR: `0.40 → 0.36`, FF: `0.20 → 0.16`
- dev v4 partial FR: `0.24 → 0.22`, RF: `0.34 → 0.32`
- blind v5 partial RF: `0.50 → 0.45`
- 반대로 dev v4 concurrent RF는 `0.24 → 0.26`, sequential RF는 `0.20 → 0.24`

현재 Voice specialist가 Music specialist보다 강하므로 FR 이득이 더 뚜렷하고 RF는 아직
불안정하다. 이것은 v51 가중치를 더 올릴 이유가 아니라 **새 Music expert가 RF를 실제로
개선하는지 먼저 확인해야 한다**는 뜻이다. 새 Music 후보를 v50 앞단에 넣은 뒤 v51의
동일한 `0.20`을 마지막에 적용해 상호작용만 확인한다.

Channel별로 dev v4 clean `0.1000 → 0.0667`, G.711 `0.2000 → 0.1944`였고,
blind v5 G.722 `0.0972 → 0.0833`, transcode `0.2639 → 0.2500`이었다. Opus-NB는
`0.4167`로 그대로여서 이 보정은 narrowband detector 자체를 대신하지 못한다.

## 구현

- `src/component_consistent_file_fusion.py`: 파일 독립, 마지막 단계 CSV postprocessor
- `scripts/build_component_consistent_file_v51_submission.py`: v50 또는 이후 specialist가
  추가된 package에서 cleanup 직전에 보정을 삽입
- `scripts/evaluate_component_consistent_file_v51.py`: sweep, phone route, monotone fusion,
  group CV 및 RF/FR/FF/RR 재현
- `tests/test_component_consistent_file_fusion.py`
- `tests/test_build_component_consistent_file_v51_submission.py`

빌더는 `component_consistent_file_v51/` 디렉터리 생성을 확인했고 focused test는
`5 passed`다. ZIP 생성, 커밋, 제출은 하지 않았다. 이 후보는 File 축의 작은 개선이며,
리더보드 SOTA 달성을 보장하거나 주장하지 않는다.
