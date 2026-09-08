# v47 File compositor audit v2

## 결론

**v47의 noisy-OR 30%를 유지한다.** 단순 수식 또는 작은 monotone logistic으로
`concurrent fake-voice + real-music`을 안전하게 고치는 후보는 찾지 못했다.

factorial dev에서는 v18 Voice, v46 Music, component-query Voice를 함께 쓰는
3-source log-sum-exp가 매우 좋아 보였다. 다른 10개 controlled File contrast의
EER을 하나도 악화시키지 않으면서 다음과 같이 개선했다.

| factorial dev | v47 noisy-OR 30% | dev-selected 3-source LSE |
|---|---:|---:|
| 전체 File EER | 0.2382 | **0.1982** |
| concurrent VF+MR File EER | 0.360 | **0.240** |
| 다른 controlled contrast 최대 EER 변화 | 0 | **0** |

그러나 설정을 고정한 뒤 잠금 평가를 한 번 수행하자 이 개선은 전이되지 않았다.

| locked set | v47 File EER | selected File EER | v47 ADS | selected ADS |
|---|---:|---:|---:|---:|
| factorial holdout | **0.2324** | 0.2476 | **0.7758** | 0.7682 |
| phone factorial | **0.1783** | 0.1917 | **0.8264** | 0.8196 |
| YuE | 0.1223 | 0.1223 | 0.8652 | 0.8652 |

따라서 dev-selected 후보는 rejected이고 제출 패키지를 만들 가치가 없다. 이 결과는
File 조합 수식의 문제가 아니라, **동시 혼합에서 일반화되는 fake-voice evidence
자체가 부족한 문제**라는 증거다.

## 데이터 격리와 정확한 기준선

후보 선택에는 `factorial_eval_1200_v2_dev` 400개만 사용했다. v18을 정확히
재구성한 뒤 다음 실제 배포 순서를 적용해 v46 pre-compositor를 만들었다.

1. EAT patch File/Music 5%
2. component-query File 2.5%, Music 5%
3. File compositor 후보

후보와 설정을 확정한 다음에만 factorial holdout, phone, YuE truth를 읽었다. 재구성한
v47 File/Voice/Music EER와 ADS는 기존 `v47_locked_audit.csv`와 절대오차
`1e-9` 안에서 일치한다.

## 비교한 수식

총 4,348개의 label-consistent 후보를 비교했다.

| family | 후보 수 | 엄격한 dev safety 통과 수 | family 내 대표 결과 |
|---|---:|---:|---|
| noisy-OR | 41 | 0 | 목표 0.36→0.32, 다른 셀 최대 +0.04 |
| max | 41 | 0 | 목표 0.36→0.28, 다른 셀 최대 +0.08 |
| 2-source log-sum-exp | 205 | 1 | 목표 0.36→0.32, 전체 0.238→0.232 |
| query-Voice blend + noisy-OR | 410 | 10 | 목표 0.36→0.28, 전체 0.238→0.225 |
| query-Voice blend + LSE | 1,640 | 66 | 목표 0.36→0.24, 전체 0.238→0.208 |
| 3-source LSE | 1,148 | 15 | **목표 0.36→0.24, 전체 0.238→0.198** |
| convex nonnegative logits | 861 | 0 | 목표 0.36→0.28, 다른 셀 최대 +0.08 |

엄격한 safety rule은 다음 세 조건을 모두 요구했다.

1. `concurrent VF+MR` File EER가 v47보다 낮을 것
2. 전체 File EER가 v47보다 높지 않을 것
3. voice-only, music-only 및 나머지 8개 mixed layout/case contrast 중 어느 것도
   v47보다 나빠지지 않을 것

hard routing이나 truth의 audio type/layout/channel을 이용한 routing은 사용하지
않았다.

## 선택된 수식

개발에서 고정된 수식은 다음과 같다. `L`은 logit이다.

```text
E = 4 * log(
      exp(L_voice_v18 / 4)
    + exp(L_music_v46 / 4)
    + 0.05 * exp(L_voice_query / 4)
)

L_file_new = 0.325 * L_file_v46 + 0.675 * E
```

모든 fake-evidence 계수가 음수가 아니므로 각 Voice/Music fake score에 대해
단조 증가한다. component-query Voice는 모델을 새로 실행하는 것이 아니라 v47에서
이미 계산되는 값이다.

## Learned monotone logistic

다음 네 feature family를 5-fold cell-stratified CV로 검사했다.

- v46 File, v18 Voice, v46 Music logits
- 위 logits + Voice/Music presence + 두 presence의 곱
- 위 logits + mixed-presence로 gate한 Voice/Music logit
- File/Voice/Music probability + presence

fake evidence 계수에는 비음수 constraint를 걸고 L2를
`0, 1e-4, 1e-3, 1e-2, 1e-1, 1`로 바꿨다. 가장 좋은 learned 후보는 target
EER를 `0.36→0.32`로 낮췄지만 partial-overlap VF+MR를 `0.28→0.36`으로
악화시켜 safety rule을 통과하지 못했다. presence 및 mixed-presence covariate도 이
trade-off를 없애지 못했다.

`OVERLAP_FRACTION`, `CHANNEL`, `SNR_DB`는 truth에는 있지만 평가 시 얻을 수 있는
prediction column이 아니므로 feature로 쓰지 않았다. 이 값을 학습/평가 때만 쓰면
실제 배포가 불가능하고, 사실상 metadata leakage가 된다.

## 잠금 cell 결과

| cell | v47 | selected | 변화 |
|---|---:|---:|---:|
| factorial concurrent VF+MR | 0.44 | 0.44 | 0 |
| factorial concurrent VR+MF | **0.36** | 0.40 | +0.04 |
| factorial sequential VF+MR | **0.20** | 0.28 | +0.08 |
| phone simultaneous VF+MR | 0.22 | **0.20** | -0.02 |
| phone simultaneous VR+MF | **0.29** | 0.33 | +0.04 |
| phone simultaneous VF+MF | **0.17** | 0.18 | +0.01 |
| YuE concurrent VF+MR | **0.25** | 0.375 | +0.125 |
| YuE concurrent VR+MF | 0.50 | **0.375** | -0.125 |

개발에서 query Voice를 조금 넣으면 `VF+MR`에 맞지만, 새로운 source/generator에서는
`VR+MF`나 sequential 셀과 순위가 바뀐다. 동일 수식이 phone VF+MR에는 조금
도움이 되지만 factorial과 YuE에는 일반화되지 않는다. 이는 file-level formula가
layout을 직접 관찰하지 못하기 때문이다.

## 다음 실험에 주는 제약

1. **v47 noisy-OR 30%를 새 기준선으로 유지한다.** 이번 후보는 package/submit하지
   않는다.
2. 더 복잡한 post-hoc compositor sweep을 중단한다. dev File EER `0.198`이라는 큰
   개선도 두 큰 잠금 세트에서 동시에 역전됐다.
3. 다음 모델은 score 세 개가 아니라 원본 latent에서 `concurrent VF+MR`의
   voice-local evidence를 학습해야 한다. File query가 XLS-R/SPEAR의 voice-local
   token에 직접 cross-attend하도록 하고 EAT music token과는 별도 branch로 둔다.
4. 학습/선택은 generator/source-disjoint `VF+MR` bank를 별도로 만들어야 한다.
   현재 factorial dev 하나에서 compositor를 고르는 것은 충분하지 않다.
5. 합격 gate는 최소한 두 개의 source-disjoint dev bank에서 target File EER 개선,
   `VR+MF`, `VF+MF`, sequential 및 music-only 최대 악화 `≤0.02`로 둔다.

## 산출물

- `audit.py`: 정확한 v46/v47 재구성, dev sweep, monotone CV, 잠금 1회 평가
- `dev_sweep.csv`: 4,348개 수식의 전체 controlled 결과
- `dev_family_summary.csv`: family별 후보 수와 대표 결과
- `learned_cv.csv`: 24개 constrained-logistic CV 결과
- `selection.json`: 잠금 전에 고정한 설정과 safety rule
- `locked_metrics.csv`: v46/v47/selected 전체 결과
- `locked_cells.csv`: mixed cell별 File EER
- `locked_predictions.csv`: exact v47와 selected File score

재현:

```bash
/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python \
  reports/component_query_mhfa_v1/file_compositor_audit_v2/audit.py
```
