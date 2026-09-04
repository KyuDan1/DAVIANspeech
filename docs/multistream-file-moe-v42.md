# v42: separation-free multistream File MoE

## 결론

v42는 v41의 최종 File expert 안에 **2-view multistream expert를 20%** 결합한
고정 logit MoE다. Voice, Music, Voice Presence, Music Presence는 v41과
bit-exact이며 File만 바뀐다.

중간 router도 같은 조건에서 검증했다. 개발셋에서는 phone-aware soft router가
모든 channel subgroup에서 비회귀였지만, 잠금 전화셋에서 ADS가 `-0.00171`
하락했다. 반면 고정 20% MoE는 factorial holdout `+0.00382`, phone
`+0.00086`, YuE `0.00000`으로 모두 비회귀였다. 따라서 router가 아니라 더
단순한 고정 MoE를 채택했다.

## 모델

모델은 source separation 없이 원본 혼합 오디오만 입력받는다. frozen XLS-R의
각 Transformer layer에 다음 세 종류의 새 prompt를 넣고, Spectra-AASIST
backend와 Voice/Music/File head를 함께 학습했다.

- base prompt 10개
- multi-scale instantaneous-frequency prompt 6개
- XLS-R frame의 TKEO energy와 feature flux로 조절되는 texture prompt 6개

아이디어는 혼합 오디오를 직접 처리하는
[MixFake](https://arxiv.org/abs/2605.23201)와 deep prompt 기반
[WPT](https://arxiv.org/abs/2504.06753)를 참고했고, 저장소 코드를 복사하지
않고 논문의 신호 정의를 독립 구현했다. separator가 새로운 생성 artifact를
만들 수 있다는 우려를 피하면서, 음성/음악의 국소 artifact를 한 latent
sequence 안에서 보게 하는 것이 목적이다.

## 데이터 격리와 학습

학습에는 등록된 8개 train bank, 총 18,936개 원본 예제를 사용했다.

- `forensic_call_train_v1`
- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2`의 train split
- `channel_invariant_factorial_train_v1`

checkpoint 선택은 학습에 쓰지 않은 8개 dev bank, 총 3,976개 파일의
`File 62.5% + Music 37.5%` 평균/최악 도메인 점수로만 수행했다. factorial
holdout, phone factorial, YuE는 weight를 고정한 뒤 한 번만 확인하는 acceptance
gate로 사용했다. Suno는 양성-only stress audit이며 학습과 모델 선택에 쓰지
않았다.

## cold-start와 warm-start

기존 WPT의 Spectra/AASIST/task head 577,291개 파라미터를 이어받고 prompt만
새로 만드는 warm-start도 같은 설정으로 학습했다.

| 모델 | dev 선택점수 | factorial holdout ADS | phone ADS | YuE ADS |
|---|---:|---:|---:|---:|
| cold multistream | **0.81014** | **0.80039** | **0.85404** | **0.86467** |
| WPT warm-start | 0.80305 | 0.79223 | 0.84643 | 0.86340 |

warm-start는 모든 잠금축에서 낮았다. 기존 음성 중심 backend의 prior가 새로운
주파수/질감 prompt의 독립성을 줄인 것으로 해석하며, 최종에는 cold-start 한
개만 사용한다.

추가로 Music loss를 70%, pairwise ranking을 15%로 높인 Music 전용 cold
specialist도 학습했다. 7개 Music dev domain의 평균/최악 EER 선택점수는
`0.78149`로, File+Music을 함께 학습한 모델의 `0.80165`보다 낮았다. Music에만
강하게 맞추면 generator-domain 간 trade-off가 커졌으므로 잠금셋 평가 전
탈락시켰다.

## temporal view ablation

v41 안의 WPT File expert를 multistream으로 일부 대체하고, 각 view 수에 대해
dev에서 weight를 먼저 고정했다.

| multistream view | dev 선택 weight | dev File EER | factorial holdout ADS | phone ADS | YuE ADS |
|---:|---:|---:|---:|---:|---:|
| v41, 추가 없음 | 0.00 | 0.17527 | 0.81927 | 0.90343 | 0.89673 |
| 1 | 0.25 | 0.16000 | 0.81636 | 0.90343 | 0.89673 |
| **2** | **0.20** | **0.15236** | **0.82309** | **0.90429** | **0.89673** |
| 3 | 0.20 | 0.15818 | 0.81927 | 0.90429 | 0.89673 |

1-view는 순차 혼합의 일부를 놓쳐 factorial holdout이 회귀했다. 시작과 끝을
보는 2-view는 3-view보다 계산이 적으면서 dev와 잠금축이 가장 안정적이었다.

## router 대 고정 MoE

별도로 이미 학습이 끝난 file-local telephone router의 확률만 사용해 다음
bounded soft router를 만들었다.

`multistream_weight = 0.10 + 0.60 * PHONE_PROB`

개발셋에서 선택된 3-view router는 File EER `0.16182`, 최악 channel EER
`0.17980`이고 모든 channel에서 v41 대비 비회귀였다. 그러나 고정 2-view
20% MoE의 전체 File EER `0.15236`보다 낮지 않았고, 잠금 전화셋에서도
회귀했다.

| 방식 | factorial holdout ADS | phone ADS | YuE ADS | 판정 |
|---|---:|---:|---:|---|
| v41 | 0.81927 | 0.90343 | 0.89673 | 기준 |
| phone soft router | **0.82400** | 0.90171 | 0.89673 | 전화 회귀로 탈락 |
| Music residual | 0.81584 | 0.90193 | 0.89673 | 두 축 회귀로 탈락 |
| **File-only fixed MoE** | 0.82309 | **0.90429** | 0.89673 | **채택** |

router가 전화 domain 자체는 잘 찾더라도, `전화인지`와 `어느 authenticity
expert가 맞는지`가 동일하지 않다. 작은 router 오차도 expert weight를 크게
움직인다. 반면 고정 MoE는 두 모델의 순위 증거를 항상 보존한다. 이번 실험은
router를 원칙적으로 배제한 것이 아니라, 동일 개발/잠금 기준에서 고정 MoE가
더 잘 일반화했기 때문에 선택한 결과다.

## Suno stress audit

학습에 넣지 않은 Suno vocals 양성-only bank에서 cold multistream의 File
판정은 3-view 기준 원본 `13/13`, codec stress `65/65`가 0.5를 넘었다.
최종 2-view에서는 원본 `13/13`, codec stress `64/65`였다. 이 수치는 EER이
아니고 양성 coverage이므로 실제 음원에 대한 false positive 성능을 뜻하지
않는다.

## 구현과 검증

- 학습/채점: `src/multistream_prompt_spectra.py`,
  `scripts/train_wpt_spectra_multitask.py`,
  `scripts/score_wpt_spectra_multitask.py`
- 제출 추론: `src/multistream_prompt_inference.py`
- 재현 평가: `scripts/evaluate_multistream_file_moe_v42.py`
- 제출 빌더: `scripts/build_multistream_file_moe_v42_submission.py`
- 관련 단위/회귀 테스트 15개 및 저장소 전체 테스트 123개 통과
- clean/전화/혼합 3파일 전체 CUDA entrypoint smoke 통과
- smoke에서 v41 대비 Voice, Music, 두 Presence는 bit-exact이고 File만 변경

최종 후보는 `multistream_file_moe_v42.zip`이다.

- 압축 크기: 8,314,124,177 bytes
- 압축 해제 크기: 9,219,407,470 bytes
- ZIP entry: 138개, 중복 0개, bytecode/cache 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 최대 단일 member: 2,387,980,808 bytes
- 전체 CRC: 오류 없음
- SHA-256: `0e1c30d0c6498e389f6d2ac42932a3664740e5451c765eb10da002cc193287f7`
