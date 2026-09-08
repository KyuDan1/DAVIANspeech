# Separation-free Music-only expert v51

2026-09-04 기준. 이 문서는 v50의 가장 큰 잔여 병목인
`MUSIC_FAKE_PROB`만 개선하기 위해 수행한 비교 실험과 동결 결정을 기록한다.
결론은 **원본 혼합 오디오의 기존 EAT+SPEAR feature를 재사용하는 Music 전용 3-head
ensemble을 v50 Music logit에 3.75%만 더하는 것**이다. 분리 모델이나 stem은 사용하지
않고, File/Voice/CPS 네 출력은 byte 수준으로 보존한다.

## 동결 결과

| 평가군 | v50 Music EER | v51 Music EER | 변화 | v50 ADS | v51 ADS |
|---|---:|---:|---:|---:|---:|
| codec mixed dev v4, 600 | 0.27333 | **0.25333** | -0.02000 | 0.76000 | **0.76600** |
| codec mixed blind v5, 240, retrospective | 0.19167 | **0.18333** | -0.00833 | 0.81083 | **0.81333** |

두 평가군 모두 전체 Music EER가 좋아졌고, 채널별 최악값도 v4
`0.41667 → 0.38333`, v5 `0.20833 → 0.20833`으로 악화되지 않았다. 동결 뒤에 v6
Music prediction은 만들지 않았고, v6 결과를 member나 가중치 선택에 사용하지 않았다.
다음 unopened blind v7에서 단 한 번만 검증한다.

별도로 개발에 쓰던 source-disjoint 7개 bank에서도 전부 개선됐다.

| 개발 bank | v50 Music EER | v51 Music EER |
|---|---:|---:|
| external mixed | 0.12500 | **0.11000** |
| channel factorial dev | 0.21143 | **0.20571** |
| mixfake music dev | 0.07625 | **0.06500** |
| source-disjoint mixed equal | 0.14000 | **0.11000** |
| source-disjoint mixed | 0.16000 | **0.12000** |
| source-disjoint music | 0.05000 | **0.03000** |
| telephone mixed dev | 0.18667 | **0.17000** |

## 방법

v47/v50이 원본 오디오에서 이미 계산하는 두 feature만 사용한다.

- EAT patch graph: 시간 노드와 주파수 노드를 별도로 유지한다.
- SPEAR component bins: 여러 SSL layer의 구간별 통계를 유지한다.
- Component-query MHFA: Music query가 두 feature에서 Music 진위에 유용한 부분을 직접
  읽는다. waveform source separation은 없다.
- 학습 loss는 Music authenticity 하나뿐이다: task weight `[0, 1, 0]`.
- seed 20260911은 일반 head, 20260913/20260914는 같은 원본의 5개 codec 변형에서
  Music logit 분산을 각각 `0.01`, `0.05`로 억제한 head다.
- 세 확률을 logit 평균한 뒤 v50 Music logit과 `96.25:3.75`로 섞는다.

식은 다음 하나다.

```text
expert = sigmoid(mean(logit(head_11), logit(head_13), logit(head_14)))
Music  = sigmoid(0.9625 * logit(v50_Music) + 0.0375 * logit(expert))
```

router는 넣지 않았다. v4와 v5에서 어느 codec이 어려운지가 일치하지 않아 작은
표본으로 학습한 phone router는 domain shortcut이 될 가능성이 더 컸다. 고정식은 각
파일만 독립적으로 처리하고, test-set 통계나 다른 파일의 결과를 보지 않는다.

## 학습과 누수 통제

학습에는 6개 train role, 총 15,800개만 사용했다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `channel_invariant_factorial_train_v1`

early stopping은 7개 development role의 Music EER로만 했다. v4/v5/v6와 사용자가 준
13개 Suno vocal 곡은 학습에 넣지 않았다. v4의 등록된 전체 role과의
`VOICE_SOURCE_ID`, `MUSIC_SOURCE_ID`, canonical song 비교는 모두 교집합 0이고, v5도
locked validation의 source/speaker/song 교집합이 0이다. codec invariant loss는 정확히
같은 `MIXTURE_ID`의 clean/G.711/G.722/Opus/transcode 다섯 변형만 묶으므로 서로 다른
곡이나 발화를 억지로 같은 label representation으로 만들지 않는다.

가짜 음악 generator는 학습에서 균형 sampling했지만, generator family 자체가 완전히
미관측인 LOGO-CV라고 주장하지 않는다. 기존 leave-one-generator-out 실험에서 Suno는
비교적 쉬웠지만 Udio/Mubert는 크게 무너졌고, 그 때문에 generator 이름이나 장르
평균을 이용한 head 대신 작은 원본-audio residual만 채택했다.

## 채널과 RF/FF 세부 결과

`RF`는 진짜 음성+가짜 음악, `FF`는 가짜 음성+가짜 음악이다. Music EER만 표시한다.

| bank | channel | v50 | v51 |
|---|---|---:|---:|
| v4 | clean | 0.10000 | **0.06667** |
| v4 | G.711 | 0.25000 | **0.23333** |
| v4 | G.722 | 0.10000 | 0.10000 |
| v4 | Opus-NB | **0.35000** | 0.36667 |
| v4 | G.711→Opus | 0.41667 | **0.38333** |
| v5 | clean | 0.16667 | 0.16667 |
| v5 | G.711 | 0.20833 | 0.20833 |
| v5 | G.722 | 0.16667 | **0.12500** |
| v5 | Opus-NB | 0.20833 | 0.20833 |
| v5 | G.711→Opus | **0.16667** | 0.20833 |

셀별로는 일관된 전승이 아니다.

| bank/layout | RF v50→v51 | FF v50→v51 |
|---|---:|---:|
| v4 concurrent | 0.24→0.28 | 0.24→0.20 |
| v4 partial overlap | 0.36→0.32 | 0.36→0.38 |
| v4 sequential | 0.14→0.10 | 0.28→0.28 |
| v5 concurrent | 0.15→0.20 | 0.20→0.25 |
| v5 partial overlap | 0.10→0.10 | 0.25→0.20 |
| v5 sequential | 0.20→0.20 | 0.25→0.25 |

따라서 “모든 RF를 해결했다”는 결과가 아니다. 다만 서로 다른 source와 작은 v5에서
전체 EER 및 worst-channel 제약을 동시에 통과한 가장 보수적인 Pareto 후보다. 두
invariant head만 7.5% 넣으면 v4는 0.22667까지 내려가지만 v5 transcode가
`0.16667 → 0.25000`으로 악화되어 버렸다.

## 비교한 대안

아래는 각 expert의 standalone Music EER다. v38 unified와 long-horizon의 작은 residual은
v4를 개선하기도 했지만 v5에서 반대로 움직여 최종 후보에서 제외했다.

| original-audio 후보 | v4 | v5 | 판단 |
|---|---:|---:|---|
| v50 Music anchor | **0.27333** | **0.19167** | 기준 |
| v38 unified dual SSL | 0.29000 | 0.28333 | v5 실패 |
| EAT/SPEAR long-horizon | 0.27667 | 0.35000 | v5 실패 |
| WPT/Spectra seed06 | 0.37667 | 0.39167 | Music에는 부적합 |
| Music-selected WPT/Spectra seed07 | 0.32333 | 0.42500 | Music에는 부적합 |
| SOFIA/MERT | 0.38333 | 0.42500 | 추가 가중치 불필요 |
| 기존 component-query | 0.29667 | 0.25833 | File/Music multi-task 한계 |
| 기존 diverse three-stream | 0.26000 | 0.23333 | v5 회귀 |

MERT는 “곡 구조”를 잘 표현하지만 생성기/장르 shortcut도 강했고, WPT는 v50의 File
분류에는 유효하지만 Music component 진위 ranking에는 맞지 않았다. 이번 결과에서 가장
효과가 있었던 차이는 새 backbone이 아니라 **기존 혼합 feature에 Music-only objective와
동일-mixture codec consistency를 적용한 것**이다.

## Suno 진단

Suno vocal 13곡은 학습/선택에 사용하지 않았다. 새 expert 단독은 13곡 중 8곡만 0.5를
넘으므로 Suno 전용 detector로는 불충분하다. 하지만 3.75% residual은 기존 v14 참고
출력의 강한 Music score를 뒤집지 않았고 13/13이 fake로 유지됐다(최소 0.96671). 이
수치는 v50 exact 재평가가 아니라 과거 v14 output에 대한 보존성 진단이므로 EER 근거로
사용하지 않는다.

## File-consistency branch와의 조합 예상

이 branch는 Music만 바꾸므로, 별도로 동결된 final-component File residual을 그대로
뒤에 적용할 수 있다. 두 효과가 해당 bank에서 그대로 합쳐진다고 산술 계산하면:

- v4: File/Voice/Music EER `0.22000 / 0.20667 / 0.25333`, ADS 약 **0.77267**
- v5: File/Voice/Music EER `0.20278 / 0.11667 / 0.18333`, ADS 약 **0.82028**

이는 로컬 조합 예상치이며 공식 리더보드 점수나 unopened blind 결과를 보장하지 않는다.

## 구현과 재현

- 동결 설정: `configs/music_only_v51_frozen.yaml`
- inference: `src/music_only_residual.py`
- training: `scripts/train_component_query_mhfa.py`
- package builder: `scripts/build_music_only_v51_submission.py`
- built directory: `music_only_v51_frozen/`
- audits: `reports/music_expert_v51/`
- tests: `tests/test_music_only_residual.py`,
  `tests/test_build_music_only_v51_submission.py`

Focused test는 `8 passed`이고, 생성된 `script.py`와 residual module은 compile을 통과했다.
빌더는 v50 script와 세 checkpoint의 SHA-256을 고정하고, 기존 package를 hardlink한 뒤
수정할 파일만 분리 복사한다. 추가 SSL forward는 없고 627k-parameter head 세 개만 기존
cache에 실행한다. ZIP 생성, 커밋, 제출은 하지 않았다.
