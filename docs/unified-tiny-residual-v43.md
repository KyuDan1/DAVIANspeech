# v43: exact-v18 기반 unified dual-SSL 초소형 residual

## 결론

2026-09-04 현재 팀의 실제 최고 제출은 `channel_invariant_moe_v18.zip`이며
총점 `0.7616379312`, ADS `0.7363412698`, CPS `0.9893078836`이다. 현재 리더보드
1위는 총점 `0.837980`, ADS `0.821160`, CPS `0.98932`다. CPS가 사실상 같으므로
핵심 격차는 ADS `0.084819`다.

최근 실제 제출은 로컬 평가에서 크게 좋아 보인 aggressive router/MoE가 hidden
test에서 역전됨을 보여 주었다. 따라서 v43은 SOTA를 한 번에 달성하는 최종 모델이
아니라, 새 unified dual-SSL expert가 hidden test와 정렬되는지를 측정하는
`exact v18 + 단일 bounded residual` 후보로 설계했다.

- source separation 없이 원본 오디오의 EAT 계층 통계와 SPEAR 시간-bin 통계를 본다.
- 개발군으로 고른 독립 seed 3개를 logit 평균한다.
- v18의 File과 Music logit에 각각 5%만 결합한다.
- Voice fake와 두 Presence 출력은 v18과 byte-for-byte 동일하다.
- 기존 EAT/SPEAR backbone pass를 재사용하므로 큰 모델 호출을 추가하지 않는다.

## 1. 실제 리더보드 ablation

아래 점수는 로컬 추정이 아니라 DACON 실제 채점 결과다.

| 제출 | 핵심 변경 | 총점 | ADS | CPS | v18 대비 ADS |
|---|---|---:|---:|---:|---:|
| v18 | channel/component-invariant head 5% | **0.761638** | **0.736341** | 0.989308 | 0 |
| v19 | invariant 3-seed MoE 10% | 0.759724 | 0.734214 | 0.989308 | -0.002127 |
| v32 | File temporal attention 20%, phone 25% | 0.750002 | 0.723413 | 0.989308 | -0.012929 |
| v33 | v32 + Music temporal residual 50% | 0.742288 | 0.714841 | 0.989308 | -0.021500 |
| inv4-w30 | paired invariant 4-head 30% | 0.750509 | 0.723976 | 0.989308 | -0.012365 |

해석은 명확하다.

1. 현재 hidden test에서는 5% 정도의 작은 독립 보정만 일반화했다.
2. 로컬 factorial/phone/YuE 평균을 최대화한 큰 가중치는 실제 EER 순서를 망가뜨렸다.
3. phone/content hard routing이나 특정 조건의 높은 가중치는 test 구성 비율과
   generator에 과적합되었다.
4. CPS 변화는 약 `0.00015` 이내라 최근 총점 차이는 거의 전부 ADS에서 발생했다.
5. 따라서 다음 제출은 exact-v18에서 한 축과 한 모델만 바꾸어 원인을 식별해야 한다.

## 2. 최근 연구에서 가져온 구조적 원칙

AT-ADD 2026 상위 시스템은 크게 두 방향이었다.

- 1위 시스템은 BEATs audio-type router 뒤에 XLS-R speech expert와 EAT-AASIST
  sound/music 및 singing expert를 두었다.
- 2위 시스템은 EAT와 XLS-R의 여러 layer token을 하나의 unified core에서 결합하고,
  speech expert를 보수적인 refiner로만 사용했다.

우리 대회는 speech/music이 한 파일에 동시에 또는 순차적으로 존재할 수 있으므로
파일 전체를 한 expert로 보내는 hard router는 직접 적용하기 어렵다. 채택한 원칙은
`원본 mixture를 항상 보는 unified core + 낮은 비중의 component specialist`다.
Demucs/SAM-Audio stem은 보조 관측으로만 쓸 수 있으며, 생성 artifact 판단의 유일한
입력으로 쓰지 않는다.

관련 자료:

- <https://arxiv.org/html/2608.00493>
- <https://arxiv.org/html/2608.29021>
- <https://arxiv.org/html/2608.23437>

## 3. v43 expert 학습과 누수 방지

unified head는 EAT의 최대 3개 원본 view별 12-layer 통계와 SPEAR의 최대 3개
view/8개 시간-bin 통계를 함께 처리한다. Voice/Music/File를 다중 과제로 학습하지만
v43에서는 잠금 감사에서 상대적으로 안정적이었던 File과 Music만 사용한다.

학습 partition은 다음 7개다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2`
- `channel_invariant_factorial_train_v1`

모델/seed 선택은 별도의 source/generator-disjoint 개발군 7개에서 했다. factorial
holdout, phone factorial, YuE, Suno는 학습·early stopping·seed 선택에 쓰지 않고
선택이 끝난 뒤 한 번만 감사했다.

seed 0~3의 모든 15개 비공집합 ensemble을 비교한 결과 seed `1+2+3`이 선택됐다.

| 선택 통계 | 값 |
|---|---:|
| 개발군 평균 ADS | 0.833096 |
| 개발군 최악 ADS | 0.780909 |
| 평균 Music EER | 0.118648 |
| 최악 Music EER | 0.194286 |

결합 비중은 `[0, 0.025, 0.05]`만 탐색했고 factorial dev의 모든 channel에서 v18보다
나빠지지 않는 조건을 걸었다. 선택 결과는 File 5%, Music 5%다.

## 4. 잠금 평가 결과

| 잠금 평가 | v18 ADS | v43 ADS | 변화 |
|---|---:|---:|---:|
| factorial holdout | 0.745506 | 0.752753 | +0.007247 |
| phone factorial | 0.733857 | 0.785750 | +0.051893 |
| YuE cross-component | 0.828448 | 0.833287 | +0.004839 |

Suno 보컬 음악 13개는 unified 4-seed 감사에서 File과 Music 모두 13/13이 0.5를
넘었다. 그러나 Suno는 fake-only 소수 감사군이므로 임계값 통과를 일반 EER 성능으로
해석하지 않는다.

중요하게도 과거에는 이와 유사한 로컬 상승이 hidden test에서 역전됐다. 위 결과는
v43을 제출할 근거이지, 팀 ADS가 곧바로 `+0.007~0.052` 오른다는 예측은 아니다.
v43의 실제 역할은 새 unified expert의 hidden 정렬 여부를 한 번의 작은 변경으로
검증하는 것이다.

## 5. 배포 검증

- 후보: `unified_tiny_residual_v43.zip`
- 압축 크기: 7,045,536,715 bytes
- 압축 해제 크기: 7,913,199,423 bytes
- entry: 115개, 중복 0개, unsafe path 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 전체 ZIP CRC: 오류 없음
- SHA-256: `3ac297dad3b47111b34a6bb69de771a8a1f06e8db704654dab0c82297a5b86cf`
- requirements: `onnxruntime-gpu==1.23.2`
- CUDA 2파일 full smoke: 성공
- v18 대비 Voice fake와 두 Presence 열의 최대 절대 차이: 정확히 0

## 6. SOTA까지 남은 일

v18에서 1위까지 필요한 ADS 상승은 `0.084819`다. 5% residual만으로 이 격차를
모두 메우기는 어렵다. v43 결과와 별개로 다음 핵심 실험이 필요하다.

1. 원본 mixture patch/token을 직접 처리하는 EAT-AASIST 음악·혼합 specialist를
   자체 학습한다. 공개 weight는 라이선스가 명시된 경우만 사용한다.
2. clean/telephone codec을 같은 원천의 paired view로 학습하여 채널 shortcut을
   제거한다.
3. fake-voice+real-music, real-voice+fake-music, RR, FF와 speech-only/music-only,
   sequential/overlap을 각각 독립 표로 평가한다.
4. 모델 선택은 평균이 아니라 generator/source/channel별 최악 EER를 함께 사용한다.
5. leaderboard에는 exact-v18 대비 한 출력축·한 변경만 올려 hidden 전이율을
   측정한다.

SOTA 후보의 조건은 로컬 평균 상승이 아니라, 최소한 세 잠금 평가의 모든 핵심 셀과
전화 변형에서 동시 개선하고 hidden probe에서도 방향이 일치하는 것이다.
