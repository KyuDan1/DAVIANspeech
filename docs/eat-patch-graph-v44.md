# v44: 분리 없는 EAT patch-graph 음악·혼합 탐지기

## 결론

2026-09-04 현재 리더보드 1위는 총점 `0.837980`, ADS `0.821160`, CPS
`0.98932`다. 우리 팀 최고 v18은 총점 `0.7616379312`, ADS `0.7363412698`,
CPS `0.9893078836`이므로 CPS는 사실상 같고 ADS를 `0.084819` 올려야 한다.

v44는 이 격차의 가장 큰 원인으로 추정되는 음악 및 혼합 파일을 위해 만든
separation-free specialist다. 기존 EAT 특징을 한 파일당 평균·표준편차로 모두
축약하지 않고, 원본 mixture의 시간×주파수 patch 배열을 보존해 학습한다. exact-v18
출력 중 File과 Music logit에만 5%를 더하며 Voice와 두 Presence 출력은 건드리지
않는다.

잠금 평가 세 종류에서 v18보다 모두 좋아졌고, 실제 제출 코드도 13개 Suno 곡을
CUDA로 끝까지 처리했다. 그러나 이는 아직 공식 SOTA 달성이 아니라 새로운 expert가
hidden test와 정렬되는지를 보는 보수적인 후보이다.

## 1. 왜 새 모델이 필요한가

기존 EAT 보조 head는 6초 view의 token을 통계량으로 축약했다. 이 방식은 음성·음악
존재 여부에는 강하지만, 생성기의 국소적인 시간/주파수 불연속과 반복 구조가
사라질 수 있다. 반대로 SAM-Audio나 Demucs로 먼저 분리하면 분리 모델이 새로운
artifact를 만들거나 원래 생성 흔적을 지울 수 있다.

v44는 다음 원칙을 사용한다.

1. 항상 원본 mixture를 보고 source separation을 하지 않는다.
2. EAT layer `1, 3, 5, 7, 9, 11`의 patch token을 유지한다.
3. 각 layer에서 주파수축을 요약한 38개 시간 node와 시간축을 요약한 8개 주파수
   node를 만든다.
4. label과 무관한 고정 Gaussian projection으로 768차원을 128차원으로 줄인다.
5. task-conditioned 2-layer graph Transformer가 Voice/Music/File 순위를 각각
   학습한다.
6. 기존 presence용 계층 통계와 patch 특징을 한 번의 EAT forward에서 함께 뽑아
   추론비용을 제한한다.

최근 AT-ADD 계열 연구의 EAT-AASIST sound/music specialist와 unified mixture
model 원칙을 가져오되, 이 대회의 동시·순차 혼합 조건 때문에 파일 전체를 하나의
expert로 보내는 hard audio-type router는 사용하지 않았다.

- <https://arxiv.org/html/2608.00493>
- <https://arxiv.org/html/2608.29021>
- <https://arxiv.org/html/2608.23437>

## 2. 학습 데이터와 누수 방지

학습에는 아래 7개 train partition만 사용했다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2`
- `channel_invariant_factorial_train_v1`

각 corpus가 같은 총 sampling mass를 갖게 한 뒤, corpus 내부에서
real/fake/presence 조합과 가능한 generator/source identity를 균형화했다. 학습
목표 비중은 Voice `0.05`, Music `0.50`, File `0.45`다. view dropout과 특징 moment
randomization을 적용했고, 모델 선택은 별도 7개 dev bank의 File/Music 평균과 최악
성능의 결합으로 했다.

`factorial_eval_1200_v2_holdout`, `phone_factorial_1200_v1`,
`yue_cross_component_audit_v1`, Suno 13곡은 학습·epoch 선택에 사용하지 않고
checkpoint가 고정된 뒤 한 번만 열었다.

선택된 seed의 개발 결과는 다음과 같다.

| 통계 | 값 |
|---|---:|
| best epoch | 56 |
| File/Music robust selection | 0.820471 |
| 7개 dev 평균 ADS | 0.811313 |
| 7개 dev 최악 ADS | 0.761558 |
| 학습 샘플 | 18,360 |
| 학습 파라미터 | 327,445 |

## 3. 잠금 평가

Voice EER은 exact-v18 그대로이고 File/Music logit에 patch expert를 5% 결합했다.

| 잠금 평가 | v18 ADS | v44 ADS | 변화 |
|---|---:|---:|---:|
| factorial holdout | 0.745506 | **0.757896** | **+0.012390** |
| phone factorial | 0.733857 | **0.790929** | **+0.057072** |
| YuE cross-component | 0.828448 | **0.849252** | **+0.020804** |

특히 phone에서 Music EER은 `0.3450 → 0.2450`, File EER은
`0.2583 → 0.2041`로 내려갔다. 분리 없이 원본 mixture의 patch를 보는 방법이 전화
변형에서도 유효하다는 강한 로컬 증거다. Suno 13곡에 대한 완성된 v44 제출
파이프라인은 File/Music fake 확률 모두 13/13에서 0.5를 넘었다.

추가로 같은 설정의 독립 seed 두 개를 학습했다. 세 seed 앙상블은 개발 선택 점수를
`0.820471 → 0.828233`으로 올렸지만 잠금 factorial/phone에서는 단일 seed보다
낮았다. 따라서 seed 수를 늘렸다는 이유만으로 배포하지 않고 v44는 처음 고정한 단일
seed를 유지했다. 이는 평균적인 dev 개선보다 generator/channel 미지 일반성이 더
중요하다는 결과다.

## 4. 배포 검증

- 제출 후보: `eat_patch_graph_v44.zip`
- 압축 크기: 7,909,771,846 bytes
- 압축 해제 크기: 7,909,749,310 bytes
- entry: 112개, 중복 0개, unsafe path 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- requirements: `onnxruntime-gpu==1.23.2`
- SHA-256: `5b0650185f4d714f0758b1faedbee5be5bb65be6b3c318eaba5fbc829861abfd`
- 전체 ZIP CRC: 오류 없음
- 전체 회귀 테스트: 129 passed
- 실제 제출 코드 CUDA smoke: Suno 13파일 성공, 임시 특징 파일 정상 삭제

압축은 저장 모드여서 압축 파일이 해제 크기보다 약간 크지만 두 제한인 10GB와
32GB를 모두 만족한다. v18에서 이미 평가 서버 실행이 확인된
`onnxruntime-gpu==1.23.2`만 설치한다.

## 5. 해석과 다음 실험

v44의 5% 결합은 세 잠금 평가에서 일관되지만 SOTA까지 필요한 ADS `+0.084819`를
한 번에 보장하지 않는다. 로컬에서는 10~20% 결합도 더 좋아졌으나, v19/v32/v33의
실제 제출 결과는 큰 residual이 hidden test에서 쉽게 역전된다는 것을 이미
보여줬다. 따라서 우선 v44로 새 expert의 hidden 방향을 측정해야 한다.

다음 우선순위는 다음과 같다.

1. v44가 hidden에서 상승하면 Music만 10%, 15%로 한 단계씩 올려 dose-response를
   측정한다. File과 Voice를 동시에 바꾸지 않는다.
2. seed ensemble 대신 generator/source/channel GroupDRO와 paired clean/codec
   consistency로 단일 모델의 최악 조건을 직접 개선한다.
3. 실제 probe에서 Music EER 기여를 다시 분리해 patch expert가 고친 축이 숨은
   문제와 일치하는지 확인한다.
4. symbolic composition처럼 국소 artifact가 약한 AI 음악은 장기 rhythm/harmony
   stream으로 보완하되, 원본 mixture specialist의 낮은 비중 residual로만 추가한다.

재현 명령은 `scripts/train_eat_patch_graph.py`,
`scripts/score_eat_patch_graph.py`, `scripts/evaluate_eat_patch_graph_v44.py`,
`scripts/build_eat_patch_graph_v44_submission.py`에 남겼다.

## 6. 후속 robust-loss ablation

v44 checkpoint를 고정한 뒤 다음 학습 변경을 같은 seed와 같은 7개 dev bank에서
비교했다. 기준 선택 점수 `0.820471`을 넘는 설정은 없었으므로 잠금 평가와 배포에서
모두 제외했다.

| 변경 | 설정 | dev robust selection | 판정 |
|---|---:|---:|---|
| corpus GroupDRO | step 0.01 | 0.803854 | 제외 |
| hard-pair rank | worst 25% pair | 0.800412 | 제외 |
| hard-pair rank | worst 50% pair | 0.805536 | 제외 |
| paired channel consistency | weight 0.05 | 0.793523 | 제외 |
| paired channel consistency | weight 0.10 | 0.800440 | 제외 |

단순히 어려운 corpus나 channel pair의 손실을 키우면 source-disjoint mixed 조건이
불안정해졌다. 이 결과 때문에 v44에는 학습 objective 변경을 넣지 않았고, 다음
학습은 약한 정규화 추가가 아니라 generator 자체를 분리한 validation 및
component-cell balanced batch 구성이 필요하다.
