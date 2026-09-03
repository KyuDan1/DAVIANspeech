# v38: separation-free unified dual-SSL Music residual

## 결론

v38은 v37을 앵커로 유지하고, 원본 혼합 오디오에서 얻은 EAT 계층 통계와 SPEAR
시간-bin 통계를 함께 처리하는 새 전문가를 `MUSIC_FAKE_PROB`에만 20% 결합한다.
결합은 확률 평균이 아니라 logit 공간의 bounded soft MoE다.

외부 hard router와 content/phone router도 비교했지만 채택하지 않았다. 라우터는
전체 평균에서 작은 이득이 있었으나, 생성기와 동시 중첩 조건별 우열이 바뀌었다.
미지 test set에 대한 일반성과 실패 비용을 우선하면 고정 20%가 더 안전했다.

## 1. 새 전문가

### 입력

- EAT: 원본 오디오의 최대 3개 view, 12개 transformer layer, 5개 통계,
  128차원 고정 Gaussian projection
- SPEAR: 같은 원본 오디오의 최대 3개 view, view당 8개 시간-bin, shallow 4개
  layer, 4개 통계, 64차원 고정 Gaussian projection
- source separation을 거치지 않은 특징만 사용한다.

EAT은 저수준 texture와 전역 CLS 정보를 함께 제공하고, SPEAR는 음성·음악이
동시 혹은 순차적으로 나타나는 국소 시간 구조를 보존한다. 각 task가 SSL layer를
따로 선택한 뒤 EAT/SPEAR token을 합쳐 한 층의 self-attention을 통과시킨다.
Voice/Music/File, presence, RR/RF/FR/FF를 함께 학습했지만, 잠금 평가에서 일관되게
강했던 Music 출력만 배포에 사용한다.

### 학습과 누수 방지

학습에는 다음 7개 bank만 사용했다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2`
- `channel_invariant_factorial_train_v1`

early stopping은 generator/source가 분리된 개발 bank 7개로 했다. factorial holdout,
phone factorial, YuE, Suno audit는 학습과 checkpoint 선택에 사용하지 않았다.
seed 2개를 독립 학습하고 logit ensemble했다.

## 2. standalone 결과

| 잠금 평가 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| factorial holdout | 0.2647 | 0.3600 | **0.1600** | 0.7476 |
| phone factorial | 0.1517 | 0.2500 | **0.0925** | 0.8464 |
| YuE cross-component | 0.2126 | 0.3979 | **0.0968** | 0.7851 |

v37의 Music EER는 각각 0.2171, 0.1075, 0.1613이다. 새 모델은 세 환경의
Music 축에서 모두 우수하지만 Voice와 일부 File 축은 약하다. 따라서 전체 모델을
고르는 router가 아니라 Music 전용 residual expert로 제한했다. Suno 보컬 음악은
standalone File과 Music 모두 13/13을 0.5 이상으로 판정했다.

## 3. router 대 soft MoE

평가 후보는 다음과 같다.

1. 고정 20% soft MoE
2. phone probability에 따른 hard/soft weight router
3. Voice/Music presence가 모두 0.7 이상이면 expert 비중을 높이는 content router
4. 두 모델 중 절대 logit이 더 큰 쪽을 택하는 confidence hard router

Voice/File/CPS는 모두 고정하고 Music 점수만 바꿔 같은 행에서 비교했다.

| 방법 | factorial ADS | phone ADS | YuE ADS | 최소 v37 대비 개선 |
|---|---:|---:|---:|---:|
| v37 | 0.75792 | 0.86046 | 0.85681 | - |
| 고정 soft MoE 0.20 | **0.76821** | **0.87021** | **0.86649** | **+0.00968** |
| content router 0.30/0.50 | 0.76992 | 0.87021 | 0.86649 | +0.00968 |
| confidence hard router | 0.77335 | 0.86796 | 0.87616 | +0.00750 |

content router가 평균으로는 근소하게 높지만 고정 MoE보다 phone에서 이득이 없고,
Mubert와 SongGen 같은 generator slice에서 더 나빴다. confidence hard router는
factorial concurrent EER를 0.18에서 0.24로 악화시키며 가장 공격적이었다.
audio layout만 완벽히 알아도 expert의 우열은 phone/channel/generator에 따라 다시
바뀌므로, layout 분류 정확도가 곧 올바른 expert 선택을 뜻하지 않는다.

500회 label-stratified paired bootstrap에서 고정 20%의 ADS 개선 확률은 factorial
98.6%, phone 100%, YuE 89%였다. YuE는 Music-present 124개뿐이라 95% 구간에 0을
포함한다. 이 불확실성도 hard routing 대신 bounded residual을 택한 이유다.

## 4. 배포 방식

v38은 기존 backbone pass를 추가하지 않는다.

- EAT presence 단계에서 이미 생성하는 계층 통계를 재사용한다.
- SPEAR dual-domain 통계 pass에서 시간-bin을 동시에 계산한다.
- 추가되는 것은 약 1.8MB head 두 개와 작은 transformer head 추론뿐이다.
- 마지막 단계에서 Music score만 `0.8 * logit(v37) + 0.2 * logit(expert)`로
  결합한다.

동일한 clean/Opus narrow-band/mixed 3파일을 v37과 v38 전체 CUDA 파이프라인에
통과시킨 결과 File, Voice, Voice Presence, Music Presence의 최대 절대 차이는
정확히 0.0이었고 Music만 변경됐다. 전체 단위/회귀 테스트는 104개 통과했다.

최종 제출 후보는 `v38_unified.zip`이다.

- 압축 크기: 7,064,630,849 bytes
- 압축 해제 크기: 7,921,966,872 bytes
- ZIP entry: 124개, 중복 0개, bytecode/cache 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 최대 단일 member: 2,387,980,808 bytes
- 전체 CRC: 오류 없음
- SHA-256: `97f16d0b3ee4682199af51fc60a742f35066b1f21d638931bf551c1455c643a7`

## 5. 해석과 다음 검증

로컬 ADS 증가는 세 잠금 bank에서 약 `+0.0097~+0.0103`이다. hidden set의 Music
분포가 같다는 보장은 없으므로 이를 leaderboard 상승 예상치로 그대로 사용하면
안 된다. 실제 제출에서는 v37이 아직 leaderboard에 올라가지 않은 상태라,
v38 점수만으로 v37 기여와 새 dual-SSL 기여를 완전히 분리할 수 없다. 제출 슬롯이
허용되면 Music-only constant probe와 v37 anchor를 함께 사용해 hidden Music EER
기여를 다시 식별하는 것이 가장 정보량이 높다.
