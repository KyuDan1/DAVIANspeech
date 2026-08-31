# 혼합 음성·음악 cross-component probing 결과

## 한 줄 결론

현재 문제는 `가짜 음성+진짜 음악(FR)`과 `진짜 음성+가짜 음악(RF)`을 구별할
정보가 encoder에 전혀 없는 것이 아니다. SPEAR 얕은 층에는 두 성분 정보가 남아
있지만, 지금 가진 학습 데이터로 붙인 선형·attention head는 새 도메인에서 충분히
일반화하지 못했다. 특히 기존 합성 평가셋은 실제 리더보드와 모델 순서를 거꾸로
평가하므로, 로컬 최고점을 그대로 제출에 반영하면 안 된다.

## 실제 점수로 확인한 병목

실제 twin anchor의 ADS는 `0.7083888889`다. 한 component 확률만 상수 0.5로 바꾼
probe 결과를 대회 식에 대입하면 anchor의 EER을 다음처럼 역산할 수 있다.

| 항목 | 역산 EER | ADS 손실 기여 |
| --- | ---: | ---: |
| File | 0.2741 | 0.1371 |
| Voice | 0.2156 | 0.0431 |
| Music | 0.3714 | 0.1114 |

Music EER이 가장 높고, 가중치가 0.5인 File EER이 총 손실에 가장 크게 기여한다.
따라서 CPS나 speech-only를 더 다듬는 것보다 RF/FR이 포함된 파일과 music fake
ranking을 개선하는 것이 우선이다. 다만 비공개 정답이 없어서 실제 점수만으로
RF와 FR 중 어느 쪽이 더 큰지까지 분리할 수는 없다.

## 평가 데이터와 누수 검사

`factorial_eval_1200_v2`는 다음 다섯 형태를 분리해 측정한다.

- voice-only 150개, music-only 150개
- 완전 동시 300개, 부분 중첩 300개, 순차 300개
- 혼합 형태마다 RR/RF/FR/FF가 동일 개수
- dev/holdout/locked 각 400개

혼합 head 학습셋 2,400개와 평가셋 사이에 정확히 같은 `VOICE_SOURCE_ID` 또는
`MUSIC_SOURCE_ID`는 0개다. 그러나 fake music 생성기 계열은 Echoes와 겹친다.
파일 누수는 없지만 generator-family 누수는 존재한다.

## 실험 1: SPEAR 13개 latent layer probing

SPEAR의 13개 층을 각각 1,280차원으로 평균 pooling하고, encoder는 고정한 채
voice/music 이진 head를 학습했다. 완전 동시·부분 중첩·순차의 여섯 component
EER을 함께 비교했다.

| 표현 | dev 평균 EER | dev 최악 EER |
| --- | ---: | ---: |
| layer 1 이진 head | **0.323** | 0.440 |
| layer 2 이진 head | 0.340 | **0.380** |
| layer 3 이진 head | 0.337 | **0.380** |
| deep layer 6–12 | 0.387–0.413 | 0.420–0.560 |

얕은 1–3층이 일관되게 낫고, 깊은 층은 오히려 artifact 구분 정보를 잃는다.
SPEAR를 쓴다면 마지막 층 하나보다 얕은 층의 국소 정보를 쓰는 것이 맞다.

RR/RF/FR/FF 공동 4분류 probe는 layer 2와 3에서 balanced accuracy `0.44`를
기록했다. 우연 수준은 `0.25`다. 즉 네 상태가 latent 공간에서 어느 정도 다르다.
layer 2의 대표 confusion matrix는 다음과 같다. 행이 정답, 열이 예측이며 순서는
RR/RF/FR/FF다.

```text
[[40, 18, 11,  6],
 [19, 33, 11, 12],
 [25, 12, 18, 20],
 [ 4, 17, 13, 41]]
```

RR과 FF는 비교적 쉽고 FR이 가장 어렵다. 공동 4분류 posterior를 다시 voice/music
확률로 주변화하면 layer 2 평균/최악 EER은 `0.323/0.380`으로, 독립 이진 head의
`0.340/0.380`보다 평균은 좋아졌다. 상호작용을 공동학습하는 방향은 타당하지만
그 자체로 충분한 성능은 아니다.

얕은 layer 1–3을 이어 붙인 128-unit nonlinear MLP도 학습했다. balanced accuracy
`0.413`, 평균/최악 EER `0.333/0.400`으로 layer 2 선형 joint head의
`0.440`, `0.323/0.380`보다 나빴다. 현재 데이터 규모에서는 더 큰 head보다 단순한
joint head가 일반화에 유리했다.

## 실험 2: layer attention과 시간 pooling

두 종류의 layer attention을 학습했다.

- component마다 전역 layer 가중치를 학습
- 샘플마다 동적으로 layer 가중치를 생성

전역 attention의 dev 평균/최악 EER은 `0.350/0.500`, 동적 attention은
`0.390/0.440`이었다. attention은 주로 얕은 층을 선택했지만 고정 layer 2/3보다
낫지 않았다. 복잡한 router가 부족한 데이터를 해결하지 못한 결과다.

오디오를 최대 세 개 시간 window로 나눠 기존 head 점수를 mean/max/min pooling한
실험도 수행했다. 완전 동시 voice EER은 `0.46`에서 변하지 않았고, 순차 voice에서
max pooling만 `0.32 → 0.30`의 작은 개선이 있었다. window 단위 평균 score만
고르는 것으로는 부족하며, 다음에는 latent token/patch 단위의 학습 가능한 query가
필요하다.

## 실험 3: speech XLS-R를 음악에 그대로 적용

`nii-yamagishilab/xls-r-2b-anti-deepfake`의 공개 speech head를 원본 음악에 그대로
적용했다.

| 조건 | Music EER |
| --- | ---: |
| music-only | 0.320 |
| 완전 동시 | 0.447 |
| 부분 중첩 | 0.480 |
| 순차 | 0.440 |

음악-only에서는 공통 생성 artifact를 일부 검출한다. 그러나 생성기별 EER이
MusicGen `0.02`에서 Producer `0.60`까지 크게 달라, 일반 artifact보다 특정 생성기
지문에 반응하는 면이 강하다. 혼합에서는 거의 chance이므로 speech head를 music
head로 그대로 재사용하는 방법은 채택하지 않는다.

같은 XLS-R embedding에 새 component linear head를 학습해도 dev 평균/최악 EER은
약 `0.35/0.44`였다. 기존 파이프라인과 결합한 결과도 dev/holdout 방향이 달라
채택하지 않았다.

## 실험 4: 기존 SPEAR 혼합 head 재사용

기존 SPEAR mixed-music head는 이 평가셋에서 music-only `0.28`, 완전 동시
`0.293`, 부분 중첩 `0.287`, 순차 `0.28`의 Music EER을 보였다. 현재 music score와
20% 결합하면 로컬 Score가 `0.7307 → 0.7641`로 올랐다.

그러나 이 개선은 제출 후보가 아니다. 정확한 역사적 코드 재실행 결과가 다음처럼
실제 리더보드와 반대였기 때문이다.

| 모델 | 로컬 ADS | 실제 ADS |
| --- | ---: | ---: |
| anchor | 0.6441 | 0.7075 |
| v10 fixed | 0.7001 | 0.6763 |

로컬에서는 v10이 `+0.0560`, 실제에서는 `-0.0312`다. Echoes 계열로 학습한
전문가가 Echoes 계열 평가 음악을 잘 맞춘 효과로 해석한다. 이 상태에서 20% 결합을
제출하면 과거 v6–v10의 회귀를 반복할 위험이 높다.

## 실험 5: Echoes 밖의 YuE native-stem audit

YuE 공식 모델로 seed 8개를 직접 생성했다. 각 출력은 12초이며, 사후 분리 모델을
쓰지 않고 YuE codec이 직접 생성한 instrumental/vocal 두 트랙을 사용했다. vocal
RMS가 충분한 seed는 6개였다. 이 데이터로 124개를 구성했다.

- primary 96개: 동시/부분/순차 RR/RF/FR/FF 각 8개
- music-only real/fake 각 8개
- native YuE vocal FR/FF 각 6개
- fake voice는 Qwen/F5/CosyVoice bank, fake music은 YuE instrumental

이 set은 모델 학습에는 넣지 않았다. 이번에 사전 후보 weight `0.05/0.10/0.20/0.30`을
한 번 비교했으므로 이제는 fresh locked가 아니라 `audit_yue`로 취급한다. Anchor
결과는 다음과 같다.

| 조건 | File EER | Voice EER | Music EER |
| --- | ---: | ---: | ---: |
| music-only | 0.125 | - | 0.125 |
| 완전 동시 | 0.375 | 0.250 | 0.500 |
| 부분 중첩 | 0.375 | 0.125 | 0.3125 |
| 순차 | 0.375 | 0.0625 | 0.4375 |

YuE 음악 자체는 잘 잡지만 다른 성분과 섞이면 music ranking이 무너진다. RR 대 FR
File EER은 동시/부분/순차 `0.375/0.250/0.250`, RR 대 RF는
`0.375/0.500/0.500`이다. 이 새 생성기 family에서는 FR보다 **RF가 더 어렵다**.
비공개 정답 없이 실제 test도 반드시 같다고 단정할 수는 없지만, RF 개선을 우선할
독립적인 근거가 생겼다.

공개 XLS-R speech head는 이 YuE primary에서 Voice/Music EER `0.250/0.521`,
music-only Music EER `0.50`이었다. 생성 음악의 평균 점수는 real보다 높았지만 파일별
순위가 불안정해 speech head 직접 재사용은 다시 실패했다.

반면 SPEAR layer-2 joint head는 primary File EER `0.208`, 기존 mixed-music head는
Music EER `0.104`였다. music-only에서는 두 music head 모두 EER `0.0`이었다. 즉
Echoes에서만 보인 개선이 아니라 처음 보는 YuE에도 방향이 유지됐다.

## 선택한 단순 후보: anchor + SPEAR 10%

Voice score와 CPS는 anchor를 그대로 유지하고 다음 두 열만 낮은 비중으로 결합한다.

```text
FILE_FAKE  = 0.90 × anchor_file  + 0.10 × SPEAR_joint(1 - P(RR))
MUSIC_FAKE = 0.90 × anchor_music + 0.10 × SPEAR_mixed_music
```

0.2 이상은 YuE/factorial에는 더 좋지만 competition_v3 Music EER을 악화시켰다.
일반성을 우선해 네 평가 bank의 ADS가 모두 개선되는 가장 단순한 `0.10`을 선택했다.

| 평가 bank | anchor ADS | 10% fusion ADS | 변화 |
| --- | ---: | ---: | ---: |
| competition_v2 | 0.80177 | 0.80784 | +0.00607 |
| competition_v3 | 0.92565 | 0.92797 | +0.00232 |
| factorial_v2 | 0.64413 | 0.65454 | +0.01041 |
| YuE primary | 0.66458 | 0.68681 | +0.02223 |

네 bank에서 모두 양의 방향이고, weight·router·threshold를 복잡하게 만들지 않는다.
따라서 현재 가장 타당한 다음 **단일 ablation 제출 후보**다. 다만 factorial과 YuE
결과를 이미 본 뒤 선택했으므로 실제 개선을 보장하지 않는다. 실제 제출에서는 이
한 변화만 anchor에 추가해 인과를 분리해야 한다.

## 분리 모델에 대한 판단

현재 결과는 “분리를 절대 쓰면 안 된다”까지 증명하지는 않는다. 그러나 다음은
확실하다.

- music stem detector는 과거 실험에서 원본 music detector보다 나빴다.
- 원본 SPEAR/XLS-R에도 혼합 상태 정보가 남아 있다.
- 순차 혼합은 원본 window만으로도 동시 혼합보다 훨씬 쉽다.

따라서 주 경로는 분리 없는 원본 patch encoder로 두고, HTDemucs/SAM 계열 stem은
독립적인 보조 증거로만 낮은 비중에서 검증하는 편이 안전하다. diffusion 여부보다
중요한 것은 분리 전후 fake ranking이 generator-disjoint 데이터에서도 보존되는지다.

## 다음 모델: 분리 없는 two-query component detector

다음 구현 우선순위는 다음 구조다.

```text
원본 오디오
  → SPEAR shallow token (layer 1–3, 시간축 유지)
  → voice query attention ─→ voice fake head
  → music query attention ─→ music fake head
  → joint RR/RF/FR/FF head ─→ component posterior 보정
  → file probability = soft OR(voice fake, music fake)
```

학습 loss는 voice BCE + music BCE + joint 4-class CE를 사용하고, RR/RF/FR/FF,
동시/부분/순차, SNR을 균형 sampling한다. encoder 전체 fine-tuning부터 시작하지 않고
얕은 layer와 query/head만 학습한다. 이 방식은 분리 artifact를 만들지 않으면서
시간적으로 순차인 경우와 실제 overlap을 모두 처리할 수 있다.

## 평가셋을 먼저 고쳐야 하는 이유

기존 locked 결과는 반복 실험 중 이미 확인했으므로 더 이상 진정한 locked set이
아니다. YuE family는 첫 새 audit tier로 만들었고 학습에는 사용하지 않았지만,
fusion weight 결과를 이미 확인했다. 또한 한 family 8곡뿐이므로 LeVo/HeartMuLa를
추가해 그중 하나를 열어보지 않은 최종 tier로 남겨야 한다.

- fake music: Echoes에 없는 YuE, LeVo, HeartMuLa 등
- real music: 학습에 쓰지 않은 FMA/MAESTRO/사용자 보유 음악을 원곡 그룹 단위 분리
- 각 generator를 train/dev/holdout에 파일 단위가 아니라 family 단위로 분리
- vocal song은 `VOICE_PRESENT=1, MUSIC_PRESENT=1`, instrumental만 music-only로 사용
- PANNs 단독으로 vocal 부재를 확정하지 않음

공식 구현 후보:

- YuE: <https://github.com/multimodal-art-projection/YuE>
- LeVo/SongGeneration: <https://github.com/tencent-ailab/SongGeneration>
- HeartMuLa: <https://heartmula.github.io/>

fresh generator-disjoint dev에서 학습·선택하고 holdout에서 같은 방향일 때만 한 후보를
locked에 통과시킨다. 실제 제출은 anchor에 한 변화만 더하는 방식으로 검증한다.

## 재현 파일

- SPEAR layer/joint probe: `scripts/probe_spear_latents.py`
- SPEAR window 보존 추출: `scripts/extract_spear_embeddings.py --preserve-windows`
- layer 결과: `reports/factorial_v2_probes/component_layers_dev.csv`
- joint 결과: `reports/factorial_v2_probes/four_class_layers_dev.csv`
- exact anchor: `reports/factorial_v2_anchor/diagnostic.csv`
- exact v10: `reports/factorial_v2_v10_exact/diagnostic.csv`
- YuE audit: `reports/yue_audit_anchor/factorial_contrasts.csv`
- fusion grid: `reports/cross_component_fusion/fusion_grid.csv`
