# v36: 계층형 EAT 음악 전문가와 전화 조건부 soft MoE

## 결론

전문가 하나를 고르는 hard router보다, 실제로 서로 다른 단서를 학습한 모델을
낮은 비중으로 합치는 soft MoE가 더 안정적이었다. 다만 전화 채널은 clean
오디오와 오류 양상이 충분히 달랐기 때문에, 기존 전화 품질 router를 이용해
**전문가를 교체하지 않고 결합 비중만 높이는 방식**은 유지했다.

최종 v36b 구조는 다음과 같다.

1. 실제 리더보드 최고점 `channel_invariant_moe_v18.zip`을 anchor로 고정한다.
2. 원본 mixture에서 EAT의 12개 중간 layer를 한 번에 읽는 음악 전문 head 두
   seed를 logit 평균한다.
3. 일반 파일은 새 음악 전문가를 Music 20%, File 10%로 결합한다.
4. 전화 품질 파일은 같은 전문가의 비중만 Music/File 각각 30%로 높인다.
5. Voice fake와 두 Presence 출력은 v18 그대로 유지한다.
6. Demucs나 diffusion separator의 출력은 새 음악 전문가에 사용하지 않는다.

로컬 잠금 평가에서 factorial+phone 합산 ADS는 `0.7339` 수준의 v18 anchor에서
`0.8304`로 상승했다. 이는 hidden leaderboard 점수의 예측값이 아니라, 제출 전
가설 검증 결과다. 과거 로컬 개선이 hidden EER 순위를 바꾸지 못한 사례가
있으므로 실제 효과는 반드시 한 번의 독립 제출로 확인해야 한다.

## 1. 왜 새 음악 전문가가 필요한가

실제 제출 probe를 v18 계열 점수에 역산하면 대략 Voice EER `0.216`, Music EER
`0.371`, File EER `0.274`였다. 총점 0.8에 필요한 ADS 약 `0.779`를 Music만
개선해 맞춘다고 가정하면 Music EER을 약 `0.229`까지 낮춰야 한다. 따라서 현재
가장 큰 병목은 Voice보다 Music이며, 특히 다음 두 경우가 File 오류로도 전파된다.

- fake voice + real music
- real voice + fake music

기존 speech anti-spoofing 모델과 짧은 구간 artifact 모델만 여러 seed로 늘리면
서로 비슷한 오류를 반복한다. 새 전문가는 일반 오디오용 EAT의 저층 음향 질감과
고층 의미 표현을 함께 사용해 representation 자체를 다르게 만들었다.

## 2. 방법

각 오디오에서 최대 세 개의 시간 view를 만들고 EAT encoder의 12개 layer를 한
번만 통과한다. 각 layer의 patch token에서 다음 다섯 통계를 계산한다.

- 시간 평균
- 시간 표준편차
- 평균 절대 시간 변화량
- Teager–Kaiser energy
- CLS token

각 768차원 통계를 label을 보지 않는 고정 Gaussian projection으로 128차원으로
줄인다. 이후 `(view, layer, statistic)` 조합을 token으로 보고 작은 Transformer와
8-head attention pooling이 Music fake 확률을 학습한다. 전체 head는 351,745개
파라미터뿐이며, EAT backbone은 v18의 presence 계산과 같은 forward를 공유한다.

이 설계는 BUT 팀의 ESDD 연구에서 보고한 all-layer hierarchical SSL fusion,
general-audio SSL의 장점을 현재 파이프라인에 맞게 경량화한 것이다. 원 논문은
다층 SSL 정보와 MHFA, domain augmentation을 사용하며 speech SSL보다 general
audio SSL이 유리한 조건을 보고한다: [BUT Systems for the 2025 IEEE Signal Processing Cup](https://arxiv.org/abs/2512.08319).

## 3. 데이터 경계와 학습

학습에는 train partition으로 명시된 7개 bank의 music-present 18,400개만
사용했다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2`
- `channel_invariant_factorial_train_v1`

sampling은 bank, real/fake label, generator/source group을 함께 균형화했다.
factorial dev, source-disjoint 계열, telephone mixed dev의 7개 bank만 checkpoint
선택에 썼다. 다음 자료는 학습과 seed 선택에서 완전히 잠갔다.

- factorial holdout
- phone factorial 1,200개
- YuE cross-component audit
- source-disjoint music telephone
- 사용자가 추가한 Suno 원본 및 codec stress

`data_guard`로 train 파일이 locked partition과 겹치지 않는지 실행 시 검사한다.
오디오 feature cache 전체는 20개 bank, 24,524개 샘플이지만 locked feature는
마지막 평가 때만 읽는다.

## 4. 학습 및 seed ablation

동일 architecture와 분할에서 seed와 Distribution-Statistic Uncertainty(DSU)를
비교했다. 선택 점수는 7개 dev bank의 `1-EER` 평균과 최악값을 반씩 합친 값이다.

| 설정 | 선택 점수 | dev 평균 Music EER | dev 최악 Music EER |
|---|---:|---:|---:|
| no-DSU seed 00 | **0.79373** | 0.15540 | 0.25714 |
| DSU seed 00 | 0.76982 | 0.18036 | 0.28000 |
| no-DSU seed 01 | 0.76338 | 0.18181 | 0.29143 |
| no-DSU seed 02 | 0.79349 | 0.16730 | **0.24571** |

DSU는 source-disjoint/telephone 성능을 전반적으로 떨어뜨렸다. 이미 학습 bank에
codec 및 채널 변형이 충분한 상황에서 latent style을 한 번 더 흔드는 것이
생성 artifact까지 지운 것으로 해석해 제외했다.

seed 00과 02의 균등 logit ensemble이 dev에서 가장 안정적이었다.

| dev bank | Music EER |
|---|---:|
| mixfake music | 0.1250 |
| external mixed | 0.1450 |
| source-disjoint mixed | 0.1300 |
| source-disjoint mixed equal | 0.1200 |
| source-disjoint music | **0.0550** |
| factorial dev | 0.2343 |
| telephone mixed dev | 0.2233 |

세 번째 seed를 추가하면 일부 쉬운 bank는 좋아졌지만 source-disjoint music과
factorial이 각각 `0.0800`, `0.2457`로 나빠졌다. 따라서 seed 개수를 늘리는 것
자체를 MoE 개선으로 간주하지 않고 두 seed만 사용했다.

## 5. hard router와 soft MoE 비교

앞선 실험에서 phone hard expert, content hard expert, confidence hard selection은
평균 또는 최악 bank를 악화시켰다. 학습된 latent attention router도 dev 이득이
locked cross-component에서 역전됐다. 이유는 세 가지다.

1. speech/music 및 phone/non-phone 경계가 이산적이지 않다.
2. router 오분류 한 번이 강한 전문가 선택 오류로 그대로 전파된다.
3. 기존 전문가들이 비슷한 representation을 보면 전문가를 골라도 오류 다양성이
   충분하지 않다.

이번에는 먼저 all-layer EAT라는 이질적 전문가를 만들고, router는 전화 품질에
따라 그 전문가의 **가중치만** 바꾼다. 전화 router는 3.4–4.2 kHz 대역 절단,
spectral envelope, cepstrum, frame dynamics를 보는 소형 선형 모델이다. 일반
source-disjoint telephone stress에서 Music/File weight 0.30이 0.15보다 좋아
phone weight를 고정했고, locked phone factorial은 선택에 사용하지 않았다.

일반 파일의 File weight를 제한하지 않으면 dev sweep은 0.40을 골랐지만 YuE에서
File EER이 크게 악화됐다. File label은 Voice/Music 중 하나의 residual 오류도
전파하므로, 사전에 `File weight <= 0.15` 안전 제약을 걸었다. 이 조건에서 dev가
선택한 값은 다음과 같다.

| 도메인 | Music weight | File weight |
|---|---:|---:|
| 일반 | 0.20 | 0.10 |
| 전화 | 0.30 | 0.30 |

즉 최종 답은 hard router도 uniform MoE도 아니다. **도메인 router가 낮은 비중의
soft MoE 강도만 조절하는 하이브리드**다.

## 6. 잠금 평가 결과

| 잠금 평가 | v18 ADS | v36b ADS | 변화 |
|---|---:|---:|---:|
| factorial holdout | 0.74551 | 0.75158 | +0.00608 |
| phone factorial | 0.73386 | **0.86132** | **+0.12746** |
| YuE cross-component | 0.82845 | **0.85409** | +0.02564 |

v36b의 세부 EER은 다음과 같다.

| 잠금 평가 | File EER | Voice EER | Music EER |
|---|---:|---:|---:|
| factorial holdout | 0.27055 | 0.22286 | 0.22857 |
| phone factorial | **0.14586** | 0.16750 | **0.10750** |
| YuE cross-component | **0.15421** | 0.10208 | **0.16129** |
| factorial + phone | **0.17569** | 0.18957 | **0.14609** |

factorial에서 Music은 `0.27429 → 0.22857`로 좋아졌지만 File은
`0.25527 → 0.27055`로 소폭 악화됐다. 반면 phone에서는 Music
`0.34500 → 0.10750`, File `0.25829 → 0.14586`으로 크게 좋아졌다. 전화 여부를
분리해서 weight만 다르게 두는 근거가 바로 이 비대칭이다.

새 음악 head 단독으로도 locked Music EER은 factorial `0.2343`, phone
`0.0800`, YuE `0.1935`, source-disjoint telephone `0.1350`이었다. 사용자가 넣은
Suno 보컬 음악은 원본 13/13을 fake 0.5 이상으로 판정했고, codec stress에서는
65개 중 61개를 fake로 유지했다. 이는 Suno에만 맞춘 결과가 아니라 Suno를
checkpoint 선택 후 처음 연 audit 결과다.

## 7. 실제 제출에서 확인할 것

v36b는 v18과 비교해 다음만 바뀐다.

- `MUSIC_FAKE_PROB`: 모든 파일에서 계층형 EAT를 20%, 전화면 30% 결합
- `FILE_FAKE_PROB`: music-present gate를 통과한 파일에서 10%, 전화면 30% 결합

`VOICE_FAKE_PROB`, `VOICE_PRESENT_PROB`, `MUSIC_PRESENT_PROB`는 그대로다. 실제
CPS도 동일해야 한다. 따라서 실제 ADS 변화가 생기면 Music과 그 File 전파의
효과로 해석할 수 있다. 다음 제출 결과에 따라 조정 원칙도 미리 고정한다.

- ADS 상승: component probe로 Music/File 기여를 분해한 뒤 해당 축만 확장한다.
- ADS 동률: hidden 순위가 다시 안 바뀐 것이므로 가중치 증가는 하지 않는다.
- ADS 하락: phone 비중보다 일반 Music weight의 전이 실패를 우선 의심한다.

## 8. 남은 연구 방향

현재 모델은 합성 질감뿐 아니라 고층 EAT token도 보지만, 실제 악기 샘플을 AI가
배열한 음악의 장기 구조를 명시적으로 모델링하지는 않는다. 리듬이 단순히
“전형적”이라는 한 단서만 쓰면 장르 편향이 매우 크다. 다음 전문가는 30–60초
구간의 beat/onset 간격, 반복구조 self-similarity, section transition을 보고,
generator를 통째로 제외한 leave-one-generator-out에서 이득이 있을 때만 낮은
비중으로 추가해야 한다. 과거 MERT long-horizon head는 Udio/Mubert LOGO에서
실패했으므로 그대로 재사용하지 않는다.

현재 단계의 핵심은 전문가 개수보다 **서로 다른 오류를 내는 표현**, router의
정확도보다 **오분류되어도 anchor를 망가뜨리지 않는 soft 결합**이다.
