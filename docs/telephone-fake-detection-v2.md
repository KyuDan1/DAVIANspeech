# 전화채널 딥페이크 탐지 v2: 연구·실험·제출 후보

2026-09-01 기준. 결론부터 말하면 **새 전화 전용 학습 head는 버리고**, 현재
리더보드에서 양의 효과가 확인된 LME+SPEAR를 유지한 채 좁은 대역 전화 파일에서만
SPEAR 결합을 `0.10 → 0.20`으로 올리는 후보를 선택했다. 음성 점수와 CPS는 전혀
바꾸지 않는다.

이 변경은 1등을 보장하는 큰 도약이 아니라, 원인이 분리된 저위험 단일 ablation이다.
0.80을 위해서는 전화채널 다음에 일반 채널의 Music/RF와 File EER을 훨씬 더 크게
낮춰야 한다.

## 1. 실제 리더보드가 말하는 것

| 제출 | ADS | CPS | 해석 |
|---|---:|---:|---|
| twin anchor | 0.708389 | 0.989325 | 비교 기준 |
| LME | **0.713857** | 0.989325 | window pooling만 변경, `+0.00547` |
| v15 SPEAR | **0.713532** | 0.989172 | File/Music 10%만 변경, `+0.00600` |
| v14 routed | 0.657413 | 0.989172 | router·pipeline·여러 expert 동시 변경으로 실패 |

v14는 전화 router가 나쁘다는 실험이 아니다. 분리 제거, voice/music expert, routing을
한꺼번에 바꿔 인과를 알 수 없게 만든 실험이다. 이번에는 LME+SPEAR 기반을 그대로
두고 **전화일 때의 SPEAR weight 하나만** 바꾼다.

기존 component probe로 역산한 anchor EER은 File `0.2741`, Voice `0.2156`, Music
`0.3714`다. CPS는 이미 `0.989`이고 Music과 File이 주 병목이다. 전화 연구도 먼저
Voice가 아닌 Music/File 손실을 확인하는 순서로 진행했다.

## 2. 연구에서 가져온 가설

- ASVspoof 2021 CRIM은 VoIP/압축 조건에서 codec augmentation, frame 표현의
  higher-order statistics pooling, 이종 모델 score fusion이 도움이 된다고 보고했다.
  [Kang et al., ASVspoof 2021](https://www.isca-archive.org/asvspoof_2021/kang21b_asvspoof.html)
- UR-AIR는 landline(G.711/G.726), cellular(AMR/GSM), VoIP(Silk/G.729/G.722)와
  device impulse response를 학습 augmentation으로 사용했다.
  [Chen et al., ASVspoof 2021](https://www.isca-archive.org/asvspoof_2021/chen21_asvspoof.html)
- 저주파 subband는 codec 변동에 덜 민감했고 ASVspoof 2021 LA에서 EER을 상대적으로
  줄였지만, 깨끗한 조건에서는 정보 손실로 나빠질 수도 있었다.
  [Wang et al., 2022](https://arxiv.org/abs/2211.06546)
- ASVspoof 5 시스템들은 codec/frequency augmentation과 여러 SSL view의 결합을
  사용했다. 다만 SSL front-end에서 augmentation이 항상 이득인 것은 아니어서
  모델별 검증이 필요하다.
  [Xie et al., 2024](https://www.isca-archive.org/asvspoof_2024/xie24_asvspoof.html),
  [Schäfer et al., 2024](https://www.isca-archive.org/asvspoof_2024/schafer24_asvspoof.html)

따라서 세 후보를 차례로 검증했다.

1. 전화채널을 고정 저대역/G.711로 한 번 더 정규화하는 multi-view
2. G.711/G.726/Opus/8 kHz로 학습한 SPEAR 얕은 층 head
3. 새 head 없이, 이미 실제 점수가 오른 SPEAR의 전화조건 결합 비중만 증가

## 3. 새 전화 factorial 평가셋

`scripts/build_phone_factorial_eval.py`로
`data/eval/phone_factorial_1200_v1`을 만들었다. 학습에는 넣지 않는다.

| 원본 형태 | 고유 콘텐츠 | codec 4종 적용 후 |
|---|---:|---:|
| Mixed RR/RF/FR/FF 균형 | 100 | 400 |
| Music-only real/fake 균형 | 100 | 400 |
| Voice-only real/fake 균형 | 100 | 400 |
| 합계 | 300 | **1,200** |

codec은 `resample8k`, `G.711 μ-law`, `G.726 24 kbps`, `Opus-NB 8 kbps`다.
같은 300개 콘텐츠에 네 codec을 반복 적용했으므로 생성기/장르 차이와 codec 차이를
분리할 수 있다. 원본은 source-disjoint mixed/music와 ASVspoof/VCC speech 평가
bank에서 가져왔다.

평가는 제출물과 바이트가 같은 LME anchor pipeline(HTDemucs, XLS-R-2B,
ArtifactNet)을 8 GPU shard로 실행하고, 같은 SPEAR head를 후처리했다.

전화 router는 이 1,200개를 `1,200/1,200` 검출했다. 대응되는 clean 원본 300개는
`0/300`만 전화로 오탐했다. 즉 이번 audit에서 hard gate 오류는 없다.

## 4. Anchor의 전화채널 병목

전체 전화 factorial의 LME anchor 결과:

| File EER | Voice EER | Music EER | ADS |
|---:|---:|---:|---:|
| 0.3100 | **0.1825** | **0.4175** | 0.68325 |

Voice는 실제 리더보드 anchor의 약 `0.216`보다 오히려 낮다. 반대로 Music은
`0.418`이고 Mixed Music EER은 `0.465`다. 즉 이 평가에서 전화 전용 speech head를
교체하는 것은 우선순위가 아니다. 전화 codec이 음악 생성 흔적과 File 순위를
무너뜨리는 것이 핵심이다.

codec별 anchor ADS는 G.711 `0.7480`, G.726 `0.7563`, 단순 8 kHz `0.7327`인 반면
Opus-NB는 **`0.6027`**이다. Opus-NB mixed File EER가 특히 높아 다음 전화 연구의
구체적인 최악 조건으로 남긴다.

## 5. 실패한 방법

### 5.1 추가 저대역/G.711 multi-view

전화 입력을 다시 FFT 저대역 또는 G.711로 canonicalize하고 기존 SPEAR와 평균했다.
일부 Music EER은 줄었지만 File EER가 다른 source bank에서 반대로 움직였다. 최종
결합도 bank별 개선 방향이 달라 채택하지 않았다. 생성 전처리를 한 번 더 거치면
남은 흔적을 더 지우는 문제가 실제로 관찰됐다.

### 5.2 codec-robust SPEAR head 학습

SPEAR layer 0--4에 대해 G.711/G.726/Opus/8 kHz 중 하나를 통째로 빼는
leave-one-codec-out 선택을 수행했다. 선택값은 layer 1, `C=0.003`, 기존 head와
50% 결합이었다.

개발셋에서는 Opus Music EER가 `0.43 → 0.38`로 좋아졌지만, 최종 source-disjoint
audit에서는 전부 악화됐다.

| final audit | 기존 | 학습 head 결합 |
|---|---:|---:|
| mixed / 8 kHz | 0.360 | 0.410 |
| mixed / Opus-NB | 0.430 | 0.440 |
| music-only / G.711 | **0.210** | 0.280 |
| music-only / G.726 | **0.235** | 0.305 |
| music-only / Opus-NB | **0.245** | 0.335 |

codec 일반화와 생성기 일반화는 다른 문제다. codec을 잘 hold-out해도 학습 음악
family에 맞춘 head가 새 source에서 실패했다. 이 head
`spear-phone-codec-robust-v1.npz`는 재현용이며 **제출에 사용하지 않는다**.

## 6. 선택한 단일 변경

```text
clean/wideband: File/Music = 0.90 × LME anchor + 0.10 × SPEAR
narrowband:     File/Music = 0.80 × LME anchor + 0.20 × SPEAR
Voice/CPS:      LME anchor 그대로
```

전화 factorial 전체 결과:

| SPEAR weight | File EER | Voice EER | Music EER | ADS |
|---:|---:|---:|---:|---:|
| 0.00 | 0.3100 | 0.1825 | 0.4175 | 0.68325 |
| 0.10 (현재 v15) | 0.3083 | 0.1825 | 0.4000 | 0.68936 |
| **0.20 (후보)** | **0.3076** | 0.1825 | **0.3875** | **0.69346** |
| 0.30 | 0.2959 | 0.1825 | 0.3675 | 0.70532 |

0.30이 로컬 최고지만 사용하지 않는다. 이미 로컬 최적 weight가 실제 대회에서
거꾸로 간 전례가 있고, 0.10만 리더보드에서 양의 방향이 확인됐다. 0.20은 네 codec
각각에서 anchor보다 좋았던 가장 작은 한 단계 증가라 일반성을 우선한 선택이다.

구현은 `src/telephone_aware_spear_fusion.py`다. router와 점수는 파일마다 독립적으로
계산하므로 평가 파일 간 통계 사용 금지 규정을 지킨다. SPEAR는 원래 v15 pass를
재사용하므로 큰 모델 추론은 추가되지 않고, 6 KB router와 FFT/cepstral 특징만
추가된다.

## 7. 예상 효과와 다음 우선순위

이번 변화는 전화 파일에서 v15 대비 로컬 ADS `+0.0041`이다. 비공개 데이터의 전화
비율이 10/20/30%라면 전체 Total의 단순 기대 증가는 약 `+0.00037/+0.00074/+0.00111`
수준이다. 따라서 안전한 개선 후보지만 0.80을 만드는 핵심 변화는 아니다.

다음 순서는 다음과 같다.

1. 현재 제출된 LME+SPEAR의 실제 점수를 먼저 받아 결합이 가산적인지 확인한다.
2. 그 기반과 오직 전화 weight만 다른 이번 후보를 제출해 인과를 유지한다.
3. 다음 진단 제출은 LME+SPEAR의 Music constant probe로 LME/SPEAR 이득이 File과
   Music 중 어디서 났는지 분리한다.
4. 큰 점수 상승은 일반 채널의 RF(진짜 음성+가짜 음악), 완전 동시 혼합, Music
   generator generalization에서 만들어야 한다. 전화 head처럼 작은 학습 head를
   늘리지 말고, generator-family를 통째로 hold-out한 원본 SPEAR shallow-token
   two-query 모델을 다음 주력 실험으로 둔다.
5. 전화 내부에서는 Opus-NB mixed File EER를 별도 병목으로 두되, 새 모델은 최소
   두 개의 완전 미관측 음악 generator family에서 동시에 개선될 때만 결합한다.

상세 수치는 `reports/phone_factorial_1200_v1/fusion_weight_results.csv`와
`reports/phone_spear_codec_v1/results.csv`에 있다.

## 8. 제출 산출물

- 디렉터리: `submit_phone_spear_v2/`
- ZIP: `phone_spear_v2.zip`
- SHA-256: `3bc3b3bd94cf7643a7bf35656d93dbc52537ead5e9db35747ab80338caf5d488`
- 압축/해제 크기: 6,590,225,627 / 7,143,529,038 bytes
- 최상위 항목: `model/`, `script.py`, `requirements.txt`만 존재
- requirements: `onnxruntime-gpu==1.23.2`
- ZIP 전체 CRC, 중복 member, 10/32 GB 제한, Python compile 통과
- 로컬 테스트: 24개 전체 통과, clean/전화 2파일 fusion smoke에서 `1/2`만
  전화 route로 선택
