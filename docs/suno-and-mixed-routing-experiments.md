# Suno 보컬곡 및 분리 없는 혼합 routing 실험

작성일: 2026-08-30

## 목적

사용자가 추가한 Suno 보컬곡 13개를 실제 사용자 생성 OOD 평가 자료로 보존하고,
다음 가설을 순서대로 확인했다.

1. 현재 음악 전문가가 Suno 완성곡을 AI 음악으로 잡는가?
2. speech용 XLS-R이 Suno의 생성 보컬도 판별하는가?
3. waveform separation 없이 순차 혼합과 동시 overlap을 구분할 수 있는가?
4. raw general-audio latent의 단순 linear head가 source-disjoint 혼합에 일반화하는가?

## 데이터 감사와 사용 원칙

- 13개 MP3, 전부 48 kHz stereo
- 길이 94.76--263.60초
- SHA-256이 모두 달라 byte-level 중복 없음
- 표본이 작고 동일 Suno 환경일 가능성이 있으므로 **학습에 넣지 않고 고정 OOD
  holdout으로 사용**한다.
- `scripts/build_suno_vocal_eval.py`가 곡별 결정적 60초 crop을 만들고 mono 16 kHz
  PCM16 FLAC으로 통일한다.
- 이 셋은 13개가 모두 양성이므로 단독 EER를 보고하지 않는다. threshold recall과
  real control을 붙인 진단만 사용한다.

## 현재 v11의 Suno 결과

| 출력 | 평균 | 중앙값 | `>=0.5` 비율 |
|---|---:|---:|---:|
| File fake | 0.512 | 0.515 | 53.8% |
| Voice fake | 0.549 | 0.522 | 69.2% |
| Music fake | 0.512 | 0.515 | 53.8% |
| Voice present | 0.256 | 0.249 | **0%** |
| Music present | 0.901 | 0.901 | 100% |

PANNs는 보컬곡을 모두 음악으로 보지만 voice presence는 13곡 모두 0.5 미만이다.
SPEAR mixture router도 13곡 모두 0에 가까워 보컬곡을 혼합으로 routing하지 못했다.
대회 CPS는 ranking metric이라 이 값만으로 CPS 실패라고 할 수는 없지만, 현재
`presence gate=0.7`을 fake routing에 재사용하면 보컬 성분이 완전히 비활성화된다.

### 음악 expert

| expert | Suno 중앙값 | FMA real 중앙값 | Suno 대 FMA Music EER |
|---|---:|---:|---:|
| XLS-R original music | 0.039 | 0.096 | 0.615 |
| XLS-R Echoes | 0.298 | 0.270 | 0.468 |
| EAT original | 0.010 | 0.124 | 0.689 |
| EAT Echoes | 0.776 | 0.420 | 0.289 |
| SPEAR music | 0.000 | 0.001 | 0.534 |
| mixed-music head | 0.988 | 0.004 | 0.142 |
| **Fourier** | **1.000** | **0.005** | **0.0025** |
| v11 최종 music | 0.515 | 0.195 | 0.076 |

Fourier가 이 Suno 셋에서는 압도적으로 강하다. FMA prospective 절반만 사용해도
EER는 0.005다. 그러나 양성 곡이 13개뿐이고 generator/codec shortcut일 수 있으므로
Fourier 단독 제출로 바로 바꾸지 않는다. Suno/Udio 및 codec 변환 holdout을 추가해
확인해야 한다.

### 보컬 expert의 한계

PANNs `Singing` 계열 점수가 높은 FMA real 13개를 약한 proxy control로 사용했다.
이 proxy에서 raw XLS-R Voice EER는 0.615, 현재 voice ensemble은 0.538이었다.
speech-only에서 강한 XLS-R을 singing voice에 그대로 적용하는 것은 맞지 않는다.
분리 stem score는 EER 0.231이지만 real stem에서도 중앙값이 0.933이라, 생성 보컬을
찾은 것이 아니라 separation artifact에 공통으로 반응한 결과일 가능성이 높다.

## 동일 길이 순차/동시 혼합 평가

기존 synthetic mixture는 simultaneous가 4초, sequential이 8초여서 길이 shortcut이
있었다. `scripts/build_eval_mixtures.py --equal-duration 8`을 추가하고 두 조건을 모두
8초로 재구성했다. Voice real/fake × Music real/fake 네 조합은 조건별로 균등하다.

원본을 약 4초 PANNs chunk 두 개로 나누고 다음 feature만 계산했다.

```text
cross_switch = abs((voice_0 - music_0) - (voice_1 - music_1))
```

waveform separation이나 생성 decoder는 전혀 사용하지 않았다.

| 결과 | 값 |
|---|---:|
| sequential/simultaneous ROC-AUC | 0.9992 |
| fit accuracy | 98% |
| holdout accuracy | 97% |
| holdout sequential recall | 94.8% |
| holdout simultaneous recall | 100% |

따라서 순차 혼합은 분리하지 않고 temporal activity routing으로 거의 해결할 수 있다.

## detector routing 결과

| 조건 | 방법 | File EER | Voice EER | Music EER | ADS |
|---|---|---:|---:|---:|---:|
| 전체 | v11 | 0.200 | 0.140 | 0.270 | 0.791 |
| 전체 | 예측 temporal router | 0.200 | 0.100 | 0.270 | 0.799 |
| 전체 | raw XLS-R voice + Fourier music | 0.160 | 0.140 | 0.220 | **0.826** |
| sequential | v11 | 0.167 | 0.100 | 0.280 | 0.813 |
| sequential | oracle raw/Fourier | 0.167 | **0.000** | 0.180 | **0.863** |
| sequential | 예측 temporal router | 0.193 | 0.020 | 0.220 | 0.833 |
| simultaneous | v11 | 0.200 | 0.120 | 0.300 | 0.786 |
| simultaneous | raw/Fourier | 0.160 | 0.120 | 0.240 | **0.824** |

하지만 external mixed에서는 raw XLS-R voice가 simultaneous Voice EER 0.41로
무너진다. 반대로 기존 base voice는 0.21이다. source-disjoint 혼합에서는 raw가
더 좋고 external에서는 base가 더 좋은 domain conflict가 있으므로 단순 weight
변경을 제출에 넣으면 안 된다.

## raw SPEAR latent 학습

새 equal-duration 200개에서 16,640-D SPEAR layer-wise embedding을 추출하고
VOICE/MUSIC/FILE linear head를 학습했다. 출처가 다른 external mixed 400개에
그대로 평가했다.

| label | train EER | external OOD EER |
|---|---:|---:|
| Voice | 0.00까지 감소 | 0.45--0.51 |
| Music | 0.00까지 감소 | 0.50--0.52 |
| File | 0.00까지 감소 | 0.54--0.56 |

layer별로 제한해도 최선은 Voice 0.415, Music 0.470이었다. competition_v2/v3까지
multi-source 학습에 넣으면 Music은 0.38까지 내려가지만 Voice는 0.42에 머문다.
즉 frozen SPEAR mean embedding + linear head는 생성 artifact보다 source identity를
외우며, overlap 해결책으로 부족하다.

## 결론과 다음 우선순위

1. **Speech-only**: separator 없이 NII XLS-R 유지.
2. **Music-only/AI 완성곡**: Fourier를 핵심 expert로 유지하되 codec 및 generator
   holdout을 확대한다.
3. **Sequential**: raw chunk activity로 구간을 routing하고, voice 구간에는 raw
   XLS-R, music 구간에는 Fourier/general-audio detector를 적용한다.
4. **Overlap**: frozen embedding linear head는 중단한다. raw waveform을 받는
   MixFake식 frequency/texture prompt tuning 또는 frame-level component query를
   multi-source mixture에서 학습해야 한다.
5. separation은 detector waveform 입력으로 사용하지 않고 상대 에너지/soft mask
   보조 신호로만 비교한다.

MixFake 공식 코드는 공개됐지만 공개 checkpoint는 없고, 전체 데이터는 CC BY 4.0
약 71.6GB다. 다음 큰 실험은 데이터와 license를 기록한 뒤 official prompt model을
재학습하거나, 같은 HHT/TKEO signal prior를 현재 general-audio encoder의 작은
frame-level head에 이식하는 것이다.

