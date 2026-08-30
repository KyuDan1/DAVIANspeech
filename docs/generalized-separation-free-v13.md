# v13: domain-balanced separation-free detector

## 목표와 결론

v12의 단순함은 유지하되 특정 생성기나 한 평가셋에 맞춘 성능이 아니라, 서로 다른 원본·혼합 방식·코덱에서 최악 성능을 개선하는 것을 목표로 다시 구성했다.

최종 v13은 다음 네 단계뿐이다.

1. PANNs로 음성/음악 존재 순위를 출력한다.
2. 원본 파형의 NII XLS-R-2B 점수와 혼합 적응 linear head를 logit 공간에서 결합해 Voice fake를 구한다.
3. 원본 파형의 Fourier 점수와 4초 구간 중 최대 점수를 logit 공간에서 결합해 Music fake를 구한다.
4. `max(Voice fake, Music fake)`를 File fake로 사용한다.

source separation, diffusion separator, stem, EAT, SPEAR와 파일 간 통계는 최종 추론에서 사용하지 않는다.

## 누수 없는 데이터 프로토콜

`configs/data_partitions.yaml`에 train, development, locked eval, OOD holdout, stress eval 역할을 고정했다. `src/data_guard.py`는 학습 truth와 보호된 평가 데이터의 다음 identity를 모두 비교한다.

- `ID`
- `GROUP_ID`
- `VOICE_SOURCE_ID`
- `MUSIC_SOURCE_ID`
- `SOURCE_FILE`

두 head의 학습에는 아래 세 혼합 train 세트, 총 2,400개만 사용했다.

- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`

v12에서 한 번 사용한 `source_disjoint_mixed_locked_v1`은 v13 학습, 가중치 선택, 최종 평가에 전혀 사용하지 않았다.

## 학습 방법

### Music head

원본 파형에서 5–8 kHz Fourier residual 3,061차원을 추출했다. 세 학습 도메인과 real/fake class의 총 가중치가 각각 같도록 sample weight를 주고 L2 logistic regression을 학습했다.

- 선택된 `C=0.01`
- 새 domain-balanced head 75% + 기존 Echoes head 25% (logit 결합과 동치인 weight 결합)
- 파일 전체 점수 30% + 4초 segment 최대 점수 70% (logit 결합)

segment max는 순차 혼합에서 음악이 존재하는 구간의 artifact가 전체 평균에 희석되지 않게 한다. separator가 아니므로 원본 artifact를 변경하지 않는다.

### Voice head

동일한 원본 파형의 XLS-R mean-pooled 1,920차원 표현으로 domain-balanced logistic regression을 학습했다. 2B encoder는 고정했다.

- 선택된 `C=0.03`
- 공개 NII head 40% + 혼합 적응 head 60% (logit 결합)

새 head를 단독 사용하지 않은 이유는 공개 NII head가 가진 새로운 speech generator 일반성을 보존하기 위해서다.

### 선택 기준

평균 ADS 최대화가 아니라 다음 여섯 개발 조건 ADS의 최솟값을 먼저 최대화하고 평균으로 tie-break했다.

- external mixed: sequential / simultaneous
- source-disjoint mixed: sequential / simultaneous
- equal-duration source-disjoint mixed: sequential / simultaneous

Suno와 prospective 음악 split은 hyperparameter 선택에 사용하지 않았다.

## 결과

### 개발 혼합 세트

| 데이터 | 조건 | v12 ADS | v13 ADS |
|---|---|---:|---:|
| external mixed | sequential | 0.824 | **0.859** |
| external mixed | simultaneous | 0.769 | **0.786** |
| source-disjoint mixed | sequential | 0.862 | **0.912** |
| source-disjoint mixed | simultaneous | 0.774 | **0.784** |
| equal-duration mixed | sequential | 0.863 | **0.888** |
| equal-duration mixed | simultaneous | 0.824 | **0.865** |

- 최악 ADS: `0.769 → 0.784`
- 여섯 조건 평균 ADS: `0.819 → 0.849`
- 최악 Voice EER: `0.41 → 0.35`

직접 end-to-end 실행한 equal-duration 결과도 sequential `0.888`, simultaneous `0.8647`로 post-hoc 계산과 일치했다.

### 선택에 사용하지 않은 테스트

- source-disjoint music prospective Music EER: `0.040`
- 사용자가 추가한 Suno 보컬곡: **13/13 fake**
- Suno 최저/중앙 Music fake: `0.958 / 0.997`

### 코덱/전화 스트레스 테스트

| 변형 | Suno recall @ 0.5 | 최저 점수 |
|---|---:|---:|
| WAV | 13/13 | 0.958 |
| FLAC | 13/13 | 0.958 |
| MP3 재인코딩 | 13/13 | 0.883 |
| OGG 재인코딩 | 13/13 | 0.994 |
| 8 kHz 전화대역 | 0/13 | 0.455 |

전화대역에서는 5–8 kHz 관찰 대역 자체가 사라진다. 두 fallback도 실험했지만 채택하지 않았다.

- 1–4 kHz Fourier head: source-disjoint Music EER 0.22, 점수가 0.51에 몰리는 shortcut
- EAT-Echoes: 전화 Suno 13/13이지만 source-disjoint 전화 Music EER 0.42
- 전화 증강 EAT 재학습: prospective EER 0.44, 전화 Suno 0/13

따라서 일반성이 검증되지 않은 fallback을 제출에 넣어 정상 음악을 악화시키지 않았다. 전화채널 음악 fake 탐지는 남은 가장 명확한 연구 과제다.

## 재현 파일

- `scripts/experiment_robust_fourier_head.py`
- `scripts/experiment_robust_xlsr_voice_head.py`
- `scripts/score_fourier_segments.py`
- `scripts/build_codec_stress_eval.py`
- `src/simple_pipeline.py`
- `model_heads/fourier-domain-balanced-v13.npz`
- `model_heads/xlsr-mixed-domain-balanced-v13.npz`

## 해석

이번 개선은 모델 수를 늘린 결과가 아니다. 원본 artifact를 보존하고, 서로 다른 학습 도메인의 영향력을 균형화하며, 순차 혼합은 시간 구간에서 관찰한다는 세 원칙에서 나왔다. Suno만 양성으로 만드는 규칙은 만들 수 있었지만 source-disjoint EER이 나빠서 폐기했다. 이 때문에 v13의 13/13 Suno 결과는 Suno threshold를 직접 맞춘 결과가 아니라, 개발 세트에서 고정한 detector가 OOD holdout에서도 유지된 결과다.
