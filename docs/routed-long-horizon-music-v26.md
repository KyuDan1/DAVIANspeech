# Routed long-horizon music expert v26

2026-09-03 기준. v26은 `temporal_layout_v25`에 원본 오디오의 EAT/SPEAR
start/middle/end 통계를 재사용하는 음악 fake expert를 추가한다. 새 separator나 새
backbone pass는 없다. 일반 오디오에서는 두 head의 MoE를 사용하고, 기존 독립
telephone router가 고신뢰 narrow-band로 판정한 파일에서만 내부 head 비율을 바꾼다.

현재 실제 최고점은 `channel_invariant_moe_v18.zip`의 Total `0.7616379312`, ADS
`0.7363412698`, CPS `0.9893078836`이다. 아래 수치는 로컬 평가이며 실제 점수를
의미하지 않는다. v25와 v26은 아직 실제 평가로 검증되지 않았다.

## 1. AI 작곡과 AI 렌더링은 다른 문제다

AI가 MIDI·악보·구성만 만들고 사람이 실제 악기로 연주하면 vocoder, neural codec,
위상 등 waveform 생성 흔적이 없을 수 있다. 이때는 생성 오디오 탐지보다
**composition forensics**에 가깝다. 그러나 “리듬이 전형적이다”만으로 판단하면
loop 기반 인간 음악, EDM, 장르 관습을 AI로 오인하기 쉽다.

실제로 이번 실험에서 start/middle/end 표현의 시간 분산만 사용한 Music EER은 여러
bank에서 약 `0.5`로 거의 무작위였다. 평균 표현은 훨씬 강했고 평균+분산은 평균보다
조금만 좋아졌다. 따라서 typical rhythm 가설을 현재 detector의 주 신호로 사용하지
않았다. 별도 작곡 expert는 human MIDI와 AI MIDI를 같은 renderer·악기로 렌더링하고,
generator-disjoint와 genre-disjoint에서 검증한 뒤에만 추가해야 한다.

이 판단은 공개 연구의 결론과도 맞는다. 음악 deepfake 탐지는 같은 생성기 분포에서는
쉬워 보여도 unseen generator와 후처리에서 급격히 어려워질 수 있다
([Detecting Music Deepfakes Is Easy but Actually Hard](https://arxiv.org/abs/2405.04181)).
또한 서로 다른 SSL 표현과 내부 일관성을 함께 보는 접근이 단일 spectral shortcut보다
일반화에 유리하다는 연구가 있다
([MoM/CLAM, TMLR](https://openreview.net/pdf/b8a5d900fe30ba4935ca788fe13ce3900c95af8d.pdf)).

## 2. 데이터 경계

학습에는 `configs/data_partitions.yaml`의 `train` 역할만 사용했다. 핵심 bank는
external/mixed, MixFake music, temporal mixed, telephone mixed,
channel-invariant factorial train이다. 다음 데이터는 학습하지 않았다.

- Factorial dev/holdout/locked
- source-disjoint dev banks
- phone factorial 1,200
- YuE cross-component audit
- 사용자가 추가한 Suno vocal 음악

두 classifier가 공유하는 128차원 random projection은 label을 보지 않고 고정 seed로
만들었다. 평가 서버의 PyTorch RNG 버전에 따라 행렬이 달라지지 않도록, 배포 파일에는
seed가 아니라 실제 projection matrix를 저장했다.

## 3. 실패한 공개 음악 detector

공개 음악 detector 두 개를 먼저 교차 생성기 조건에서 검사했다.

| detector | 관찰 | 결정 |
|---|---|---|
| Orphea YuE detector | YuE Music EER `0.4194`, real/fake 모두 높은 fake score | 제외 |
| Suno-vs-GTZAN architecture 재현 | Factorial `0.4857~0.5029`, phone `0.49`, YuE `0.2581` | 제외 |

특정 real corpus와 특정 generator를 분리하는 detector는 실제 대회의 다양한 real/fake
분포에서 shortcut이 되기 쉬웠다. Suno 샘플만 잘 맞는 모델을 제출하지 않고, 여러
generator와 source가 섞인 학습·평가를 유지했다.

## 4. 새 long-horizon expert

EAT와 SPEAR가 이미 만든 원본 오디오 3-view 통계에 다음 연산만 추가한다.

1. 각 SSL token을 LayerNorm하고 고정 Gaussian matrix로 128차원 투영
2. EAT 4 statistics와 SPEAR 13 layer × 4 statistics를 block identity를 유지해 연결
3. start/middle/end의 masked mean과 population standard deviation 계산
4. view count 한 개를 추가해 총 14,337차원 feature 구성
5. 두 개의 linear music classifier 출력

첫 head는 dataset/generator-balanced average risk로 학습했다. 두 번째 head는
smooth worst-generator Group-DRO로 학습했다. 두 head 모두 같은 projection과 같은
학습 bank를 쓰고, classifier만 다르다.

### 독립 head와 고정 MoE

Music EER은 낮을수록 좋다.

| 평가군 | average-risk | Group-DRO | 고정 MoE (`0.4/0.6`) |
|---|---:|---:|---:|
| Factorial holdout | 0.2457 | 0.2171 | **0.2114** |
| Phone factorial | 0.1575 | 0.1750 | **0.1550** |
| YuE | **0.1452** | 0.2258 | 0.1613 |

7개 development bank 평균 EER은 average-risk `0.2017`, Group-DRO `0.1850`, 고정
MoE `0.1825`였다. 따라서 일반 오디오의 기본값은 고정 MoE가 맞다.

Leave-one-generator-out에서는 Brev `0.0912`, Suno `0.2676`으로 일반화했지만 Udio
`0.4708`, Mubert `0.4819`로 실패했다. 공통 생성 단서가 존재하더라도 아직 모든 unseen
generator에 일반화한다고 주장할 수 없다.

## 5. MoE와 router 비교

구성별 hard routing은 채택하지 않았다. 개발 Factorial에서는 music-only에
Group-DRO가 좋았지만 holdout과 다른 source-disjoint bank에서 방향이 달라졌다. 모델
불일치값으로 route하는 규칙도 생성기 종류를 간접 식별할 위험이 있어 넣지 않았다.

전화 route는 개발 데이터에서 근거가 일관됐다. `telephone_mixed_dev_v1` Music EER은
average-risk `0.2433`, 고정 MoE `0.2467`, Group-DRO `0.2567`이었다. 따라서 기존
telephone router가 선택한 파일만 average-risk head를 쓰고, 나머지는 고정 MoE를 쓴다.
모델 전체나 최종 확률 calibration을 바꾸지 않고 **내부 두 head의 비율만** 바꾸므로
과거의 강한 phone score correction보다 위험이 작다.

v18을 정확히 재구성한 뒤 같은 외부 결합을 적용한 결과다.

| 평가군 | v18 ADS | 고정 MoE | phone-routed MoE |
|---|---:|---:|---:|
| Factorial holdout | 0.74551 | **0.76094** | **0.76094** |
| Phone factorial | 0.73386 | 0.80432 | **0.81314** |
| YuE | 0.82845 | **0.86734** | **0.86734** |
| Factorial+Phone 통합 순위 | 0.72218 | 0.79176 | **0.79346** |

통합 순위에서도 route가 좋아졌기 때문에 phone subset만 재보정해 전체 EER을 망치는
현상은 이번 좁은 route에서 관찰되지 않았다.

## 6. 최종 결합 규칙

```text
non-phone expert = logit_MoE(average=0.4, Group-DRO=0.6)
phone expert     = average-risk head

Music = logit_fuse(old_Music, expert, 0.4)
File  = logit_fuse(old_File, expert, 0.2)
        only if MUSIC_PRESENT_PROB >= 0.7
Voice, Voice Presence, Music Presence = unchanged
```

File route는 검증된 Music Presence만 사용한다. 고정 File 20%, hard presence gate,
presence-proportional soft route를 비교했으며 hard 0.7은 dev/holdout/YuE에서 고정
결합과 동률이고 phone ADS를 `0.80225→0.80432`로 개선했다. 더 복잡한 soft route는
평가군별 방향이 달라 제외했다.

v26은 이 단계를 v18 invariant head 뒤, Spectra Voice와 sparse/layout consistency
앞에 둔다. 따라서 개선된 Music score를 마지막 file-local consistency가 사용할 수
있다. 새 expert는 원본 EAT/SPEAR만 사용하며 stem authenticity score를 만들지 않는다.

## 7. 배포 검증

- 학습 스코어와 배포 스코어 최대 절대 오차 `4.1e-7`, 전체 rank 동일
- 3파일 GPU end-to-end smoke 성공, telephone route `1/3`
- 전체 test suite `67 passed`
- 새 checkpoint 약 `1.4MB`, 추가 backbone pass 없음
- ZIP 최상위: `model/`, `script.py`, `requirements.txt`
- ZIP member `133`, 중복 `0`, CRC 오류 없음
- ZIP `9,173,453,543` bytes, 압축 해제 `9,173,425,907` bytes
- 최대 member `2,387,980,808` bytes
- SHA-256: `7f8909800c8fe639166bfc9ed7a2fd9a203cdaeefbf83b1d7aae0e375da39bc7`

제출 파일은 저장소 루트의 `routed_long_horizon_v26.zip`이다. 실제 리더보드에서
개선되지 않으면 가장 먼저 분리할 ablation은 `v25 + long-horizon Music only`와
`phone internal route off`다. 로컬 개선폭이 크더라도 실제 0.8 달성을 보장하지는 않으며,
다음 실제 점수로 eval mixture 비율과 router calibration을 다시 정렬해야 한다.
