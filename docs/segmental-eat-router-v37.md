# v37: segment-content EAT와 router-vs-MoE 검증

## 결론

새 음악 전문가는 source separation 없이 원본 mixture의 최대 8개 구간을 보고,
EAT 12개 layer의 음향·의미 정보를 시간 순서대로 결합한다. 이 전문가를 기존
계층형 EAT 전문가와 **고정 25% logit soft MoE**로 합치는 방식을 선택했다.

파일을 전문가 하나로 보내는 hard router, 전화 여부에 따른 내부 expert weight,
confidence/길이 기반 router, 별도 router-train 데이터로 학습한 bounded router를
모두 비교했지만 고정 MoE보다 안정적이지 않았다. 기존 전화 router는 v36b와
동일하게 최종 음악 expert가 전체 파이프라인에 들어가는 비중만 조절한다.

새 expert는 분리된 stem을 사용하지 않는다. 다만 v18 anchor의 XLS-R 음성 경로는
그대로이므로 전체 파이프라인에서 Demucs를 제거한 것은 아니다.

## 1. 연구 가설

실제 악기 sample을 사용한 AI 음악은 짧은 구간의 합성 질감만으로는 놓칠 수 있다.
따라서 다음 두 단서를 분리해 검증했다.

1. `content`: 여러 구간의 EAT 표현을 순서가 있는 Transformer로 결합한다.
2. `structure`: 구간 간 self-similarity로 반복, section 변화, 장기 구성을 본다.

Segment Transformer는 beat-aware segment와 self-similarity 경로가 AI 음악 탐지에
유효할 수 있음을 보고한다: [Segment Transformer](https://arxiv.org/abs/2509.08283).
하지만 생성기·후처리 변화에서 성능이 쉽게 무너진다는 연구도 있으므로, 리듬이
전형적이라는 단일 가설을 채택하지 않고 generator/source-disjoint 검증을
우선했다: [unseen-generator robustness study](https://arxiv.org/abs/2501.10111),
[augmentation robustness study](https://arxiv.org/abs/2507.10447).

## 2. 데이터 경계

전문가 학습에는 v36과 같은 train partition 7개의 music-present 18,400개만
사용했다. dev/holdout/phone/YuE/Suno는 학습에 들어가지 않았다. `data_guard`가
source identity 기준 누수를 검사한다.

checkpoint와 내부 MoE weight 선택은 다음 7개 development bank에서 수행했다.

- mixfake music dev
- external mixed
- source-disjoint mixed / equal / music
- factorial dev
- telephone mixed dev

router 일반화 확인을 위해 clean source와 대응되는 전화 변환 3개 bank 800개를
추가로 계산했다. 이 중 development 두 bank는 router 선택 검증에, stress bank는
선택 후 확인에만 사용했다. factorial holdout, phone factorial, YuE,
source-disjoint music telephone, Suno는 고정안을 정한 뒤 한 번만 열었다.

학습형 router는 모델 학습/평가와 겹치지 않는 `router_train`만 사용했다.

- `multigen_music_presence_train_v1`: music-present 474개
- `phone_presence_factorial_train_v1`: music-present 2,000개, 5개 전화 codec

두 파일 모두 누수 검사를 통과했다.

## 3. 모델

각 파일을 6초 crop으로 자르되 길이에 따라 최대 8개까지 배치한다. 10~20초
오디오를 거의 같은 crop 8개로 채우지 않도록 목표 overlap을 최대 50%로 두며,
4~6초 파일은 한 view만 사용한다.

각 view에 대해 EAT 12개 layer에서 다음 통계를 구한다.

- mean, standard deviation
- mean absolute temporal delta
- Teager–Kaiser energy
- CLS token

각 768차원 통계를 label-free Gaussian projection으로 128차원화한다. 공유
layer/statistic Transformer가 view embedding을 만들고, content Transformer가
이를 시간 순서대로 처리한다. backbone과 projection은 기존 계층형 EAT expert와
공유하고 두 content seed를 logit 평균한다.

배포에서는 기존 start/middle/end view와 새 segment view의 합집합을 EAT에 한 번만
통과시킨다. 기존 endpoint statistics와 새 segment statistics를 각 head에 따로
전달하므로 diffusion separator artifact가 새 음악 판단에 들어오지 않는다.

## 4. content와 장기 구조 ablation

선택 점수는 `1 - 0.5 × (평균 Music EER + 최악 Music EER)`이다.

| 모델 | dev 선택 점수 | 평균 EER | 최악 EER |
|---|---:|---:|---:|
| 기존 2-seed hierarchical | **0.80910** | 0.14752 | 0.23429 |
| segment content seed 00 | 0.78594 | 0.15956 | 0.26857 |
| segment content seed 02 | 0.79659 | 0.15538 | 0.25143 |
| segment content 2-seed | 0.80690 | **0.14619** | 0.24000 |
| content + structure | 0.77321 | 0.19025 | 0.26333 |
| structure only | 0.58607 | 0.33598 | **0.49188** |

`structure only`는 가장 어려운 mixfake bank에서 거의 무작위였다. 리듬과 반복
구조만으로 AI 음악을 판별하면 장르/길이 shortcut을 배우기 쉽다는 뜻이다.
따라서 self-similarity branch는 제외했다. content expert 단독도 기존 모델을
대체하지 못하지만 오류가 달라 낮은 비중의 residual로는 유효했다.

기존 hierarchical과 segment-content 2-seed ensemble을 25% logit 결합하면 7개
dev bank의 선택 점수가 `0.80910 → 0.82101`로 올랐다. Music EER은 다음과 같다.

| dev bank | 기존 | 고정 25% MoE |
|---|---:|---:|
| mixfake music | 0.1250 | **0.1088** |
| external mixed | 0.1450 | **0.1350** |
| source-disjoint mixed | 0.1300 | **0.1200** |
| source-disjoint mixed equal | 0.1200 | **0.1100** |
| source-disjoint music | 0.0550 | 0.0550 |
| factorial dev | 0.2343 | **0.2171** |
| telephone mixed dev | 0.2233 | **0.2200** |

## 5. router와 MoE 직접 비교

추가 전화 bank를 포함한 10개 bank에서 동일한 frozen prediction만 사용했다.

| 전략 | 평균 EER | 최악 EER | 고정 MoE 대비 단일-bank 최대 악화 |
|---|---:|---:|---:|
| 고정 25% soft MoE | 0.16459 | 0.340 | **0.000** |
| confidence soft router | **0.16384** | 0.340 | 0.000 |
| phone soft router 25/40% | 0.16509 | 0.330 | 0.020 |
| confidence hard expert | 0.16864 | **0.310** | 0.050 |
| phone hard expert | 0.17526 | 0.310 | 0.060 |
| segment expert only | 0.17533 | 0.310 | 0.060 |

전화 hard route가 실패한 이유는 전화 데이터 안에서도 expert 우열이 바뀌기
때문이다.

| 전화 변환 bank | hierarchical | segment only | 고정 25% |
|---|---:|---:|---:|
| external mixed telephone | **0.160** | 0.180 | **0.160** |
| source-disjoint mixed telephone | 0.360 | **0.310** | 0.340 |
| source-disjoint equal telephone | 0.220 | 0.240 | **0.180** |

전화 weight를 40%로 높이면 첫 두 bank는 각각 0.155/0.330으로 좋아지지만 세 번째가
`0.180→0.200`으로 나빠졌다. 전화라는 관측값만으로는 expert reliability를 알 수
없다.

별도 router-train 2,474개에서는 hierarchical/segment confidence, disagreement,
view 수, 전화 여부를 입력으로 하고 segment weight를 5~45%로 제한한 linear/MLP
router도 학습했다. 약한 규제는 dev bank를 0.01~0.02 EER 악화시켰고, 강한 규제는
모든 파일에 약 25%를 출력하는 고정 MoE로 수렴했다.

confidence soft router는 dev 평균을 0.00075 EER 더 낮췄지만, 고정 후 연 locked
factorial에서 Music EER을 `0.2000→0.2057`로 악화시켰다. 작은 dev 이득보다
generator-disjoint maximin을 우선해 배포에서 제외했다.

관련 재현 결과:

- `reports/segmental_eat_music_v2/router_vs_moe/summary.csv`
- `reports/segmental_eat_music_v2/router_vs_moe/per_dataset.csv`

## 6. 잠금 평가

먼저 음악 expert 자체의 고정 25% 결합을 확인했다.

| locked/stress bank | hierarchical | segment only | 고정 25% |
|---|---:|---:|---:|
| factorial holdout | 0.2343 | 0.2114 | **0.2000** |
| phone factorial | 0.0800 | 0.0875 | **0.0675** |
| YuE cross-component | **0.1935** | 0.2097 | **0.1935** |
| source-disjoint music telephone | 0.1350 | **0.1050** | 0.1350 |

Suno 보컬 음악은 기존 hierarchical이 원본 13/13, codec stress 61/65를 fake로
판정했다. 고정 MoE는 원본 13/13과 codec stress **65/65**를 모두 0.5 이상으로
유지했다.

이 inner MoE를 v36b의 outer Music/File weight에 넣은 전체 ADS는 다음과 같다.
이는 로컬 검증 점수이며 실제 hidden leaderboard 예상값으로 해석하면 안 된다.

| 평가축 | v18 | v36b | v37 | v36b 대비 |
|---|---:|---:|---:|---:|
| factorial holdout | 0.74551 | 0.75158 | **0.75792** | +0.00634 |
| phone factorial | 0.73386 | **0.86132** | 0.86046 | -0.00086 |
| YuE cross-component | 0.82845 | 0.85409 | **0.85681** | +0.00272 |
| factorial + phone | 0.72218 | 0.83042 | **0.83146** | +0.00105 |

v37의 File/Voice/Music EER은 factorial
`0.26473/0.22286/0.21714`, phone
`0.14757/0.16750/0.10750`, YuE
`0.14878/0.10208/0.16129`이다. Voice와 CPS는 v36b 그대로다.

## 7. 배포 검증

- 후보: `segmental_eat_moe_v37_submit.zip`
- 압축 크기: 7,050,430,804 bytes
- ZIP 내부 해제 크기: 7,918,365,444 bytes
- entry: 118개, 중복 0개
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- 전체 CRC: 오류 없음
- SHA-256: `d66c2f230a6ffbb144c20dabcbc85a9e7068e71898646b9dde7f847f0d1c6725`
- 전체 테스트: 101개 통과
- clean/phone/mixed 3파일 CUDA smoke: 통과
- v36b 대비 변경 열: File/Music만 변경, Voice/CPS 세 열 동일

B200에서 60초 파일 8개의 EAT endpoint 통계는 평균 0.095초, 최대 8-view 통계는
0.276초였다. EAT 부분은 약 2.9배지만 절대 증가는 batch당 약 0.18초였다. 평가
서버 L4의 정확한 증가는 실측 전까지 확정할 수 없으나, 기존 전체 실행 약 41분과
60분 제한 사이에는 여유가 있다.

## 최종 판단

이번 실험에서는 고정 soft MoE가 정답이다. 단, 이는 router라는 아이디어가 항상
나쁘다는 뜻이 아니다. 다음 router를 채택하려면 다음 조건을 동시에 만족해야 한다.

1. expert별로 우열이 바뀌는 원인이 전화/길이처럼 실제 입력에서 안정적으로
   관측 가능해야 한다.
2. leave-one-generator/source/channel-out에서 고정 MoE보다 나쁜 bank가 없어야 한다.
3. hard selection이 아니라 bounded residual weight로 실패 비용을 제한해야 한다.

현재 phone/content/confidence feature는 첫 번째 조건을 만족하지 못했다. 그래서
v37은 expert diversity는 사용하되 expert 선택은 하지 않는다.
