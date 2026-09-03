# Joint temporal MoE v30

2026-09-03 기준. v30은 v29를 고정한 채, 같은 원본 오디오 SPEAR temporal-bin
특징에서 음성/음악의 존재와 진위를 함께 학습한 작은 expert를 추가한다. 새 source
separation이나 대형 encoder pass는 없다.

## 결론

- 전화/비전화 hard routing보다 모든 샘플에 낮은 비율로 결합하는 soft MoE가 더
  일반적이었다.
- 다만 expert 내부에서는 bin별 voice/music presence가 해당 component authenticity
  pooling의 soft router로 동작한다.
- 최종 후보는 v29와 joint expert를 logit 공간에서 결합한다.
  `File weight=0.25`, `Voice weight=0.25`이며 Music과 두 CPS 열은 바꾸지 않는다.
- 여러 seed 투표도 시험했지만 단일 seed09가 독립 평가축의 최악 성능과 평균 성능을
  함께 봤을 때 더 좋았다.

즉 구조는 **파일 수준에서는 보수적인 MoE, latent temporal-bin 수준에서는
component-aware soft routing**이다.

## 왜 separator 뒤가 아니라 원본 mixture인가

현재 base pipeline에는 HTDemucs stem과 원본 mixture를 사용하는 branch가 모두 있다.
HTDemucs 자체는 diffusion separator는 아니지만, 어떤 learned separator도 입력을
재합성하면서 생성 흔적을 약화하거나 새로운 흔적을 만들 수 있다. v30의 새 expert는
separator 출력을 전혀 사용하지 않고 원본 mixture의 SPEAR hidden state를 직접 본다.

각 10초 crop을 8개 bin으로 나누고 shallow layer 0--3의 평균, 분산, 차분, 비선형
에너지 통계를 고정 projection한 1,024차원 특징을 사용한다. shared MLP는 bin마다
다음을 출력한다.

1. voice presence
2. music presence
3. voice fake
4. music fake

Voice fake는 local voice presence로, Music fake는 local music presence로 가중한
log-mean-exp로 집계한다. 그래서 순차 혼합과 동시 혼합을 같은 파일에서 처리하되,
음성과 음악 중 하나를 파일 단위로 먼저 hard 선택하지 않는다.

## 학습과 누수 방지

학습 코드는 `scripts/train_spear_temporal_joint_mil.py`다. 데이터셋, file label,
voice/music layout cell을 균형화했고 local component label, component fake, file fake,
presence를 함께 학습했다. 아래 평가는 학습 및 checkpoint 선택에서 제외했다.

- `factorial_eval_1200_v2` holdout
- `phone_factorial_1200_v1`
- YuE cross-component audit
- 사용자 제공 Suno 보컬 음악 13곡

선택된 seed09 head는 96-hidden MLP이며 0.90 MB다. Suno 13곡의 독립 추론에서 Music
fake 평균은 0.8958, File fake 평균은 0.9107이었다. v30은 이 head의 Music/CPS 출력은
실제 제출에 쓰지 않는다.

## router 대 MoE 실험

비교 시 v29를 완전히 고정했다. 전화 router만 joint expert를 켜는 hard/soft domain
route, 전체 샘플 soft MoE, 세 seed 단독 및 ensemble, File/Voice/Music 출력 조합을
비교했다.

전화 route는 전화 audit에서 최대 약 `+0.022 ADS`였지만 일반 factorial과 dev의
telephone slice에서 방향이 일정하지 않았다. 가장 공격적인 phone-routed F/V/M
설정은 factorial+phone 통합 ADS가 0.83752였지만 dev는 0.77945에서 0.77351로,
factorial은 0.78517에서 0.78117로 내려갔다. hidden test의 전화 비율에 의존하는
설정이라 탈락시켰다.

반대로 seed09의 전체 soft MoE는 Music을 건드리지 않고 File/Voice만 결합할 때 네
독립 평가축을 모두 개선했다.

| 평가축 | v29 ADS | v30 ADS | 변화 |
|---|---:|---:|---:|
| 선택용 dev | 0.77945 | 0.78875 | +0.00930 |
| factorial holdout | 0.78517 | 0.79332 | +0.00816 |
| phone factorial | 0.84111 | 0.85325 | +0.01214 |
| YuE cross-component | 0.88542 | 0.88814 | +0.00272 |
| factorial+phone 통합 | 0.82376 | 0.83300 | +0.00924 |

`File=0.25`에서 Voice weight 0.20--0.35가 비슷한 plateau를 만들었다. 전체 평균만
최대화한 0.30 대신 일부 순차 혼합 slice의 변동을 줄이는 중앙값 0.25를 잠갔다.
CPS는 제출에서 전혀 바꾸지 않으므로 v29와 동일하다.

## 실제 악기 기반 AI 작곡 신호 실험

사용자 가설처럼 실제 악기 sample을 쓰는 생성 음악은 짧은 codec/vocoder artifact가
약할 수 있다. 관련 연구도 long-range dependence의 중요성을 보고한다
([SONICS](https://arxiv.org/abs/2408.14080)). 다만 알려지지 않은 codec과 가벼운
후처리에서 성능이 크게 떨어지는 것이 현재 AI 음악 탐지의 핵심 문제다
([generalization study](https://arxiv.org/abs/2501.10111),
[augmentation study](https://arxiv.org/abs/2507.10447)).

그래서 MERT start/mid/end 구조 차이, 곡률, cosine distance를 이용한 구조-only
probe를 따로 학습했다. 결과는 dev selection 0.5846, 보강한 MLP도 최대 0.7298로
기존 MERT expert보다 약했다. 리듬/구조의 전형성만으로는 장르, loop, quantization된
real 음악을 AI로 오인하는 confound가 커서 v30에 넣지 않았다. 이 결과는 구조 신호를
버려야 한다는 뜻이 아니라, 더 다양한 real loop/전자음악과 generator-disjoint
학습 없이는 제출 모델에 사용하면 안 된다는 뜻이다.

## 배포 설계

- `src/spear_temporal_joint_head.py`: component-aware latent soft router
- `src/spear_temporal_joint_inference.py`: 기존 temporal-bin 통계 재사용 및 File/Voice
  logit fusion
- `scripts/train_spear_temporal_joint_mil.py`: leakage guard가 있는 학습
- `scripts/build_joint_temporal_moe_v30_submission.py`: 검증된 v29 ZIP에서 v30 생성
- `reports/spear_temporal_joint_v1/router_fusion_sweep.csv`: hard/soft route ablation
- `reports/spear_temporal_joint_v1/ensemble_axis_sweep.csv`: seed/축/가중치 ablation
- `reports/spear_temporal_joint_v1/subgroup_robustness_sweep.csv`: 하위군 안정성

3파일 전체 entrypoint GPU smoke에서 clean, Opus narrow-band phone, mixed audio를 모두
처리했고 전화 router는 1/3만 선택했다. 출력은 finite `[0, 1]`이며 새 후처리가
의도대로 File/Voice만 갱신하는 것은 unit test로 확인했다.

- 전체 회귀 테스트: 83개 통과
- ZIP: 7,055,095,719 bytes, 해제 7,910,903,007 bytes, 120개 엔트리
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- 중복 경로/pycache/4GB 초과 멤버: 없음
- ZIP CRC: 이상 없음 (`testzip=None`)
- SHA256: `4033bd52c20b3041bbf964da02b999863a0e3faf8abd8c0f98852278d67d9bfb`
