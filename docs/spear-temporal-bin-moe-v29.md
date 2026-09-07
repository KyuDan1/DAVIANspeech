# SPEAR temporal-bin MoE v29

2026-09-03 기준. v29는 실제 리더보드 최고점 `channel_invariant_moe_v18`
(Total 0.761638, ADS 0.736341, CPS 0.989308)에서 이어진 v27/v28 위에,
원본 오디오의 짧은 구간을 직접 판정하는 SPEAR expert를 추가한다. 소스 분리는
사용하지 않는다.

## 결론

- 콘텐츠 종류를 기준으로 expert 하나를 고르는 hard router보다 soft MoE가 안전했다.
- 전화 채널처럼 의미가 분명한 도메인에는 기존 router를 유지한다.
- 전화 도메인 안에서도 authenticity 모델 하나를 완전히 선택하지 않고 MoE를 쓴다.
- 새 expert는 기존 SPEAR encoder pass에 얹히므로 큰 모델 추론은 추가되지 않는다.

즉 최종 구조는 **도메인은 router, 진위 판정은 MoE**다.

## 방법

기존 SPEAR 두 번째 pass는 원본 오디오의 시작/중간/끝 10초 crop을 이미 처리한다.
v29는 각 crop을 8개의 약 1.25초 bin으로 나누고 SPEAR shallow layer 0--3에서 다음
통계를 만든다.

1. 평균과 표준편차
2. 인접 프레임 차분
3. TKEO 계열 에너지
4. 고정 random projection 1280→64

작은 64-hidden MLP가 bin별 `music presence`와 `music fake`를 예측한다. 파일 점수는
bin presence로 가중한 log-mean-exp로 집계한다. 이 구조는 순차 혼합에서 음악이
있는 구간만 보고, 동시 혼합에서는 보컬에 덜 의존하면서 음악의 국소 artifact를
찾는 내부 soft router 역할을 한다.

학습에는 train/dev union 22,400개를 사용했다. 다음은 끝까지 학습에서 제외했다.

- `factorial_eval_1200_v2` holdout
- `phone_factorial_1200_v1`
- YuE cross-component audit
- 사용자가 추가한 Suno 보컬 음악 13곡

## head 선택

선형 head, class-balanced loss, Group-DRO, channel consistency, 64-hidden MLP를 비교했다.
최종 seed05 MLP의 개발셋 평균 Music EER은 0.13327, 최악 도메인 EER은 0.20667이다.
3-seed 평균도 시험했지만 독립 audit selection이 seed05 단독보다 낮아 단일 head를
채택했다. 전체 시스템은 이 head와 기존 장주기 head를 결합하는 MoE다.

Suno 13곡은 학습하지 않았는데도 모두 fake 확률 0.5를 넘었다. 확률 범위는
`0.8486--0.9872`, 평균은 `0.9275`였다. 이는 현재 acoustic/temporal 특징이 특정
Suno 파일에만 학습된 결과는 아니라는 최소한의 독립 확인이다.

## router 대 MoE ablation

아래 수치는 v28을 동일한 anchor로 재구성한 결과다. `factorial+phone`은 서로 다른
원천의 일반 holdout 400개와 전화 변형 1,200개를 합친 1,600개 평가다.

| 방식 | 설명 | factorial+phone ADS |
|---|---|---:|
| v28 anchor | 새 expert 없음 | 0.80627 |
| soft MoE | 기존 60% + temporal-bin 40% | **0.82376** |
| music-presence hard route | presence≥0.7이면 새 expert 100% | 0.82063 |
| confidence hard route | 더 확신하는 expert 하나 선택 | 0.82481 |
| phone weight route | phone은 70%, 그 외 40% | 0.82624 |
| phone head+weight route | phone은 seed04 70%, 그 외 seed05 40% | 0.82898 |

마지막 전화 route는 합친 평가에서는 가장 높았지만 선택용 dev에서는 ADS가
`0.77946→0.77260`으로 하락했다. 같은 `telephone_flac`의 dev/holdout에서도 강한
가중치가 불안정했다. hidden test의 전화 비율만 믿고 이 설정을 고르면 synthetic
phone audit에 과적합될 위험이 있으므로 v29에는 넣지 않았다. v29는 네 독립 bank가
모두 좋아진 고정 40% soft MoE를 사용한다.

| 평가군 | v28 ADS | v29 ADS | 변화 | Music EER 변화 |
|---|---:|---:|---:|---:|
| 선택용 dev | 0.76535 | 0.77945 | +0.01410 | 0.21714→0.18286 |
| factorial holdout | 0.76948 | 0.78517 | +0.01569 | 0.22286→0.20571 |
| phone factorial | 0.82607 | 0.84111 | +0.01504 | 0.17500→0.13750 |
| YuE | 0.88058 | 0.88542 | +0.00484 | 0.16129→0.14516 |

factorial+phone 통합 ADS는 `0.80627→0.82376`이다. Voice 출력은 건드리지 않는다.
Music 개선이 File에 전달되도록 v28 consistency를 다시 계산하되, 원래 File 순위의
25%를 남기고 갱신 결과를 75%만 반영한다.

## AI가 실제 악기 샘플만 사용한 경우

실제 악기 음원을 재배열한 생성 음악은 codec/보코더 흔적이 약할 수 있으므로
artifact detector 하나로 완전히 해결되지 않는다. 사용자가 제안한 리듬의 전형성은
가능한 신호지만 장르, 드럼 루프, quantization 때문에 real 음악도 매우 규칙적이어서
단독 판정 기준으로 쓰면 오탐 가능성이 크다.

다음 단계는 현재 forensic branch와 별도로 긴 시간축의 구조를 보는 branch를 만든다.
beat 간격 분포, section 반복, self-similarity, harmony/chroma 전이, embedding novelty를
30--60초 단위로 측정하되 생성기별 분리 holdout에서만 채택한다. router가 이를 hard
선택하기보다 두 branch의 uncertainty와 전화/codec 도메인을 입력으로 받아 낮은
가중치부터 soft gating하는 것이 안전하다.

## 배포 검증

- standalone extractor와 제출 통합 extractor의 기존 통계 최대 오차: 0
- temporal-bin 특징 최대 오차: 0, mask 완전 일치
- 3파일 GPU end-to-end smoke: 성공
- 전화/비전화 혼합 smoke에서 router: 1/3 선택
- 출력 5개 확률: 모두 finite, `[0, 1]`
- 전체 회귀 테스트: 78개 통과
- ZIP: 7,054,346,376 bytes, 해제 7,909,993,631 bytes, 116개 엔트리
- ZIP CRC: 이상 없음, 중복/pycache 없음, 최상위는 `model/`, `script.py`,
  `requirements.txt`만 존재
- SHA256: `3c658b3f09de925edd241b3ab30388439f509e2adb6c97adc1a17cace9828e67`

주요 파일은 다음과 같다.

- `src/spear_temporal_bins.py`
- `src/spear_temporal_bin_head.py`
- `src/spear_temporal_bin_inference.py`
- `scripts/train_spear_temporal_bin_mil.py`
- `scripts/evaluate_temporal_bin_fusion.py`
- `scripts/build_spear_temporal_bin_v29_submission.py`
- `reports/spear_temporal_bin_v1/router_vs_moe.csv`
