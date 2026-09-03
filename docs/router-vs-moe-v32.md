# Router vs. MoE 실험과 v32 후보

## 결론

샘플 전체를 하나의 전문가로 보내는 hard router는 사용하지 않는다. 현재 가장
안전한 구조는 다음의 계층형 MoE다.

1. 실제 리더보드 최고점인 v18을 그대로 anchor로 사용한다.
2. 음성/음악 분리 없이 원본 SPEAR의 1.25초 bin 8개를 보는 temporal-attention
   전문가를 추가한다.
3. 이 전문가는 `FILE_FAKE_PROB`에만 작게 결합한다.
4. 전화 router가 narrow-band 전화 파일이면 이 전문가의 비중만 20%에서
   25%로 높인다.
5. `VOICE_FAKE_PROB`, `MUSIC_FAKE_PROB`, 두 presence 확률은 v18과 동일하게
   유지한다.

즉, router가 기존 모델을 교체하는 구조가 아니라 검증된 MoE 위에서 특정
전문가의 기여도만 제한적으로 조절한다. 이 방식이 domain 오분류의 피해를
줄이면서 전화 구간의 이득을 취했다.

## 1. hard router와 soft MoE 직접 비교

동일한 frozen prediction을 사용하여 clean/phone/mixed 외부 뱅크 6개와
factorial dev/holdout, phone factorial, YuE 등 총 10개 뱅크를 평가했다.

| 전략 | 평균 ADS | fixed MoE 대비 평균 변화 | 최악 뱅크 변화 |
|---|---:|---:|---:|
| fixed soft MoE | 0.834580 | 0 | 0 |
| phone-domain soft weight | 0.837591 | +0.003010 | -0.001667 |
| confidence hard expert | 0.837772 | +0.003192 | -0.019715 |
| phone hard expert | 0.815343 | -0.019237 | -0.054333 |
| content hard expert | 0.813770 | -0.020810 | -0.073727 |

hard routing은 평균이 좋아 보이는 설정도 한 뱅크에서 약 0.02 ADS를 잃었다.
학습된 content router는 손실이 더 컸다. Phone/non-phone 가중치를 10개 뱅크에
맞추면 평균 약 +0.0021이었지만 leave-one-bank-out에서는 평균 이득이
약 +0.00002로 사라졌고 최악 holdout은 -0.01325였다. 따라서 단순 domain별
가중치 튜닝도 일반화 근거가 부족하다.

관련 결과:

- `reports/router_training_v1/joint_router_vs_moe/summary.csv`
- `reports/router_training_v1/domain_router_allbanks_sweep.csv`
- `reports/router_training_v1/domain_router_lobo.csv`

## 2. 분리 없는 temporal-attention 전문가

SPEAR 원본 오디오 표현을 3개 view와 8개 시간 bin으로 구성하고, Transformer
encoder가 총 24개 token을 함께 보도록 했다. 각 bin에서 Voice/Music presence와
fake를 예측하고 component-aware LME로 파일 확률을 모은다. Demucs/SAM-Audio
stem을 이 전문가의 입력으로 쓰지 않으므로 분리 모델이 생성한 artifact에
의존하지 않는다.

학습 데이터는 모두 별도의 train partition이며 factorial/phone/YuE/Suno 감사
세트는 학습과 checkpoint 선택에서 제외했다. 네 가지 규제를 같은 seed로
비교했다.

| attention 학습 | 내부 선택 점수 | factorial ADS | phone ADS | YuE ADS |
|---|---:|---:|---:|---:|
| 규제 없음 | 0.791755 | 0.730216 | 0.835500 | 0.796658 |
| 성분+채널 0.1 | 0.786883 | 0.724494 | **0.840214** | 0.779661 |
| 채널만 0.1 | **0.793255** | 0.728260 | 0.832429 | **0.798325** |
| 성분+채널 0.02 | 0.788714 | 0.725463 | 0.836679 | 0.790967 |

attention 단독 모델은 기존 MoE를 대체할 정도로 강하지 않았다. 다만 기존
모델과 오류가 달라 작은 residual expert로 사용할 때 전화 구간을 보완했다.
성분 counterfactual 일관성은 독립적인 Voice/Music 판별을 지나치게 묶어 전체
일반화를 떨어뜨렸지만, 전화 파일의 File 판별에는 보완성이 있었다.

## 3. 최종 residual/router ablation

먼저 아직 leaderboard에서 검증하지 않은 v30을 anchor로 사용하면 성분+채널
일관성 attention의 non-phone 12.5%, phone 20% 결합이 dev `+0.0038`, phone
`+0.0238` ADS였고 factorial/YuE는 동률이었다. 그러나 이 제출은 v29와 v30의
변경까지 누적되어 실제 점수 변화의 원인을 분리할 수 없다.

따라서 최종 선택은 실제 제출 최고점 v18을 frozen prediction에서 정확히
재구성한 뒤 다시 했다. 네 attention head 중 내부 선택 점수도 가장 높은
채널 일관성-only head를 File logit에만 결합했다.

| 결합 방식 | dev 변화 | factorial 변화 | phone 변화 | YuE 변화 |
|---|---:|---:|---:|---:|
| 전 구간 20% | +0.0285 | +0.0200 | +0.0379 | +0.0132 |
| router: non-phone 20%, phone 25% | **+0.0324** | **+0.0200** | **+0.0400** | **+0.0132** |

router가 없는 고정 soft MoE도 네 뱅크 모두 좋아졌고, router는 factorial/YuE를
바꾸지 않으면서 전화가 포함된 dev와 phone bank만 추가 개선했다. 따라서 이
결과는 전화 router에 의존해 만들어진 우연한 평균 상승이 아니다. 더 큰 비중도
일부 로컬 조합에서는 좋았지만 unseen generator/channel의 위험을 줄이기 위해
20/25%로 제한했다.

## 4. 기각한 음악 codec-token 전문가

X-Codec 토큰의 unigram/transition probe와 작은 CoMoE형 Transformer도
실험했다. 약 6천 개 music-present 샘플 규모에서는 factorial EER 약 0.40,
source-disjoint 약 0.51, YuE 약 0.55로 기존 SPEAR 음악 전문가보다 나빴다.
Suno 13개도 전부 탐지하지 못했다. 논문 규모의 대규모 학습 코퍼스 없이 해당
방법을 제출 앙상블에 넣으면 generator 일반화가 악화된다고 판단해 제외했다.

## 5. 제출 후보와 남은 위험

- 후보: `v18_channel_attention_router_v32.zip`
- 변경 범위: 실제 성공 제출 v18 대비 1.3MB attention head 하나만 추가
- 추론 시간 증가: 기존 두 번째 SPEAR pass에서 temporal bin을 함께 추출하므로
  backbone pass는 늘지 않고 작은 head forward만 추가
- CPS: 의도적으로 변경하지 않음
- 검증: 전체 test suite 91개 통과, clean/phone/mixed 3파일 전체 스모크 통과

v32의 로컬 결과는 router를 작은 보정기로 쓰는 것이 hard expert selection보다
안전하다는 근거를 제공한다. v18과의 유일한 확률 변경은 File attention 결합이며
Voice/Music/CPS는 그대로이므로, 실제 제출 결과가 바뀌면 File EER 변화로 직접
해석할 수 있다. 이후에는 leaderboard 결과를 기준으로 로컬 File 개선폭의 전이율을
계산해야 한다.
