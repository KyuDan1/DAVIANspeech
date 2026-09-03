# Router vs. MoE 실험과 v32 후보

## 결론

샘플 전체를 하나의 전문가로 보내는 hard router는 사용하지 않는다. v32/v33의
실제 점수와 새 latent-attention router의 locked audit까지 반영하면 현재 가장
안전한 구조는 다음과 같다.

1. 실제 리더보드 최고점인 v18을 그대로 anchor로 사용한다.
2. 분리 없이 원본 EAT/SPEAR 통계를 보는 channel/component-invariant head 네 개를
   동일 비중으로 투표한다.
3. 이 네-head 평균을 File/Voice/Music에 같은 30%로 낮게 결합한다.
4. 전문가 하나를 선택하는 hard route와 phone/content별 큰 가중치 변경은 하지
   않는다.
5. `VOICE_PRESENT_PROB`, `MUSIC_PRESENT_PROB`는 v18과 동일하게 유지한다.

즉, 현재 제출 후보는 router가 기존 모델을 교체하는 구조가 아니라 독립 seed와
규제를 가진 전문가의 합의를 더하는 구조다. 중간 hidden representation을 보는
attention gate도 직접 학습했지만 locked cross-component에서 역전되어 제출에서는
제외했다.

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
- ZIP: 7,053,527,053 bytes, 해제 7,909,100,324 bytes, 112개 엔트리
- ZIP 구조/CRC: 중복 0, 오류 0, 최상위 `model/`, `script.py`,
  `requirements.txt`만 존재
- SHA-256: `69d9f3a02c1c2b59ade25a4b7e2f1afe14ffd5e55fb74c41a407043ec6fcc94a`

v32의 로컬 결과는 router를 작은 보정기로 쓰는 것이 hard expert selection보다
안전하다는 근거를 제공한다. v18과의 유일한 확률 변경은 File attention 결합이며
Voice/Music/CPS는 그대로이므로, 실제 제출 결과가 바뀌면 File EER 변화로 직접
해석할 수 있다. 이후에는 leaderboard 결과를 기준으로 로컬 File 개선폭의 전이율을
계산해야 한다.

## 6. 실제 제출과 v33 Music residual

2026-09-04 KST 자정에 v32를 `v32_router.zip`이라는 짧은 별칭으로 API
제출했다. DACON API의 파일명 길이 제한 때문에 원래 긴 파일명은 업로드 전에
거절됐고, 동일 inode/SHA-256 파일을 짧은 이름으로 다시 보내 성공 응답을
확인했다. 실제 채점 결과는 총점 `0.7616379312`, ADS `0.7363412698`, CPS
`0.9893078836`으로 v18과 정확히 같았다. 따라서 로컬에서 관찰한 File EER 개선은
hidden test의 EER 교차점 순위를 바꾸지 못했다. 이후 후보의 기대값에서는 File
attention의 로컬 개선폭을 제외하고 Voice/Music의 독립 전이만 보수적으로 본다.

그동안 exact-v18 anchor에 기존 SPEAR temporal-bin Music head를 독립적으로
결합했다. 이 head는 eval bank에 학습하지 않았으며 원본 오디오의 시간축
representation만 사용한다. `MUSIC_FAKE_PROB` 외의 열은 수정하지 않는다.

| 후보 | dev ADS 변화 | factorial 변화 | phone 변화 | YuE 변화 |
|---|---:|---:|---:|---:|
| File router만(v32) | +0.0324 | +0.0200 | +0.0400 | +0.0132 |
| Music 50%만 | +0.0240 | +0.0120 | +0.0503 | +0.0097 |
| File router + Music 50%(v33) | **+0.0564** | **+0.0320** | **+0.0903** | **+0.0229** |

Music 가중치 0.6과 전화 구간 0.7이 로컬 평균은 가장 높았지만, 실제 제출에서
큰 보조 가중치가 역전된 전례가 있어 v33은 전 구간 0.5로 제한했다. 세 seed를
확률/로짓 평균한 voting도 phone에서는 약간 개선됐으나 factorial에서는 가장
좋은 단일 seed보다 나빠 채택하지 않았다. 따라서 현재 결론은 다음과 같다.

1. hard expert selection은 사용하지 않는다.
2. 검증된 v18 soft MoE를 anchor로 보존한다.
3. File에는 phone router로 20/25% residual만 허용한다.
4. Music에는 domain router 없이 독립 residual 50%를 적용한다.

준비된 다음 후보는 `v18_attention_music_v33_fixed.zip`이다. 기존 SPEAR pass가
생성한 statistics를 두 head가 공유하므로 backbone 호출은 늘지 않는다. ZIP은
7,054,152,976 bytes, 해제 약 7.91GB, 116개 엔트리이며 중복 0, 전체 CRC 오류
0이다. 전체 테스트 92개와 clean/phone/mixed 3파일 GPU 스모크를 통과했다.
SHA-256은
`319abd23c9f739f9275d066ed6a82dc18d1705b50c104f5fb831c8c188f4b8a9`이다.

## 7. v34: router 대신 channel/component-invariant soft MoE

v33 다음에는 EAT와 SPEAR의 이미 계산된 통계를 함께 읽는 네 개의
channel/component-invariant head를 비교했다. 네 head는 서로 다른 seed와 규제를
사용하지만 별도 backbone pass는 필요하지 않는다. 학습에는 train partition만
사용했고 factorial dev/holdout, phone factorial, YuE audit는 학습과 checkpoint
선택에서 제외했다.

전화 파일에만 결합 비중을 높이는 router도 직접 비교했다. 전 구간 10% 결합과
non-phone 10%/phone 12.5% 결합의 ADS 변화는 다음과 같았다.

| 결합 | dev | factorial | phone | YuE |
|---|---:|---:|---:|---:|
| 고정 10% MoE | +0.0090 | +0.0057 | +0.0400 | +0.0151 |
| 전화 router 10/12.5% | +0.0061 | +0.0057 | +0.0433 | +0.0151 |

router는 phone에서 `+0.0034`를 더 얻는 대신 dev에서 `-0.0029`를 잃었다. 평균
이득은 거의 같고 unseen test의 전화 비율에 의존하므로 고정 soft MoE를 선택했다.
출력축별 maximin 탐색에서는 File/Voice/Music weight `0.125/0.20/0.10`이 네
평가축 모두 양의 방향이었다.

| 평가축 | v33 ADS | v34 ADS | 변화 |
|---|---:|---:|---:|
| dev | 0.754286 | 0.763247 | +0.008961 |
| factorial holdout | 0.777506 | 0.785506 | +0.008000 |
| phone factorial | 0.824107 | 0.866429 | +0.042321 |
| YuE cross-component | 0.851373 | 0.871598 | +0.020225 |

특히 phone의 File/Voice/Music EER은 각각
`0.1783/0.1675/0.1775 → 0.1341/0.1450/0.1250`으로 낮아졌다. 현재 가장 어려운
`fake voice + real music`과 real-real의 File EER도 factorial에서
`0.3733 → 0.3467`, phone에서 `0.2700 → 0.2600`으로 작게 개선됐다. 다만 YuE의
Voice EER은 일부 악화되어 큰 가중치나 hard selection은 사용하지 않았다.

따라서 현재 architecture 원칙은 다음과 같다.

1. domain/content router가 base expert를 교체하지 않는다.
2. 원본 mixture의 latent token 내부에서는 component-presence attention을 soft
   router로 사용한다.
3. 파일 수준에서는 독립 seed와 representation을 낮은 가중치로 투표한다.
4. 전화 router는 여러 평가축에서 손실이 없는 residual의 작은 비중 조절에만 쓴다.

네 invariant member 사이의 routing도 별도로 확인했다. audit 4개만 최대화하면
`fixedteacher/ch01/seed01/seed02 = 0/0.125/0.125/0.75`가 uniform보다 v33 residual의
최악 개선폭을 `+0.0080 → +0.0115`로 올렸다. 그러나 이 비율은 별도의 일반화
개발군 6개 중 5개에서 uniform ensemble보다 나빴고, 최악 손실은 `-0.0104 ADS`였다.
특히 telephone mixed dev가 `0.8118 → 0.8013`으로 떨어져 phone router의 근거도
되지 못했다. 1/8 단위의 모든 nonnegative 4-member 조합 165개 중 여섯 개발군에서
uniform보다 하나도 나빠지지 않은 조합은 `0.25/0.25/0.25/0.25`뿐이었다. 따라서
v34는 member router 없이 균등 투표를 유지한다.

배포 후보 `v34_invariant.zip`은 압축 7,065,350,643 bytes, ZIP 내부 해제
7,922,543,971 bytes, 121개 엔트리다. 최상위 구조와 중복 검사를 통과했고 전체
CRC 오류는 없다. clean/Opus narrow-band phone/mixed 3파일 CUDA smoke를 통과했으며
CPS는 v33과 동일하고 ADS 세 열만 의도대로 달라졌다. SHA-256은
`61ae6380ca33978476e97830ff48d968f1a7d6e4ff84cee2fea99661b507df34`이다.

## 8. v35 후보: component-to-File 구조적 MoE

대회 정의의 `File fake = present component 중 하나라도 fake` 관계를 이용해 v34
File logit과 `max(Voice fake logit, Music fake logit)`을 50% 결합했다. CPS를
가중하지 않은 이유는 absent component의 presence 오차가 File에 전파되는 경로를
추가하지 않기 위해서다. 이 후보는 다른 네 출력은 건드리지 않는다.

| 방식 | dev 변화 | factorial 변화 | phone 변화 | YuE 변화 | 최악 변화 |
|---|---:|---:|---:|---:|---:|
| 전 파일 고정 결합 | +0.0076 | +0.0085 | +0.0059 | +0.0132 | **+0.0059** |
| voice-dominant soft router | +0.0115 | +0.0038 | +0.0062 | +0.0160 | +0.0038 |

soft router의 평균은 약간 높지만 고정 결합의 maximin이 더 좋았다. 다만 세부
셀에서는 고정 결합이 `fake voice + real music`을 크게 개선하는 대신 일부
`real voice + fake music`과 music-only를 악화시켰다. 따라서 v35는 준비만 하고
v33/v34의 실제 leaderboard 전이를 확인하기 전에는 제출하지 않는다.

최종 수정 패키지 `v35_consistency_fixed.zip`은 누락 모듈을 명시적으로 포함한다.
압축 7,065,351,996 bytes, ZIP 내부 해제 7,922,547,421 bytes, 122개 엔트리이며
최상위 구조, 중복, 전체 CRC 검사를 통과했다. clean/phone/mixed CUDA smoke에서도
성공했고 v34 대비 File만 변경되는 것을 확인했다. SHA-256은
`8b0f90ec00281bdfbf8bd4af468ee1483c260281cbae6ab65f5763fbb18cd4d2`이다.

## 9. v33 실제 점수와 clean v18+inv4 후보

v33도 실제 채점에서 총점 `0.7616379312`, ADS `0.7363412698`, CPS
`0.9893078836`으로 v18 및 v32와 정확히 같았다. 즉 다음 두 변경은 로컬에서는
좋았지만 hidden test의 EER 교차 순서를 바꾸지 못했다.

- v32: SPEAR temporal attention을 File에 20/25% route
- v33: SPEAR temporal-bin Music expert를 50% 추가

따라서 다음 실제 제출은 이 두 neutral 변경을 모두 제거하고 exact v18에 새
paired invariant ensemble만 추가했다. 새 v4 ensemble 단독 성능은 과거 v19의
3-head ensemble보다 모든 주요 audit에서 높았다.

| 평가축 | 과거 v19 | 새 paired v4 |
|---|---:|---:|
| factorial holdout ADS | 0.6967 | 0.7716 |
| phone factorial ADS | 0.6869 | 0.8716 |
| YuE cross-component ADS | 0.8023 | 0.8327 |
| source-disjoint equal ADS | 0.7910 | 0.8490 |
| telephone mixed dev ADS | 0.7980 | 0.8118 |

축별 로컬 최대점은 File/Voice/Music `0.75/0.10/0.75`였지만, phone speech-only
File EER이 `0.05 → 0.12`로 악화되고 YuE의 fake-voice/real-music 및 dev
music-only도 역행했다. 그래서 같은 30%를 쓰는 `0.30/0.30/0.30`을 선택했다.
exact v18 대비 ADS 변화는 dev `+0.0474`, factorial `+0.0336`, phone
`+0.1399`, YuE `+0.0326`으로 네 축 모두 양수였다.

최종 제출물은 `v18_inv4_w30.zip`이다. 압축 7,063,621,168 bytes, ZIP 내부
해제 7,920,458,124 bytes, 112개 엔트리이며 최상위 구조·중복·전체 CRC와
clean/Opus narrow-band/mixed CUDA smoke를 통과했다. SHA-256은
`573eaedd503d5f300b88a8ee77c2bba672689a22bd201205b11f1732a711df61`이다.
2026-09-04 KST에 DACON API 성공 응답을 확인했다. 실제 채점도 총점
`0.7616379312`, ADS `0.7363412698`, CPS `0.9893078836`으로 v18/v32/v33과
정확히 같았고 공개 행 시간은 `2026-09-04 01:30:48 KST`로 갱신됐다. 로컬에서
네 audit 축이 모두 크게 좋아졌어도 hidden EER 순위는 하나도 바뀌지 않았다.
따라서 후속 후보는 이 EAT/SPEAR residual의 가중치를 더 조절하지 않고, 다른
representation과 학습 원천을 가진 새 전문가에서 시작해야 한다.

## 10. 중간 표현 attention router 직접 비교

관련 연구에는 서로 다른 결론이 모두 존재한다.

- [Hidden-Domain Routing for All-Type Audio Deepfake Detection](https://arxiv.org/abs/2608.00493)은
  먼저 BEATs로 speech/sound/singing/music 중 하나를 고르고 Speech-XLSR 또는
  EAT branch로 보내 AT-ADD Track2 1위를 기록했다. 이 설정은 네 audio type이
  상호 배타적이라는 점이 중요하다.
- [Attention-based Mixture of Experts for Robust Speech Deepfake Detection](https://arxiv.org/abs/2509.17585)은
  각 detector의 마지막 hidden embedding을 token으로 보고 Transformer gate가
  모든 expert를 서로 attention한 뒤 soft weight를 만든다. 서로 다른 architecture를
  pooled data에서 학습하는 방식이 unseen domain EER을 개선했다.
- [Harder or Different?](https://www.isca-archive.org/interspeech_2024/muller24b_interspeech.html)는
  unseen fake의 성능 저하가 단순히 더 어려워져서가 아니라 generator/domain의
  차이에서 주로 온다고 보고했다. 따라서 router도 보지 못한 domain에서는 새로운
  실패점이 될 수 있다.

우리 문제에서는 한 파일이 speech-only, music-only, sequential mixture,
overlapping mixture 중 하나일 수 있어 첫 논문의 상호 배타적 hard route를 그대로
적용할 수 없다. 대신 새 `BoundedAttentionRouter`를 구현했다.

1. 네 invariant expert의 Voice/Music/File별 128차원 판정 직전 hidden을 받는다.
2. expert 네 개를 token으로 Transformer self-attention한다.
3. 축별 soft weight를 만들되 uniform `0.25`에서 제한된 범위만 이동한다.
4. train 6개 뱅크만 업데이트에 사용하고 dev 6개로 checkpoint를 선택한다.
5. factorial holdout, phone factorial, YuE는 선택 후 한 번만 연다.

강도 50% 단일 router는 dev 선택 점수가 uniform보다 `+0.00615`였으나 locked
factorial `-0.00496`, YuE `-0.00859` ADS로 역전됐다. 더 보수적인 강도 25%
3-seed router ensemble도 v18에 30% 결합했을 때 아래와 같았다.

| routed 축 | dev | factorial | phone | YuE | 최악 변화 |
|---|---:|---:|---:|---:|---:|
| Music만 | 0.0000 | +0.0017 | 0.0000 | 0.0000 | **0.0000** |
| File만 | +0.0038 | +0.0038 | 0.0000 | -0.0027 | -0.0027 |
| Voice만 | 0.0000 | -0.0023 | +0.0010 | -0.0038 | -0.0038 |
| File+Music | +0.0038 | +0.0055 | 0.0000 | -0.0027 | -0.0027 |
| 전체 | +0.0038 | +0.0032 | +0.0010 | -0.0065 | -0.0065 |

Music-only gate만 손실이 없었지만 실효 개선은 한 축에서 `+0.0017 ADS`뿐이고
나머지 EER은 모두 동률이다. 실제 v32/v33에서 더 큰 로컬 개선도 완전히 동률이었던
점을 고려하면, 이 gate를 넣을 근거가 부족하다. 따라서 현재 선택은 다음과 같다.

- 제출: 네 member의 uniform soft MoE
- 제외: hard route, phone/content expert 교체, learned File/Voice gate
- 보류: Music-only bounded gate; 다른 architecture의 music expert가 추가되어
  보완성이 커질 때 다시 평가

AI가 실제 악기 샘플을 배열하거나 composition만 생성한 음악은 codec artifact만으로
잡히지 않을 수 있다. 이는 router로 해결되는 문제가 아니라 rhythm/chroma/phrase의
장기 구조를 보는 별도 music expert가 필요한 경우다. 그런 expert가 충분히 강해진
뒤에는 `music-only → 장기구조 expert`, `speech-only → XLS-R`, `mixed → soft MoE`
형태의 reject-option router를 다시 검토할 수 있다. 지금처럼 전문가들이 거의 같은
EAT/SPEAR 통계와 구조를 공유할 때는 gate보다 균등투표의 일반성이 높다.

마지막으로 같은 invariant head를 입력 stream별로 강제로 분해해
`Voice→SPEAR-only`, `Music→EAT-only`, `File→joint` routing도 확인했다. 이것은
hidden-domain routing 논문의 branch 구성을 현재 모델에 가장 가깝게 옮긴 실험이다.

| 방식 | dev 평균 변화 | dev 최악 | locked 평균 변화 | locked 최악 |
|---|---:|---:|---:|---:|
| hard axis route | -0.0151 | -0.0290 | -0.0090 | -0.0160 |
| 25% soft axis route | -0.0035 | -0.0180 | -0.0009 | -0.0069 |
| 50% soft axis route | -0.0055 | -0.0190 | +0.0003 | -0.0091 |
| joint uniform MoE | 0 | 0 | 0 | 0 |

v18에 30% 결합한 25% soft route도 factorial `-0.0057`, phone `-0.0065`,
YuE `-0.0097 ADS`였다. EAT와 SPEAR는 각각 music/speech prior가 강하지만 실제
mixture에서는 반대 stream도 authenticity 단서를 제공한다. 따라서 입력 stream을
잘라 전문가를 만드는 방식도 기각한다. 새로운 architecture로 독립 학습한
Speech-XLSR와 Music-EAT처럼 진짜로 다른 전문가가 준비되기 전에는 joint input을
유지한다.
