# 원본 혼합음 SPEAR Voice MIL v53

2026-09-04 기준. 이 실험은 전화/Opus에서 음성 진위를 보강하기 위해 원본 혼합
오디오의 SPEAR 시간 특징만 사용하는 작은 Voice 전문가를 학습한 것이다. 결론부터
말하면, authorized development에서는 좋아졌지만 동결 후 v5/v6에서 재현되지 않아
**최종 제출에는 넣지 않는다**. 실패 결과까지 보존하는 이유는 같은 계열 residual을
다시 크게 섞는 실험을 막기 위해서다.

## 설계

분리 stem에는 손대지 않았다. v52가 이미 한 번 계산하는
`marcoyang/spear-xlarge-speech-audio-v2`의 원본 혼합음 시간 bin을 재사용하므로
추가 SSL backbone pass는 0회다.

각 시간 bin에서 작은 MLP가 두 값을 낸다.

- 해당 bin에 음성이 존재할 확률
- 해당 bin의 음성이 가짜일 logit

첫 번째 값으로 두 번째 값을 가중한 log-mean-exp MIL을 수행한다. 그래서 순차 혼합은
음성 구간만 보고, overlap 혼합에서는 음악이 큰 구간의 영향을 줄이는 것이 의도다.
학습 시 정확한 음성 범위가 있는 합성 mixture에서는 local presence/fake loss를 함께
준다. 최종 후보는 seed와 clean↔codec consistency 강도가 다른 세 head의 logit 평균을
기존 XLS-R 중심 Voice 점수에 15%만 residual로 결합했다.

## 데이터 누수 방지와 선택 규칙

학습 17,760개는 기존 허가된 train partition만 사용했다. ASVspoof/EchoFake와 최근
11개 TTS generator를 dataset 및 generator-label cell 단위로 균형 sampling했다.
정확한 clean↔codec pair 4,000개에는 logit consistency를 적용했다.

모델·seed·weight 선택은 다음 authorized development 4,680개에서만 수행했다.

- clean/mixed/telephone source-disjoint banks
- telephone mixed development
- factorial train의 별도 development split
- multigen voice와 temporal mixture의 별도 development split

`codec_mixed_dev_v4`, blind v5/v6/v7은 train, early stopping, member 선택, weight
선택에 사용하지 않았다. `0.15` weight와 세 checkpoint를 먼저
`configs/spear_voice_mil_v53_frozen.yaml`에 동결한 뒤 v4/v5/v6만 각각 한 번
회고 평가했다. v7은 inference도 score도 하지 않았다.

## Authorized development 결과

세 head를 uniform-logit ensemble하고 기존 anchor에 15% 결합했을 때:

| 지표 | anchor | v53 |
|---|---:|---:|
| robust selection | 0.758804 | **0.783572** |
| dataset 평균 Voice EER | 0.194987 | **0.174398** |
| dataset 최악 Voice EER | 0.320000 | **0.280000** |
| generator 평균 Voice EER | 0.203310 | **0.180979** |
| generator 최악 Voice EER | 0.397674 | **0.394817** |
| telephone 평균 Voice EER | 0.238889 | **0.202778** |
| channel 최악 Voice EER | 0.310345 | 0.310345 |

그러나 채널별로는 telephone/noisy가 개선되는 대신 MP3와 stereo가 악화됐다.
`0.15`는 최악 channel EER를 악화시키지 않은 범위에서 선택한 보수적 weight였지만,
모든 채널에서 개선되는 전문가는 아니었다.

### 미지 TTS LOGO

11개 TTS generator를 세 fold로 나누어 해당 fold generator를 전부 제외하고 다시
학습했다. held-out generator Voice EER의 fold 평균/최댓값은 다음과 같다.

| fold | 평균 EER | 최악 EER |
|---|---:|---:|
| A | 0.180676 | 0.234835 |
| B | 0.217844 | 0.262202 |
| C | 0.172180 | 0.193029 |

즉 TTS generator shortcut만 외운 모델은 아니지만, 완전한 generator 불변성도
달성하지 못했다. 특히 LLaSA `0.2622`, OpenVoice-V2 `0.25`, FireRedTTS
`0.2348`이 남은 약점이다.

## 동결 후 v4/v5/v6 회고 진단

아래 수치를 보기 전에 architecture, checkpoint 세 개, Voice logit weight 0.15를
모두 동결했다. 결과를 본 뒤 재튜닝하지 않았다.

| bank | anchor Voice EER | v53 Voice EER | anchor ADS | v53 ADS |
|---|---:|---:|---:|---:|
| codec mixed v4 | 0.20667 | **0.20000** | 0.76600 | **0.76733** |
| blind v5 | **0.11667** | 0.13333 | **0.81333** | 0.81000 |
| blind v6 | **0.08333** | 0.10000 | **0.78917** | 0.78583 |

v4에서는 clean/G.722와 partial/sequential FR이 개선됐지만, 목표였던 Opus EER는
`0.35→0.3667`, G.711→Opus transcode는 `0.3167→0.35`로 악화됐다. v5와 v6도
Opus/transcode가 일관되게 악화됐다. v6에서는 clean Voice EER가 0인 강한 anchor에
G.711 false ordering까지 새로 추가됐다.

이 패턴은 “codec-invariant하게 학습한 작은 head”가 곧 “codec 때문에 사라진
forensic artifact를 복원하는 head”는 아니라는 뜻이다. SPEAR의 speech/audio semantic
표현은 음성 위치를 찾는 데 유용하지만, 압축 후 남는 진위 artifact의 순위는 기존
XLS-R anti-deepfake anchor보다 불안정했다. authorized telephone 개선만으로 blind
telephone 일반화를 보장할 수 없었다.

## 최종 판단과 다음 실험 조건

v53 Voice residual은 **배포 거부**다. v52의 네 output과 최종 File consistency 흐름에
추가하지 않는다. 특히 blind 결과를 보고 weight를 15%에서 더 작게 고르는 행위도
하지 않는다.

후속 Voice 보강은 다음 조건을 동시에 만족할 때만 채택하는 편이 안전하다.

1. 기존 XLS-R-2B Voice anchor를 유지하고 새 forensic 모델은 매우 낮은 weight의
   독립 evidence로만 사용한다.
2. clean뿐 아니라 G.711, G.722, narrowband Opus, G.711→Opus 각각에서 비악화
   제약을 selection objective에 직접 넣는다. 평균 phone EER만 보면 안 된다.
3. RR 대 FR, RF 대 FF를 concurrent/partial/sequential별로 확인한다.
4. 미지 TTS LOGO의 최악 generator와 최악 codec의 교차 조건을 별도 holdout으로 둔다.
5. 새 bank 결과를 본 뒤 weight를 조정하지 않고, 한 번 동결한 후보만 평가한다.

## 재현 파일

- 모델: `src/spear_voice_mil_head.py`
- 추론/Voice-only residual: `src/spear_voice_mil_inference.py`
- 학습/LOGO: `scripts/train_spear_voice_mil.py`
- 동결 후 one-shot 평가: `scripts/evaluate_spear_voice_mil_frozen.py`
- 동결 명세와 checkpoint hash: `configs/spear_voice_mil_v53_frozen.yaml`
- 테스트: `tests/test_spear_voice_mil.py`
- authorized reports 및 frozen diagnostic: `reports/spear_voice_mil_v53/`

세 배포 후보 checkpoint는 각각 약 915 KiB라 용량 문제는 없다. 구현 테스트는
presence routing, loss, balanced sampling/codec pairs, Voice 이외 네 output 보존을
검증한다.
