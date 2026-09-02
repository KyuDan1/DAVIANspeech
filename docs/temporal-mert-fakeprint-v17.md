# Temporal + MERT + modern fakeprint v17 연구·배포 기록

2026-09-02 기준. 이 후보는 실제 최고점 `lme_spear_v1`의 구조를 유지하면서,
서로 다른 실패 원인을 보는 세 종류의 **원본 오디오 expert**를 작은 비중으로
결합한다. 새 expert들은 stem을 만들지 않지만 실제 최고점의 legacy XLS-R anchor는
HTDemucs voice stem을 계속 사용한다. 따라서 전체가 완전 무분리인 것은 아니며,
원본 branch가 분리 과정에서 약해질 수 있는 단서를 보완한다. 전화 채널에는 ADS
hard route를 별도 적용하지 않는다.

## 1. 현재 기준점과 이번 변경

| 항목 | 실제 리더보드 |
|---|---:|
| 현재 최고 `lme_spear_v1` Total | 0.7444743 |
| 현재 최고 ADS | 0.7172857 |
| 현재 최고 CPS | 0.9891721 |
| 1위 Total / ADS / CPS | 0.82289 / 0.80354 / 0.99707 |

기존 probe로 추정한 현재 EER은 File 약 `0.2741`, Voice 약 `0.2156`, Music 약
`0.3714`다. 가장 큰 병목은 Music이고, 특히 실제 음성+가짜 음악(RF), 가짜
음성+실제 음악(FR), 부분 중첩·순차 혼합에서 한 성분의 단서가 다른 성분 점수에
전파되는 현상이 문제다.

v17의 실제 추론 순서는 다음과 같다.

1. 실제 검증된 XLS-R-2B `logmeanexp(T=5)` anchor
2. EAT/PANNs CPS와 SPEAR 원본-mixture expert
3. separation-free temporal MIL: File/Voice/Music 각각 logit `0.05`
4. MERT/SOFIA music expert: File `0.025`, Music `0.0125`
5. Suno/Udio spectral fakeprint: File/Music 각각 `0.025`

Voice fake score에는 music expert를 넣지 않았다. File은 대회 정의상 Voice 또는
Music 중 하나만 fake여도 fake이므로 두 music expert의 약한 증거를 받는다.

## 2. 전화 데이터 현황과 결론

`phone_factorial_1200_v1`은 학습에 사용하지 않는 고정 평가 bank다. 동일한 clean
원천 300개에 네 가지 전화 변형을 적용해 총 1,200개를 만들었다.

- 8 kHz resampling
- G.711 μ-law
- G.726 24 kbps
- Opus narrowband 8 kHz

voice-only, music-only, mixed가 각각 400개이며 RR/RF/FR/FF 조합을 따로 평가한다.
전화 router는 1,200/1,200을 전화로 검출했고 대응 clean 300개는 0/300만 전화로
오탐했다. 즉 router 자체보다 **route 이후의 score calibration**이 문제였다.

| 전화 평가 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| 기존 anchor | 0.3100 | 0.1825 | 0.4175 | 0.68936 |
| temporal MIL 결합 | - | - | - | 0.70339 |

현대 음악 fakeprint의 코덱별 Music EER은 G.711 `0.12`, G.726 `0.19`, 단순
8 kHz `0.11`, Opus-NB `0.405`였다. 전화라는 조건 전체가 어려운 것이 아니라,
Opus 손실압축이 생성 artifact를 지우거나 새로운 codec artifact를 만드는 경우가
핵심 병목이라는 뜻이다.

전화 전용 dual-domain 점수를 강하게 적용하면 전화 ADS는 `0.73575`까지 올랐지만,
일반 Factorial ADS가 `0.72538 → 0.71564`로 내려갔다. EER은 평가 파일 전체의
상대 순위를 사용하므로 전화 subset만 재보정하면 clean/phone 사이 순위가 틀어진다.
따라서 v17의 전화 router는 검증된 Voice Presence 10% 보정에만 사용하고 ADS에는
hard route를 사용하지 않는다.

## 3. 새 expert 1: MERT/SOFIA

MERT는 speech 전용 representation과 다른 music-aware SSL 축을 제공한다. 원본
mixture를 그대로 입력하므로 separator가 생성 artifact를 새로 섞을 위험이 없다.
기존 SOFIA/MERT 공개 checkpoint를 오프라인 패키지에 포함했고, 세 독립 audit에서
File `0.10`, Music `0.05` 결합이 모두 양의 방향이었다.

| audit | temporal 기준 | 강한 MERT 결합 | 변화 |
|---|---:|---:|---:|
| Factorial holdout | 0.72195 | 0.74372 | +0.02177 |
| Phone factorial | 0.69475 | 0.70050 | +0.00575 |
| YuE generator | 0.75116 | 0.76712 | +0.01596 |

실제 leaderboard에서 큰 local-optimal weight가 역전된 경험이 있어 배포 weight는
그보다 4배 작은 File `0.025`, Music `0.0125`로 제한했다. 이 보수적 결합에서도
Factorial `0.72195 → 0.72496`, Phone `0.69475 → 0.69561`, YuE
`0.75116 → 0.77197`로 방향이 유지됐다.

## 4. 새 expert 2: 현대 AI 음악 spectral fakeprint

[`lofcz/ai-music-detector`](https://github.com/lofcz/ai-music-detector)의 MIT 공개
가중치를 사용했다. 이 모델은 FMA real과 SONICS의 Suno/Udio fake로 학습된 작은
spectral logistic classifier다. 16 kHz 원본 오디오에서 1--8 kHz 평균 spectrum과
lower-hull residual을 계산하므로 GPU, separator, 추가 다운로드가 필요 없다.

공개 ONNX는 float32 sigmoid가 극단 점수에서 정확히 0/1로 포화됐다. v17은 공개
`weights.npz`에서 float64 pre-sigmoid margin을 직접 계산하고 최종 logit 결합만
안전 범위로 clip한다. 공식 구현과 feature 평균 절대 오차는 `4.5e-5`였다.

| standalone 평가 | File EER | Music EER |
|---|---:|---:|
| external mixed | 0.4300 | 0.3900 |
| factorial 1,200 | 0.3870 | 0.3457 |
| YuE | 0.3084 | 0.2419 |
| phone 1,200 | 0.3873 | 0.2100 |

단독 SoTA 모델은 아니지만 speech SSL과 오류 상관이 다른 보조 expert다. 사용자가
추가한 Suno vocal 음악 13개는 13/13을 fake로 판정했고 최소 확률은 `0.99957`였다.
MERT 뒤에 File/Music 각각 2.5%만 결합했을 때 최종 ADS는 Factorial `0.72457`,
Phone `0.69982`, YuE `0.77197`이었다. Factorial에서는 MERT-only보다 `0.00039`
낮지만 temporal 기준보다는 높고, 전화와 YuE에서 동시에 좋아져 일반성 우선으로
채택했다.

## 5. 검토 후 제외한 XLSR-SLS

[`SpeechAntiSpoofingBenchmarks/XLSR-SLS`](https://huggingface.co/SpeechAntiSpoofingBenchmarks/XLSR-SLS)는
ASVspoof speech dev에서 EER `0.0`이었지만 다음 미관측 조건에서 일반화하지 못했다.

| 평가 | EER 또는 관찰 |
|---|---:|
| source-disjoint mixed Voice | 0.14--0.15 |
| phone raw Voice | 0.3567--0.3733 |
| source-disjoint music | 0.315--0.355 |
| Qwen3/F5/CosyVoice multigen Voice | 0.567--0.625 |

HTDemucs voice stem을 사용해도 external mixed Voice EER가 `0.425 → 0.430~0.435`로
개선되지 않았다. 오래된 speech spoof corpus에는 강하지만 현대 TTS, 음악 혼합,
전화 codec에서 score 방향까지 바뀌어 제출 ensemble에서는 제외했다. 이는 한
ASVspoof benchmark의 고성능을 대회 일반성으로 바로 해석하면 안 된다는 근거다.

## 6. 데이터 누수와 평가 원칙

- 전화 1,200, Factorial 1,200, external mixed, YuE, 사용자 Suno는 모두 평가 전용
- temporal head 학습 bank와 speaker/music/source ID가 겹치지 않도록 group split
- RR/RF/FR/FF와 concurrent/partial/sequential/sparse를 각각 보고 전체 평균으로
  약점을 숨기지 않음
- 실제 비공개 test에서는 파일별 독립 추론만 수행하며 test 통계로 학습·보정하지 않음
- 새 temporal/MERT/fakeprint expert는 원본 오디오를 입력받음
- 실제 최고점 legacy XLS-R branch의 HTDemucs stem은 유지하되 새 separator는 추가하지 않음

## 7. 배포 검증

후보 디렉터리는 `temporal_mert_fakeprint_v17/`이다.

- clean mixed 1개 + Opus-NB phone mixed 1개 end-to-end smoke 통과
- 전화 router가 두 파일 중 phone 1개만 선택
- 출력 5개 확률이 모두 finite이며 `[0, 1]`
- EAT/SPEAR 임시 통계 파일은 종료 후 삭제
- 전체 unit/integration test: `48 passed`
- `torch`/`torchaudio`를 requirements에서 재설치하지 않음
- `onnxruntime-gpu==1.23.2` 유지
- 외부 인터넷 다운로드 없음
- ZIP member 103개, 중복 0개, CRC 오류 없음
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- ZIP `7,904,643,592` bytes, 압축 해제 `7,904,622,864` bytes
- 최대 member `2,387,980,808` bytes
- SHA-256: `0d19db953ae5d08bfb232315f097b8361cb87b197b8e0afda564ad1872cd3a45`

이번 후보의 핵심 판단은 “전화면 다른 detector로 바꾼다”가 아니라, 채널에 덜
종속적인 원본-mixture expert들의 낮은 가중치 합의로 Opus와 현대 음악 생성기의
실패를 동시에 줄이는 것이다. 다만 로컬 개선 폭을 실제 점수로 간주하지 않고,
v16/v17 실제 결과를 받은 뒤 각 추가 expert의 인과를 다시 좁혀야 한다.
