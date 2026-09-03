# Sparse-call consensus + voice-only File consistency v24

2026-09-03 기준. 대회 주제가 보이스피싱 대응이고 주관 기관이 국립과학수사연구원인
점을 고려해, 단순 codec 변형보다 **실제 통화의 시간 구조**를 우선한 후보다. 공지에는
전화채널이 일부라고만 되어 있으므로 모든 파일을 전화 전용 모델로 hard-route하지는
않는다.

현재 실제 최고점은 `channel_invariant_moe_v18.zip`의 Total
`0.7616379312`, ADS `0.7363412698`, CPS `0.9893078836`이다. v24는 이 경로에 v22
Spectra stem expert, 시간 구간 합의, File/Voice 일관성만 순서대로 추가한다.

## 1. 평가 데이터

### 학습용 call bank

`forensic_call_train_v1`은 576개 clean parent call이다.

- 두 화자의 교대 발화
- 두 화자의 길이가 비슷한 `balanced_turns`와 두 번째 화자가 5--9%만 차지하는
  `sparse_second_speaker`
- Voice case `RR/RF/FR/FF`
- 음악 `absent/real/fake`, background 또는 hold 구간
- 최종 혼합 뒤 G.711 μ-law, G.722, narrowband Opus, G.711→Opus 재인코딩
- 각 4초 창에 파일 label이 아니라 실제 `VOICE_FAKE_RANGES`와 겹치는 시간을 사용

원천 speaker/music group을 먼저 train 384개와 dev 192개 parent로 나눴다. 같은
원천이나 같은 parent의 codec 변형이 양쪽에 걸치지 않는다. `data_guard`가 이 분할과
locked-eval 원천 누수를 검사한다.

### 잠금 forensic audit

`forensic_call_audit_v1`은 학습 bank와 원천이 겹치지 않는 240개 parent × 5개 channel,
총 1,200개다. 후보 선택에는 쓰지 않고 고정 후보의 최종 확인에만 사용한다. 실제 수사
통화 녹음은 아니므로 결과는 구조/codec stress test이지 실제 전화 분포의 성능 보장은
아니다.

## 2. 분리 전 원본과 voice stem 대조

동일한 공개 XLS-R-2B anti-deepfake head를 원본 통화와 HTDemucs voice stem의 같은
4초 창에 적용했다.

| 범위 | voice stem EER | 원본 EER |
|---|---:|---:|
| 전체 | 0.3733 | 0.3733 |
| clean | 0.3000 | **0.2500** |
| G.711 | 0.3833 | **0.3694** |
| G.722 | **0.2861** | 0.3000 |
| Opus narrowband | 0.3833 | **0.3667** |
| G.711→Opus | 0.4000 | **0.3833** |

원본이 일부 전화 channel에서는 낫지만, `real_fake`, `fake_real`, `fake_fake` 대조와
복합 대화에서는 stem이 더 안정적이었다. 학습 dev에 원본을 세 번째 expert로 넣은
3-view head도 2-view head를 이기지 못했다. 따라서 원본 XLS-R 추가 pass는 계산량만
늘리고 채택하지 않았다. 현재 구조는 HTDemucs-only도 아니다. EAT, SPEAR, MERT,
modern fakeprint는 계속 원본 혼합음을 보므로 원본과 stem의 두 view를 이미 함께 쓴다.

## 3. v23: 시간 구간 XLS-R/Spectra 합의

voice stem의 같은 시간 위치에서 다음 네 feature만 사용한다.

1. released XLS-R fake logit
2. Spectra-AASIST fake margin
3. 두 값의 곱
4. 두 값의 절대 disagreement

작은 balanced logistic head를 call-train의 local window label로 학습하고, 창 margin은
temperature 2의 log-mean-exp로 파일 단위 집계한다. 기존 v22 Voice logit 40%와 이
expert 60%를 결합한다. 분리 과정에서 거의 무음이 된 stem은 invalid로 표시해 결합에서
제외한다.

독립 검증에서 Voice EER은 다음과 같이 변했다.

| 평가군 | 기존 | v23 |
|---|---:|---:|
| Factorial 전체 | 0.2419 | **0.2038** |
| Phone factorial | 0.1425 | **0.1400** |
| Forensic locked | 0.3367 | **0.3267** |

## 4. v24: voice-only File 일관성

v23은 sparse fake voice를 찾아도 배점 0.5인 `FILE_FAKE_PROB`를 바꾸지 않았다. 한편
대회 정의상 음성만 존재하는 파일에서는 `FILE_FAKE == VOICE_FAKE`다. 후단의 여러
file expert 때문에 두 점수의 scale이 어긋나는 것을 다음 조건에서만 완화한다.

```text
VOICE_PRESENT_PROB >= 0.5 and MUSIC_PRESENT_PROB < 0.5:
    File_logit = 0.60 * old_File_logit + 0.40 * final_Voice_logit
```

혼합과 음악-only 파일, Music fake score, 두 Presence score는 건드리지 않는다. 0.40은
factorial dev의 첫 안정 plateau에서 고정했고 holdout/locked/phone/forensic에는 그대로
적용했다.

| 평가군 | 기존 File | v24 File | 기존 Voice | v24 Voice | 기존 ADS | v24 ADS |
|---|---:|---:|---:|---:|---:|---:|
| Factorial dev | 0.2800 | **0.2724** | 0.2571 | **0.2400** | 0.7126 | **0.7198** |
| Factorial holdout | 0.2895 | **0.2800** | 0.2343 | **0.1771** | 0.7056 | **0.7217** |
| Factorial locked | 0.3124 | **0.3124** | 0.2229 | **0.2000** | 0.6861 | **0.6907** |
| Phone factorial | 0.2617 | **0.2041** | 0.1425 | **0.1400** | 0.7401 | **0.7694** |
| Forensic-call locked | 0.3440 | **0.3300** | 0.3367 | **0.3267** | 0.6429 | **0.6519** |

표의 각 행에서 Music EER은 완전히 동일하다. 모든 판단은 해당 파일의 점수와 시간
창만 사용하므로 평가 파일 독립성 규칙도 지킨다.

## 5. AI 작곡 후 실제 악기 연주인 경우

Suno/Udio/YuE처럼 파형까지 생성한 음악과, AI가 MIDI/곡 구조만 만들고 실제 사람이
연주·녹음한 음악은 다른 forensic 문제다. 후자에는 neural codec, vocoder, 위상,
deconvolution 흔적이 존재하지 않을 수 있다. 리듬의 전형성 하나로 분류하면 loop 기반
human music, EDM, 장르 관습을 가짜로 오인할 위험이 크다.

현재 공개 연구도 주로 Suno/Udio 또는 waveform generator의 음향 흔적을 탐지하며,
unseen generator와 가벼운 audio manipulation 일반화를 핵심 난제로 지적한다.
구조적 반복·화성·리듬은 연구할 가치가 있지만 대회 label과 일치하는
AI-composed/real-performed 자료가 확보되기 전에는 제출 score에 넣지 않는다.

- Afchar et al., *AI-Generated Music Detection and its Challenges*:
  <https://arxiv.org/abs/2501.10111>
- Afchar et al., *A Fourier Explanation of AI-music Artifacts*:
  <https://arxiv.org/abs/2506.19108>
- Dai et al., *What is missing in deep music generation?*:
  <https://archives.ismir.net/ismir2022/paper/000079.pdf>

향후 compositional expert는 beat/bar-aware MERT 또는 SPEAR sequence에서 section
self-similarity, repetition entropy, tempo-grid residual을 추출하고, human-composed와
AI-composed를 동일 악기 renderer로 렌더링한 통제 자료에서 generator/genre-disjoint로
검증해야 한다.

## 6. 제출물 검증

후보 ZIP은 `sparse_call_consistency_v24.zip`이다.

- ZIP 크기: `9,172,037,816 bytes` (< 10 GB)
- 압축 해제 합계: `9,172,010,824 bytes` (< 32 GB)
- 최대 단일 member: `2,387,980,808 bytes` (< 4 GiB sentinel 문제 회피)
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- 전체 ZIP CRC: 오류 없음
- SHA-256:
  `b9c482d41276c551987c1707a70652a59f91d362b9066d8364322103e5e65daa`
- 3-file end-to-end GPU smoke: 성공
- 전체 unit/regression: 61개 통과
- 임시 XLS-R/Spectra/EAT/SPEAR 통계: 실행 종료 후 삭제 확인

로컬 개선폭을 그대로 leaderboard에 대입할 수는 없다. 보수적으로는 v18의
`0.76164`에서 Total `0.768--0.785` 범위를 기대한다. 실제 test가 긴 voice-only 통화를
많이 포함하면 상단, 음악/혼합이 대부분이면 하단에 가까울 것이다. 0.8을 보장하는
후보는 아니며 남은 주 병목은 generator-disjoint Music EER와 혼합 RF/FR File EER다.
