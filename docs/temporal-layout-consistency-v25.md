# Temporal layout consistency v25

2026-09-03 기준. v25는 `sparse_call_consistency_v24`의 모델과 확률을 유지하면서,
**음성과 음악이 시간상 국소적으로 나타나는 비전화 혼합 파일**의
`FILE_FAKE_PROB`만 성분 점수와 일치시키는 보수적 후처리 후보이다. 새 신경망 추론과
test-set 전체 통계는 사용하지 않는다.

## 1. 동기

대회 정의상 음성이나 음악 중 하나라도 fake면 File도 fake다. 그러나 현재 MoE의
File head와 Voice/Music head는 서로 다른 expert를 거치므로, 순차 혼합과 hold music에서
한 성분이 강한 fake 증거를 내도 File 순위가 이를 반영하지 못할 수 있다.

반대로 동시 중첩 음원에서 무조건 `max(Voice, Music)`을 적용하면 분리 누출과 잘못된
component score까지 File에 전달된다. 실제 audit에서도 partial/sequential은 좋아졌지만
concurrent와 background-music 통화는 나빠졌다. 따라서 성분을 authenticity 입력으로
직접 쓰기 전에 시간 layout을 먼저 판정한다.

## 2. 학습 없는 temporal-layout router

HTDemucs는 기존 파이프라인이 이미 실행하므로 추가 분리나 모델 pass는 없다. 원본,
voice stem, music stem을 0.5초 frame/0.25초 hop으로 나누고 각 stem에서 다음 값을 구한다.

```text
ratio[t] = stem_RMS[t] / (original_RMS[t] + eps)
gap(stem) = log1p(Q90(ratio)) - log1p(Q50(ratio))
layout_score = max(gap(voice), gap(music))
```

항상 함께 들리는 성분은 Q90과 Q50이 비슷하다. 일부 구간에만 등장하는 순차 음성,
부분 overlap, hold music은 차이가 커진다. 이 score는 amplitude에 거의 불변이고 fake
label을 전혀 사용하지 않는다. stem은 **routing용 에너지**만 제공하며 fake 증거로
사용하지 않는다.

Factorial dev의 mixed 파일에서 concurrent를 음성/음악이 시간적으로 분리된
partial/sequential과 구분하는 ROC-AUC는 `0.9061`이었다. 높은 specificity를 우선해
`layout_score >= 0.35`를 사용하면 dev에서 sequential/localized recall `0.715`,
concurrent specificity `0.860`이다.

## 3. 적용 규칙

먼저 v24의 temporal Voice consensus와 voice-only File consistency를 모두 적용한다.
그 뒤 아래 조건을 동시에 만족할 때만 File logit을 고친다.

```text
VOICE_PRESENT_PROB >= 0.5
MUSIC_PRESENT_PROB >= 0.5
layout_score >= 0.35
telephone_router == false

component_or = max(final_VOICE_FAKE_PROB, MUSIC_FAKE_PROB)
File_logit = 0.30 * old_File_logit + 0.70 * component_or_logit
```

`0.70`은 Factorial dev에서 File EER가 처음 명확히 개선되는 보수적 weight다. 기존
narrowband telephone router가 선택한 파일에는 적용하지 않는다. Phone factorial에서
혼합 consistency가 소폭 불리했고, 현재 전화용 MoE의 장점을 보존하기 위해서다.

## 4. 독립 검증

아래는 v24와 v25의 File EER다. Voice/Music fake와 두 Presence 열은 동일하다.

| 평가군 | v24 | v25 | 변화 |
|---|---:|---:|---:|
| Factorial dev | 0.2724 | **0.2647** | -0.0076 |
| Factorial holdout | 0.2800 | **0.2724** | -0.0076 |
| Factorial locked | 0.3124 | **0.2895** | -0.0229 |
| Phone factorial | 0.2041 | **0.2041** | 0.0000 |
| Forensic-call locked | 0.3300 | **0.3295** | -0.0005 |

Forensic의 전체 변화가 작은 이유는 G.711/Opus/transcode 파일에서 의도적으로 기존
전화 경로를 그대로 보존했기 때문이다. router가 보정을 허용한 clean과 G.722 subset의
File EER은 각각 `0.2450→0.2250`, `0.2950→0.2725`였다.

추가 generator/domain audit 결과는 다음과 같다.

- YuE 124개: concurrent 32개 중 잘못 route된 파일 `0`; 전체 File EER
  `0.1861→0.1861`로 비열화
- Source-disjoint simultaneous mixture 200개: File EER
  `0.1633→0.1633`으로 비열화
- YuE partial 32개 중 3개, sequential 32개 중 11개만 선택되어 high-specificity라는
  설계 목적과 일치

이는 큰 점프를 보장하는 새 detector가 아니라, 현재 MoE의 성분 논리를 특정 layout에서
복구하는 저위험 보정이다. 실제 test에서 sequential/hold 혼합이 많을수록 이득이 커진다.

## 5. AI 작곡 + 실제 악기 문제

AI가 MIDI/악보/구성만 만들고 사람이 실제 악기로 연주·녹음했다면 neural codec,
vocoder, 위상 같은 waveform 생성 artifact가 없을 수 있다. 이 경우 “리듬이 전형적”인
지만으로 fake를 정하면 loop 기반 human music, EDM, 장르 관습을 대량 오탐할 위험이
있다. 현재 v25에는 이 가설을 넣지 않았다.

별도 compositional expert는 다음 통제 실험을 통과한 뒤에만 추가한다.

1. human MIDI와 여러 LLM/music model의 AI MIDI를 수집한다.
2. 양쪽을 **같은 악기 renderer와 음색**으로 렌더링해 음향 차이를 제거한다.
3. tempo/장르/길이를 맞추고 composer와 generator 단위로 분리한다.
4. beat/bar-aware SPEAR 또는 MERT sequence에서 repetition entropy, section
   self-similarity, tempo-grid residual, phrase-length 분포를 학습한다.
5. generator-disjoint와 genre-disjoint 양쪽에서 human false-positive가 유지될 때만
   낮은 가중치의 별도 expert로 사용한다.

즉 waveform-forensics와 composition-forensics는 하나의 score로 억지로 섞지 않고,
서로 다른 expert와 검증셋으로 관리한다.

## 6. 구현 및 안전성

- `src/component_consistency.py`: file-local layout score와 postprocessor
- `scripts/build_layout_consistency_submission.py`: v24를 hardlink-safe하게 복제해 v25 구성
- `scripts/extract_separator_presence_stats.py`: 기존 stem 재사용 audit 지원
- `tests/test_component_consistency.py`: sequential cue, gate, 전화 제외, 열 불변성 검사

3파일 GPU end-to-end smoke에서 실행에 성공했고 telephone router는 phone sample
1/3만 선택했다. 새 postprocessor는 해당 smoke의 3개 파일이 route 조건을 충족하지 않아
fake 열을 바꾸지 않았으며, 실행 종료 뒤 layout/telephone 임시 통계가 남지 않았다.
