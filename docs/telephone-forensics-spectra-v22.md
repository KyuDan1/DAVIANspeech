# 실제 통화 가설과 Spectra stem MoE v22

2026-09-03 기준. 주관 기관과 보이스피싱 대응이라는 대회 배경을 고려하면 비공개
평가셋에 실제 또는 실제에 가까운 전화 통화가 포함될 가능성이 높다. 다만 공지에는
전화채널이 **일부 샘플**이라고 명시되어 있으므로, 모든 파일을 전화용 모델로
교체하지 않고 일반 오디오 경로를 보존한다.

현재 공개 최고 기준점은 `channel_invariant_moe_v18.zip`의
Total `0.7616379312`, ADS `0.7363412698`, CPS `0.9893078836`이다.

## 1. 실제 전화 평가에서 필요한 조건

실제 통화는 단순 8 kHz 저역통과와 다르다. 다음 nuisance가 생성 흔적을 약화시키거나
그 자체가 가짜 흔적처럼 보일 수 있다.

- PSTN/VoIP 대역 제한과 G.711 A/μ-law, GSM, G.722, Opus 재인코딩
- AGC, clipping, 잡음 억제, 패킷 손실 보상, 배경 잡음과 재생 녹음
- 발신자·피해자·ARS의 순차 발화, hold music, speech/music overlap
- 4--60초 길이에서 일부 구간만 가짜인 경우

ASVspoof 2021 LA도 PSTN/VoIP와 여러 코덱을 핵심 평가 조건으로 정의한다.
2026 Odyssey 연구는 대역폭 감소보다 codec이 더 큰 성능 저하 요인이며, RawBoost와
random quantization augmentation이 실제 전화 학습 데이터보다 나을 수 있다고
보고했다.

- ASVspoof 2021 evaluation plan:
  <https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf>
- Klein et al., *The Effect of Telephony Transmission on Source Tracing of
  Audio Deepfakes*: <https://www.isca-archive.org/odyssey_2026/klein26_odyssey.html>

## 2. 공식 통신 채널 audit

공식 ASVspoof 2021 LA에서 길이 4--60초, codec별 real/fake 100개씩을 골라 총
1,200개의 `asvspoof2021_la_channel_audit_v1`을 만들었다. 데이터 역할은
`locked_eval`로 고정했다.

중요한 누수 한계가 있다. 현재 XLS-R-2B AntiDeepfake의 논문상 post-training
데이터에 ASVspoof 2021 LA가 포함된다. 따라서 이 audit은 새로운 생성기에 대한
일반화 성능을 주장하거나 weight를 선택하는 데 쓰지 않고, 고정된 후보들의 상대적
채널 안정성만 확인한다.

전화 router는 authenticity label과 거의 무관하게 협대역을 찾았다.

| codec | routed 비율 |
|---|---:|
| A-law | 92.0% |
| μ-law | 92.0% |
| GSM | 94.5% |
| PSTN | 95.5% |
| G.722 wideband | 0.0% |
| Opus wideband | 0.0% |

그러나 router가 정확한 것과 최종 EER이 개선되는 것은 별개였다.

| 후보 | File EER | Voice EER | 판단 |
|---|---:|---:|---|
| v18 single invariant 5% | **0.00833** | **0.00667** | 기준 |
| v20 paired global 2.5% | 0.00833 | 0.00667 | EER 변화 없음 |
| v21 paired phone-route +20% | 0.01500 | 0.01000 | 명확한 회귀, 폐기 |

v21은 router가 선택한 748개에서 File EER `0.01337→0.02406`, Voice EER
`0.01070→0.01604`로 악화됐다. synthetic phone subset만 강하게 보정하면
clean/wideband와의 전역 순위가 깨진다는 이전 관찰이 실제 코덱 audit에서도
재현됐다. 따라서 v22에는 hard telephone ADS correction을 넣지 않는다.

## 3. 분리 방식에 대한 결론

현재 제출의 실제 separator는 diffusion 기반 SAM-Audio가 아니라 판별형
HTDemucs v4다. 그래도 어떤 분리기도 생성 흔적을 약화시키거나 separator 고유
artifact를 추가할 수 있으므로, 기존 구조는 다음처럼 원본과 stem을 함께 쓴다.

- HTDemucs voice/music stem + XLS-R legacy evidence
- original audio EAT, SPEAR, MERT, modern fakeprint, temporal/invariant heads
- stem을 제거하지 않고 original-audio 전문가로 손실된 흔적을 보완

즉 separation-only도 separation-free-only도 아니다. 두 view의 오류가 다른 것을
작은 weight로 결합한다.

## 4. Spectra-AASIST 실험

새 전문가는 공개 `lab260/Spectra-AASIST`다. Wav2Vec2 XLS-R-300M encoder와
AASIST graph head를 사용하며, 공개 checkpoint 전체를 오프라인으로 포함했다.

- 모델: <https://huggingface.co/lab260/Spectra-AASIST>
- 고정 revision: `eb65c2662d9e646d72557b3f4bdd08b000068c7f`
- checkpoint SHA-256:
  `2e2727a7397f78d28b0a2a2b8ee031ff08143b9c431ea7f06fc29a808b0180db`
- 모델 카드 metadata: Apache-2.0, 본문: MIT
- 모델 카드는 pre-release/unpublished라고 밝히므로 자체 locked audit 결과만 믿는다.

원본 혼합음에 직접 쓰면 일반 Voice EER `0.310`, 전화 Voice EER `0.1875`로 약했다.
반면 이미 계산하는 HTDemucs vocal stem에서 처음·중간·끝 4.0375초 세 구간의 fake
logit을 **평균**하면 max/LME보다 안정적이었다. Separator를 한 번 더 실행하지 않는다.

### Voice 결과

선택은 factorial dev에서만 했고, 이후 holdout/phone/한국어 다중 TTS에는 weight를
고정했다. v22는 v18 Voice logit 90% + Spectra mean logit 10%다.

| 평가군 | v18 Voice EER | Spectra 단독 | v22 결합 | 역할 |
|---|---:|---:|---:|---|
| factorial dev | 0.2857 | 0.2400 | **0.2571** | 선택 |
| factorial holdout | 0.2229 | 0.2171 | **0.2114** | 독립 검증 |
| phone factorial | 0.1675 | 0.1800 | **0.1600** | 독립 전화 검증 |
| 최신 한국어 TTS 288 | 0.0833 | 0.0729 | **0.0833** | locked, 비악화 |

3-window mean은 최신 한국어 TTS에서 1-window `0.0755→0.0729`, ASVspoof voice dev에서
`0.0114→0.0000`으로 개선됐다. 반대로 max와 LME는 한국어 TTS에서 `0.0859`로
나빠져 사용하지 않는다. 이는 기존 XLS-R에서 max 대신 LME가 실제 점수를 올린 것과
같이, 긴 파일의 극단값 편향을 피해야 한다는 증거다.

### File 결과

Fake Music인데 Real Voice인 RF cell을 억누르지 않도록, v18이
`VOICE_PRESENT_PROB >= 0.5`이고 `VOICE_FAKE_PROB >= MUSIC_FAKE_PROB`라고 판단한
파일에만 Spectra logit 5%를 File에 결합한다. 이 조건과 weight는 dev에서 고정했다.

| 평가군 | v18 File EER | v22 File EER |
|---|---:|---:|
| factorial dev | 0.2971 | **0.2876** |
| factorial holdout | 0.2553 | **0.2476** |
| phone factorial | 0.2583 | **0.2583** |
| 최신 한국어 TTS 288 | 0.0859 | **0.0859** |

Music score와 두 Presence score는 bit-for-bit 유지한다. 모든 판단은 파일 내부
정보만 사용하므로 평가 파일 독립성 규칙도 지킨다.

## 5. 배포 후보와 현실적 기대

후보는 `spectra_stem_v22_fixed/`다. 전체 entrypoint smoke에서 다음을 확인했다.

- 기존 v18 대비 변경 열은 File/Voice뿐
- Music/Voice Presence/Music Presence는 완전 동일
- 출력 ID 일치, 모든 확률 finite `[0, 1]`
- EAT/SPEAR/Spectra 임시 통계 삭제
- B200 측정 peak CUDA reserved `11.059 GiB`; L4 22.4 GiB 한도 아래에 여유
- 압축 전 패키지 `9,172,096,029 bytes`, 32 GB 제한 아래

로컬 개선폭을 실제 분포에 그대로 대입할 수는 없다. 전화 비중과 혼합 비중에 따라
v18 Total `0.76164`에서 대략 `0.763--0.771`을 기대하는 보수적 후보이며, 이것만으로
0.8을 보장하지 않는다. 0.8의 가장 큰 남은 병목은 Music EER와 RF/FR 혼합에서의
File EER다. v22 실제 결과 이후 다음 우선순위는 generator-disjoint 음악/혼합
전문가이며, 전화 hard routing의 weight 확대는 하지 않는다.
