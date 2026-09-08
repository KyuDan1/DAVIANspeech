# v66: 최고 제출의 혼합·전화채널 오류 진단

2026-09-05. **새 모델의 개선 결과가 아니다.** v50+v57 최고 제출의 authorized
development 예측을 분석했다. 공식 test 구성이나 EER을 역산하지 않는다.

## 데이터와 비교법

codec mixed dev v4의 600행은 120개 혼합 오디오 × 5채널이다. 원천 음성과 음악은
각각 120개이며, 같은 혼합의 clean/G.711/G.722/Opus/재압축을 짝지어 비교한다.
600개의 독립 녹음으로 해석하지 않는다. 모두 두 성분이 있는 mixed이므로 pure speech,
pure music 및 CPS AUC에 대한 검증은 아니다. 길이는 8.0–17.45초다.

RR=진짜 음성+진짜 음악, FR=가짜 음성+진짜 음악,
RF=진짜 음성+가짜 음악, FF=둘 다 가짜.
한 cell 안에서는 label이 상수라 EER이 정의되지 않는다. File은 RR과 각 Fake cell을,
Voice/Music은 다른 성분의 진위를 고정한 두 cell을 비교한다.

## 채널별 성분 EER

| 채널 | File | Voice | Music |
|---|---:|---:|---:|
| clean | 0.10000 | 0.11667 | 0.11667 |
| G.711 | 0.20000 | 0.10000 | 0.20000 |
| G.722 | 0.16667 | 0.10000 | 0.08333 |
| Opus NB 8k | 0.33333 | 0.35000 | 0.36667 |
| G.711→Opus | 0.30000 | 0.31667 | 0.40000 |

이 bank에서는 **전화 전체가 하나의 난이도가 아니다.** G.722의 Music은 clean보다
좋지만 Opus 계열은 양 성분 모두 크게 나쁘다. 따라서 bandwidth만 보는 router가
자동으로 해결할 것이라고 가정할 수 없다. 코덱별 학습/검증이 필요하다.

## 어느 혼합인가: File EER

| 채널 | RR vs FR | RR vs RF | RR vs FF |
|---|---:|---:|---:|
| clean | 0.16667 | 0.06667 | 0.03333 |
| G.711 | 0.16667 | 0.23333 | 0.13333 |
| G.722 | 0.23333 | 0.16667 | 0.06667 |
| Opus NB 8k | 0.36667 | 0.36667 | 0.26667 |
| G.711→Opus | 0.30000 | 0.33333 | 0.30000 |

각 비교는 30 real + 30 fake다. clean에서는 FR이 더 어렵지만 G.711에서는 RF가
더 어렵고 Opus에서는 둘 다 어렵다. “가짜 음성+진짜 음악만 해결하면 된다”는 결론은
이 결과와 맞지 않는다. 각 조건의 표본이 작고, 상대 순위의 통계적 확정은 주장하지 않는다.

혼합 배치별 pooled File/Music EER은 동시 혼합 0.2800/0.2300,
부분 겹침 0.2567/0.3200, 순차 혼합 0.1433/0.2000이다. source가 배치 사이에
다르므로 순차 방식 자체의 인과 효과로 해석하지 않는다.

## 같은 오디오에서 무엇이 변하는가

Opus 변형 후 Music Fake 확률의 변화:

- RF: 평균 0.30350 하락, 30/30개에서 하락.
- FF: 평균 0.33128 하락, 28/30개에서 하락.
- 실제 음악 RR/FR은 평균 약 0.078/0.079 상승.

즉, 가짜 음악을 Real 쪽으로 미는 현상이 강하고 양 클래스 간 순서 구분도 나빠진다.
단순히 출력 전체가 같은 방향으로 이동한 현상은 아니다.
clean 개발 EER threshold를 진단용으로 고정하면 RF의 FNR은 0.100→0.667,
FF는 0.133→0.767이다. **이 threshold는 배포에 사용하지 않는다.**
EER은 이미 threshold를 훑는 지표이므로 모든 점수에 같은 단조 변환을 적용하는
것만으로 이 ranking 악화를 고칠 수 없다.

## 방법론에 반영할 우선순위

1. 진행 중인 v64/v65로 음량 민감도와 코덱 증강의 효과를 판별한다.
   v64 epoch 2의 전체 development File EER은 0.25169로,
   동일 epoch v60 BCE 0.23230보다 아직 나쁘다. 중간 결과이며 완료 전 후보를 고르지 않는다.
2. Music의 Opus/RF/FF 검증을 별도 필수 보고로 유지한다. File-only 개선을 Music 개선으로
   주장하지 않는다. 기존 Music-selected WPT의 실패 기록도 있으므로 동일한 구조의
   작은 재조합만 다시 돌리는 것을 우선하지 않는다 (`docs/music-only-channel-robust-v51.md`).
3. 다음 구조적 후보는 저대역 음악 신호에서 직접 학습하는 표현과, 혼합 전후/코덱 전후
   대응 학습을 비교하는 방향이다. 구체적 구현과 승격은 별도 사전 프로토콜을 요구한다.
   현재 이 진단으로 새 모델을 통과시킨 것은 아니다.

## 관련 최신 연구를 다시 읽은 결과

[BAMM 연구](https://arxiv.org/html/2608.07359v1)는 8kHz CNN을 음악+음성 혼합으로
학습했을 때 실제 방송 ROC-AUC가 0.707→0.775로 개선되지만 여전히 어려움을 보고한다.
배경에 묻힌 가짜 음악을 놓치는 결과가 이번 관찰과 유사하다. 단, BAMM의 AI label은
여러 detector의 합의로 만든 것이어서 생성 provenance가 확인된 정답과 같다고 볼 수
없다. 우리 확정 정답 평가를 대체하지 않는다. 이 논문의 모델을 아직 내려받거나 쓰지 않았다.
논문에 적힌 `DaveLoay/BAMM-dataset` 저장소는 공개 GitHub API 조회에서 404를 반환했다.
공개 코드/가중치·이용조건은 현재 확인하지 못했으며, 접근 가능하다고 가정하지 않는다.

[Log-frequency robustness 연구](https://arxiv.org/html/2607.27454v1)는 주파수축
이동에 견디는 fakeprint 필터를 제안한다. 이는 속도 변형에 대한 방법이며 전화 코덱으로
정보가 제거되는 문제의 해결 증거는 아니다. 사용 대역도 SONICS 1–7kHz,
Suno v5 5–16kHz여서 좁은 전화 대역에 그대로 적용할 우선순위는 낮다.

## 재현 및 범위

```bash
python scripts/audit_mixed_channel_failures_v66.py \
  --output reports/mixed_channel_failures_v66
python -m pytest -q tests/test_mixed_channel_failures_v66.py
```

4개 테스트 통과. CSV key 완전 일치, paired metadata/채널 완비, single-class EER 처리,
고정 threshold와 label별 signed 변화 계산을 검사했다. 출력은
`reports/mixed_channel_failures_v66/{eers,contrasts,fixed_clean_threshold,paired_changes}.csv`.
`report.json`에 원자료와 코드 SHA-256을 기록했다. 개발 진단이며 locked 데이터, 학습,
새 ZIP/API 제출은 이번 분석에 사용하지 않았다.
