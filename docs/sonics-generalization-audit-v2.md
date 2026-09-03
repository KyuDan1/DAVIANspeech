# SONICS 원본 오디오 일반화 재검증

2026-09-02 기준. 과거 v14에서 전화 음악 expert로 사용했던
`awsaf49/sonics-spectttra-gamma-5s`를 현재 v19 계열에 다시 넣을 가치가 있는지
검증했다. 모델은 SONICS의 Suno/Udio와 FMA로 학습된 5초 SpecTTTra 분류기이며,
음원 분리 없이 16 kHz 원본 오디오를 직접 입력한다.

## 결론

**현재 제출 ensemble에는 추가하지 않는다.** 전화 음악 단독에서는 일부 도움이
되지만, 혼합 오디오와 미관측 YuE 생성기로 일반화되지 않는다. 1% 이하 결합은 거의
중립이고, 성능을 움직일 정도의 5--10% 결합은 YuE에서 회귀한다. 이미 패키지에 있는
다른 원본 오디오 expert와 비교해 추가 모델 크기와 추론 비용을 정당화할 이득이 없다.

## 평가 방법

- 학습이나 calibration 없이 공개 checkpoint 그대로 사용
- 5초 non-overlap window와 마지막 tail을 평가한 뒤 확률 평균
- Factorial holdout 400개, phone factorial 1,200개, YuE 124개
- v19 순서와 동일하게 LME+SPEAR, temporal, MERT, fakeprint, invariant head를
  적용한 점수 뒤에 SONICS를 logit 결합
- Voice score와 CPS는 변경하지 않음
- 모든 평가는 파일별 독립적으로 수행

평가 스크립트는 외부 `sonics` Python 패키지 대신 저장소의 dependency-free
`src/sonics_detector.py`를 사용하도록 고쳤다. 따라서 평가 서버 기본
PyTorch/torchaudio 외의 런타임 다운로드가 없다.

## 단독 성능

| 평가군 | File EER | Music EER |
|---|---:|---:|
| Factorial holdout | 0.4076 | 0.4114 |
| Phone factorial | 0.4024 | 0.3200 |
| YuE | 0.5265 | 0.5484 |

전화 단독 음악에서 과거 EER 0.325와 비슷한 신호는 재현됐지만, 전화 혼합에 대한
과거 EER은 0.374--0.466이었다. 특히 YuE에서는 score 방향까지 뒤집혀 SONICS
학습 생성기 밖의 일반 artifact detector로 해석할 수 없다.

## v19 뒤 결합

| weight (File/Music) | Factorial ADS | Phone ADS | YuE ADS |
|---:|---:|---:|---:|
| 0% | 0.7453 | 0.7319 | **0.8360** |
| 1% | 0.7453 | 0.7322 | **0.8360** |
| 2.5% | 0.7453 | 0.7331 | **0.8360** |
| 5% | **0.7470** | **0.7331** | 0.8333 |
| 10% | **0.7470** | **0.7331** | 0.8312 |

1%에서 Factorial mixed File EER은 `0.3444 -> 0.3356`으로 좋아졌지만 Music
EER은 `0.2933 -> 0.3000`으로 나빠졌다. Phone G.726 Music EER은
`0.28 -> 0.27`이지만 전체 Phone 이득은 ADS 약 0.00036에 그쳤다. 이 작은
부분 이득을 위해 YuE 회귀 위험과 별도 69 MB transformer pass를 추가하지 않는다.

수치는 `reports/sonics_generalization_v2/fusion_sweep.csv`와
`reports/sonics_generalization_v2/slices.csv`에 저장했다. v19 재구성 수치는
체크포인트 audit 저장 방식의 미세한 차이 때문에 기존 문서와 최대 약 0.002 ADS
차이가 있지만, SONICS 추가 전후 비교에는 동일한 기준을 사용했다.

## 다음 우선순위

SONICS 추가보다 실제 v18에서 전이가 확인된 channel/component invariance를
개선한다. 현재 invariant head의 전화 학습은 단순 8 kHz round-trip만 포함하지만,
`phone_presence_factorial_train_v1`에는 G.711, G.726, Opus-NB, PSTN과 단일/혼합
오디오 2,400개가 이미 누수 없이 준비돼 있다. 이 데이터를 같은 EAT/SPEAR head에
추가해 가장 약한 Opus-NB와 music-only를 직접 보강한다.
