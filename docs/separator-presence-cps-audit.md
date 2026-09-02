# HTDemucs stem presence CPS audit

2026-09-02 기준. v16/v18의 실제 CPS가 약 `0.98930`에 머문 원인을 좁히기 위해,
기존 파이프라인이 이미 계산하는 HTDemucs vocal/non-vocal stem의 에너지와 시간
점유율을 presence 보조 신호로 검증했다. 결론은 **방향은 맞지만 최종 모델에 넣을
만큼의 이득이 없어 폐기**한다는 것이다.

## 방법과 대회 규칙

HTDemucs는 SAM-Audio와 달리 diffusion separator가 아니다. 다만 어떤 separator든
원본 생성 artifact를 바꿀 수 있으므로 fake 판정에는 새 stem 점수를 추가하지 않고,
구성 성분의 존재 여부인 CPS에만 사용했다.

각 파일에서 다음 값을 독립적으로 계산했다.

- 원본 대비 vocal/non-vocal stem energy
- 두 stem 사이의 energy share
- 0.5초 frame별 energy ratio의 50/75/90/100 percentile
- vocal/non-vocal 우세 frame 비율과 reconstruction error

평가 파일 전체의 평균·순위·분포로 한 파일을 보정하지 않았다. 최종 후보가 되더라도
각 파일의 값만 사용하는 구조이므로 파일 단위 독립 예측 규칙을 만족한다.

## 잘못된 중간 기준을 제거한 과정

첫 audit에서는 temporal ADS 실험용 prediction CSV를 기준으로 사용해 Factorial CPS가
`0.99586→0.99663`으로 보였다. 그러나 그 CSV는 최종 telephone Voice head까지 모두
반영한 CPS 배포 출력이 아니었다. 따라서 이 수치는 채택하지 않았다.

캐시된 PANNs, EAT AudioSet, EAT latent probe와 telephone head를 이용해 실제 v16+
CPS 경로를 다시 구성했다.

```text
Voice = 0.65 PANNs + 0.35 EAT
Music = 0.60 × (0.10 PANNs + 0.90 EAT) + 0.40 × EAT latent probe
telephone Voice = 기존 Voice와 phone head를 logit 0.90:0.10 결합
```

재구성 시 Factorial holdout 400개 중 67개, phone factorial 1,200개 중 1,200개가
전화로 라우팅됐다. 이는 기존 배포 기록과 일치한다.

## 최종 결과

| 고정 평가군 | 최종 CPS 기준 | stem 저비중 결합 최고 | 변화 |
|---|---:|---:|---:|
| Factorial holdout 400 | 0.999514 | 0.999571 | +0.000057 |
| Phone factorial 1,200 | 0.999353 | 0.999370 | +0.000017 |

Factorial에서는 Music frame Q90을 7.5% 결합해 Music AUC가
`0.999886→1.000000`이 됐지만 Voice AUC `0.999143`은 그대로였다. 전화에서는
1% 결합 시 Voice share가 `0.998906→0.998909`, Music energy가
`0.999800→0.999831`로 움직인 것이 전부다. 더 큰 비중은 대부분 악화했다.

## 해석과 결정

이득이 Total에 주는 효과는 최대 수백만 분의 일 수준이다. 반면 실제 리더보드 CPS는
`0.98930`으로 로컬의 `0.999+`보다 훨씬 낮다. 이는 현재 합성 presence 평가가 이미
포화됐고 실제 오류 유형을 충분히 포함하지 못한다는 뜻이다. stem 통계를 더 맞추는
것은 실제 분포 불일치를 해결하지 못하며, 로컬 holdout을 반복 최적화할 위험이 더
크다.

따라서 stem presence fusion은 제출에 포함하지 않는다. 다음 CPS 연구는 전화 변형이
아니라 실제 배경 소음, 짧은 발화/짧은 음악, 보컬과 비언어 사람 소리, 무반주 노래,
말소리가 섞인 방송·환경음처럼 AudioSet class 경계가 흔들리는 독립 원천으로 새
audit을 만드는 것이 우선이다. Presence 결과는 계속 ADS gate와 분리한다.

재현 코드는 다음과 같다.

- `scripts/extract_separator_presence_stats.py`
- `scripts/reconstruct_cps_presence.py`
- `scripts/evaluate_separator_presence.py`
- `tests/test_separator_presence_stats.py`
