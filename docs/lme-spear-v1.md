# LME + SPEAR 결합 제출

## 리더보드 근거

두 방법은 같은 anchor pipeline의 서로 다른 부분을 바꾼다.

| 제출 | Total | ADS | CPS | 변경 |
|---|---:|---:|---:|---|
| v15의 직접 비교 anchor | 0.7356958 | 0.7075317 | 0.9891721 | max pooling |
| SPEAR v15 | 0.7410958 | 0.7135317 | 0.9891721 | 출력의 File/Music에 SPEAR 10% |
| LME의 직접 비교 anchor | 0.7364825 | 0.7083889 | 0.9893250 | max pooling |
| LME v1 | **0.7414039** | **0.7138571** | 0.9893250 | XLS-R window를 logmeanexp(τ=5) |

SPEAR의 실측 이득은 ADS `+0.006000`, LME는 ADS `+0.005468`이다. CPS는 두
실험 모두 변하지 않았다. 두 anchor의 점수가 조금 달라 단순 합산 기대치는
Total 약 `0.7460~0.7468`이지만, EER은 확률 순위 지표이고 SPEAR가 LME 이후의
File/Music 확률을 다시 섞으므로 실제 점수는 제출로 확인해야 한다.

LME voice probe로 계산한 Voice EER은 `0.2200`으로 anchor `0.2156`보다 오히려
나빠졌다. 따라서 LME의 이득은 File/Music에서 발생했다. SPEAR도 Voice와 CPS를
전혀 수정하지 않고 File/Music만 개선했다. 두 방법이 완전히 다른 성분에서
이득을 낸다고 말할 수는 없지만, 같은 File/Music 병목을 서로 다른 신호로
개선한다는 점에서 결합 우선순위가 높다.

## 정확한 결합

`submit_anchor_spear_v15`에 다음 순서만 적용했다.

1. HTDemucs voice/music stem의 XLS-R window score를 max 대신
   `logmeanexp(temperature=5)`로 집계한다.
2. 기존 방식대로 파일을 모두 처리한다.
3. 원본 오디오 SPEAR를 별도 pass로 실행하고 File/Music 확률에 각각 10%를
   선형 결합한다.

v15 안의 사전 LME `pipeline.py` SHA-256은
`3431dd81c7c4eca1c4a65e8d7e891cf92fb10509a691a3d69a7a1d5fe2960554`로,
팀원의 LME 변경 직전 커밋과 정확히 같았다. 결합 후 pipeline은 팀원의 LME
구현과 바이트 단위로 같고 SHA-256은
`5247ae12588f93bf80d6d4c1a1d6e12fe088b9f78dddaf1bd7315b7de967f437`다.

## 산출물 및 검증

- 제출 파일: `lme_spear_v1.zip`
- SHA-256: `cab23d8d513027d5ea469686ada517c95738f2ea8e92c62feb7f09d5c4fe6467`
- 압축/해제 크기: 6.1538/6.6529 GiB
- 원본 v15 대비 변경 member: `script.py`, `model/src/pipeline.py`만
- `requirements.txt`: `onnxruntime-gpu==1.23.2`
- 실제 음원 end-to-end smoke: PANNs → HTDemucs → XLS-R LME → ArtifactNet →
  SPEAR → 최종 CSV 생성 통과
- ZIP 전체 CRC, 중복 member, 최상위 구조, 10/32GB 제한 검사 통과
