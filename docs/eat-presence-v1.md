# EAT 3-view Presence v1

## 목표와 결론

현재 최고 제출 `lme_spear_v1.zip`의 CPS `0.9891721`을 개선하고, PANNs presence
오류가 File fake의 hard gate로 전파되는 문제를 함께 줄인다. 최종 후보는
source separation 없이 원본 오디오에서 EAT AudioSet score를 구하고 PANNs와
결합한다. 학습 데이터나 평가 데이터로 head를 fitting하지 않은 train-free
ensemble이다.

## 구조

길이 6초의 시작·중앙·끝 view를 EAT-base AS2M에 입력한다. 6초보다 짧으면 패딩한
view 하나만 쓴다. 각 view의 527개 AudioSet 확률 중 voice/music label group의
최댓값을 취한다.

```text
Voice presence = 0.70 × PANNs + 0.30 × EAT-3view
Music presence = 0.10 × PANNs + 0.90 × EAT-3view
File gate      = 0.60
```

PANNs는 약 4초의 모든 구간을 보므로 짧게 등장한 성분에 강하고, 일반 오디오 SSL인
EAT는 음악과 보컬의 의미 표현에 강하다. 최대 세 view만 사용해 4초~60초 입력에서
연산량을 제한한다. HTDemucs나 diffusion separator 출력은 presence에 사용하지 않는다.

File score는 새 presence와 gate 0.60으로 anchor component의 max를 다시 계산한 뒤,
기존처럼 SPEAR joint file을 10% 결합한다. w30 실험은 폐기하고 실제 최고점에서
검증된 SPEAR 10%를 유지한다.

## CPS 결과

| 평가군 | 기존 CPS | EAT 결합 CPS | 변화 |
| --- | ---: | ---: | ---: |
| Factorial v2 | 0.994108 | **0.997787** | +0.003679 |
| 전면 전화 Factorial | 0.970331 | **0.985909** | +0.015578 |
| Competition v2 | 0.968523 | **0.975899** | +0.007376 |
| Competition v3 | 0.973345 | **0.976582** | +0.003237 |

YuE audit은 모든 파일에 음악이 있어 Voice AUC만 계산 가능하며
`0.983796→0.997106`으로 개선됐다. 다섯 평가군에서 계산 가능한 Voice/Music AUC가
모두 상승했다.

## File gate 전파 결과

아래 ADS는 LME anchor + SPEAR 10%를 기준으로 presence만 분리한 경우와, 새
presence를 gate 0.60에 실제 전파한 경우의 비교다.

| 평가군 | 기존 File gate 유지 | 새 gate 0.60 | 변화 |
| --- | ---: | ---: | ---: |
| Factorial v2 | 0.656632 | **0.667117** | +0.010485 |
| YuE audit | 0.666518 | **0.685200** | +0.018682 |
| Competition v2 | 0.822729 | **0.822729** | 0.000000 |
| Competition v3 | 0.746747 | **0.747967** | +0.001220 |

즉 CPS만 올리고 ADS는 그대로 두는 decoupled 방식보다, 검증된 gate 변경을 적용하는
편이 네 평가군 모두에서 비열등하다. gate 0.5는 일부 평가군에서 더 높았지만
Competition v2가 소폭 하락해 선택하지 않았다.

## 실행과 제출 안전성

- EAT는 anchor 추론이 끝난 뒤 로드하고 종료 후 CUDA cache를 비운 다음 SPEAR를
  로드한다. XLS-R/EAT/SPEAR를 동시에 GPU에 올리지 않는다.
- B200 기준 모델 로드를 포함해 150개 추론 59.5초, 실제 scoring 구간은 8.3초였다.
  현재 최고 제출의 35분 실행시간에 추가되어도 60분 제한 안으로 예상된다.
- 평가 서버 기본 `torch`, `torchaudio`, `transformers`만 사용하며 `timm`은 설치하지
  않고 기존 local compatibility shim을 사용한다.
- 실제 package entrypoint 한 파일 end-to-end smoke를 통과했다.

EAT는 AudioSet-2M으로 fine-tune된 general-audio encoder다. 원 논문은 효율적인
audio SSL과 downstream audio understanding을 다룬다:
<https://arxiv.org/abs/2401.03497>

## 재현 파일

- inference: `src/eat_presence.py`
- post-fusion: `src/eat_presence_fusion.py`
- score extraction: `scripts/score_eat_presence.py`
- cross-audit: `scripts/evaluate_eat_presence_fusion.py`
- metrics: `reports/eat_presence_v1/metrics.csv`
- package builder: `scripts/build_eat_presence_submission.py`
- package: `submit_lme_spear_eat_presence_v1/`
