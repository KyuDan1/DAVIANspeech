# v65: 넓은 음량 증강을 통한 WPT File 재학습

상태: 전체 학습 및 사전 고정 개발 판정 완료. **채택하지 않음**.
아래 실행 중 기록은 실험 이력이며, 최종 결과가 우선한다.

## 최종 결과 — 2026-09-05

best epoch 8. 단독 모델의 전체 개발 File EER은 0.207316으로 대조군
0.207841보다 아주 조금 낮았다. 그러나 실제 제출의 WPT slot 교체에서는
개발 File EER이 **0.233333 → 0.245556**으로 악화했다. 최대 채널 악화는
6.111%p였다. 주 판정 gate를 통과하지 않아 제출 모델에 반영하지 않는다.
다른 네 출력의 text-exact 보존은 통과했다. locked 평가나 API 제출은 하지 않았다.

증거: `reports/paired_gain_wpt_v65/development_audit/decision.json`.

## 질문과 대조군

기존 제출 WPT는 같은 오디오의 음량만 바꾸어도 File 확률이 크게 달라졌다
(`docs/wpt-gain-v63.md`). 추론 단계에서만 peak normalization을 추가한 v63은
음량 민감도는 줄었지만 개발 File EER은 악화했다. 출력 안정성은 정확도와 다르다.

v64는 학습부터 normalization을 적용한다. v65는 별개의 접근이다. 원래 전처리와
추론을 유지하고, **학습 음량 배율만 uniform [0.7, 1.3]에서 [0.1, 1.3]으로 확장**한다.
이를 통해 낮은 음량과 양자화/전화 코덱을 거친 경우를 학습한다. 전화 감지 router,
새로운 추론 분기, 추가 모델, 평가 파일 간 통계는 도입하지 않는다.

대조군은 완료된 v60 `paired_bce`이며 seed, 데이터, sampler, 원본/코덱 pair,
모델 초기값, 학습률, 학습량, File loss와 checkpoint 선택 기준을 유지한다.
`peak_normalize`는 False다. 같은 오디오에서 뽑은 원본/전화 pair는 같은 gain을 공유한다.
평가에서는 무작위 gain을 적용하지 않는다. gain은 생성 여부와 무관하고 label을 바꾸지 않는다.

전화 변환은 기존 `apply_channel`의 peak limiter를 거친다. 따라서 높은 peak에서는
원본 branch와 코덱 branch의 최종 배율이 정확히 같지는 않다. 같은 gain으로 시작한다는
뜻이며, 이 비선형 전처리도 대조군과 동일하다. 상한은 대조군의 1.3을 유지했고,
새로 도입한 조건은 낮은 음량 쪽이다.
이 실험은 모든 전화 왜곡이나 장시간 혼합의 해결을 의미하지 않는다.

## 누수 및 재현성

strict train/development 분리를 재검사한 뒤 학습한다. v61 locked 장시간 bank는
학습, checkpoint 선택, gain 범위 선택에 사용하지 않는다. train만 증강한다.

공용 trainer의 gain 설정은 기본값 [0.7,1.3]을 보존한다. 회귀 테스트에서 기존
배율 draw, codec 선택 뒤 RNG 상태, 원본/코덱 pair 대응, 평가 입력 무변형을 검사한다.
실행 중인 v64는 변경 전 trainer를 이미 로드했고 그 source snapshot과 SHA를 보존했다.
새 run도 실제 학습 source/config와 SHA를 보존한다.

## 사전 고정 평가

1. 학습 완료 후 v60 BCE와 동일 development 4,577행을 비교한다.
2. 기존 최고 제출의 WPT File logit slot 0.4를 새 전문가로 교체한다.
   개발 600행에서 다른 네 확률 열은 그대로 유지한다. 비중 sweep은 하지 않는다.
3. 주 판정은 File EER 최소 1%p 개선, domain/channel 최대 악화 2.5%p 이하,
   mean domain 개선 및 worst domain 비악화다. 단독 모델 비교는 진단일 뿐이다.
4. 통과해도 아직 공식 개선이 아니다. 독립 파일 재현, 사전 동결한 prospective 평가,
   장시간 검증, 실제 전체 추론의 L4 시간 및 오프라인 ZIP 검증이 남는다.

v64/v65 중 하나를 prospective 평가로 가져갈 경우 개발 기준으로 먼저 하나를 고정한다.
잠가둔 데이터에서 둘 다 점수를 보고 더 좋은 모델을 고르는 방식은 사용하지 않는다.
자동 API 제출은 하지 않는다.

## 실행 기록

신규 gain/기존 peak/training 테스트 14개와 coverage/독립 추론/slot/완료 판정/
장시간 데이터 테스트 46개가 통과했다. smoke는 전체 source 분리 검사를 거친 뒤
8 train/8 dev에서 gradient update, checkpoint 저장, 추론을 완료했다. smoke EER 0은
정확도 근거가 아니며, 실제 gradient/eval 구간은 약 17.85초였다.

전체 run의 명령과 PID는 `reports/paired_gain_wpt_v65/launch.json`, 학습 로그는
`paired_gain.log`에 기록한다. 실행 코드 6개와 config를 같은 run의 `source_snapshot/`에
보존했다. 실제 PID를 확인하며 학습 완료를 기다리는 개발 평가 작업도 시작했다.
v64는 GPU 0에서 그대로 계속 학습한다. locked 데이터 추론과 API 제출은 하지 않았다.

전체 run은 18,738 train/4,577 development 및 identity overlap 0을 확인했고,
첫 gradient step loss 6.88781 이후 계속 학습 중이다. source snapshot의 trainer/config
SHA-256은 실제 실행 manifest 및 원본 파일과 일치한다. 아직 전체 개발 평가 결과는 없다.

이후 epoch 2의 첫 전체 개발 평가가 완료됐다. File EER 0.245625,
selection 0.742357이며 대조군 동일 epoch의 0.232298/0.757634보다 나쁘다.
이는 중간 관찰이고, 사전 학습량 및 완료 판정을 유지한다.

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_paired_wpt_file_v60.py \
  --config configs/paired_gain_wpt_v65.yaml --variant paired_gain \
  --output reports/paired_gain_wpt_v65_smoke --smoke
python scripts/run_paired_wpt_v60.py --config configs/paired_gain_wpt_v65.yaml \
  --output reports/paired_gain_wpt_v65 --gpus 7
python scripts/evaluate_paired_gain_v65.py --wait-for-training \
  --output reports/paired_gain_wpt_v65/development_audit
```
