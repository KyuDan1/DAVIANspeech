# v62: 기존 WPT 자리만 교체하는 비교

상태: 기존 제출 재현 및 slot 개발 비교 완료. 전체 EER 동률/채널 악화로 탈락.
공식 성능 개선은 아직 없다.

## v60과 무엇이 다른가

v60은 새 full-coverage WPT의 File 출력을 단독으로 기존 전체 File 출력과 비교한다.
이 기준은 그대로 유지한다. 반면 실제 최고 v50+v57 제출은 File을 결정할 때 여러
모델의 근거를 결합하고, 그중 기존 WPT의 비중이 0.4다.

v62는 새 모델을 덧붙이거나 나머지 모델을 버리지 않고 **WPT 자리 하나만 교체**한다.
비중은 제출 코드의 0.4를 그대로 쓰며 sweep하지 않는다. 다른 네 출력은 문자열까지
그대로 유지한다. 학습 checkpoint는 완료된 두 v60 run 중 사전 정의한 전체 development
selection이 높은 것을 택한다. 600행에서 두 slot 후보를 골라보는 방식은 아니다.

| 실험 | File 출력 | 다른 네 출력 |
|---|---|---|
| v60 직접 교체 | 새 WPT만 사용 | 실제 패키지 적용 전 별도 확인 필요 |
| v62 slot 교체 | 기존 나머지 모델 0.6 + 새 WPT 0.4 | 기존 최고 제출 그대로 |

새로운 모델 구조/학습의 효과와 기존 앙상블 유지의 효과를 구별하는 development 실험이다.
v60이 실패하더라도 그 gate를 통과한 것으로 바꾸지 않는다.

## 정확한 비교 계산

확률 p의 logit을 L(p)라 하면 제출된 WPT 결합은 다음과 같다.

`L(final) = 0.6 L(base) + 0.4 L(old_WPT)`

따라서 중간 base 출력을 다시 저장하지 않아도 개발 비교에서 다음 식으로 교체할 수 있다.

`L(replaced) = L(final) + 0.4 [L(new_WPT) - L(old_WPT)]`

이는 `final`에 WPT를 다시 더하는 방식이 아니다. logit clipping은 기존 구현과 동일한
1e-5이며, 복원된 base가 그 범위를 벗어나면 조용히 자르지 않고 비교를 실패시킨다.
동일 WPT로 바꿨을 때 File도 bit-exact로 유지되도록 처리한다.

이 algebra는 **개발 비교용**이다. 배포 시에는 실제 파이프라인의 해당 WPT 호출을
새 모델로 교체한다. 구 WPT와 신 WPT를 모두 실행하여 runtime을 늘리는 방식이 아니다.
WPT 이후 File에 다른 변경이 없는 현재 v50+v57 graph에만 적용할 수 있다.

## 기존 제출과의 연결 확인

`scripts/audit_wpt_anchor_v62.py`는 최고 ZIP과 펼친 디렉터리 사이의 코드·가중치·설정
40개 파일을 SHA-256으로 비교했다. 현재 dirty worktree 코드 대신 그 **제출 패키지의
추론 코드**를 불러, authorized codec development 600행에서 다시 실행했다.

- 기존 WPT 5 views, temperature 2, batch 6.
- 기존 캐시와 Voice/Music/File 확률 최대 차이 **0.0**.
- NVIDIA B200에서 해당 WPT 재추론 약 17.60초. 전체 pipeline/L4 측정이 아니다.
- 기존 expert를 자기 자신으로 바꾸는 600행 identity check: File bit-exact.
- 복원한 base logit 범위 -3.0268~10.0475로 기존 clipping 범위 안에 있다.
- Slot 재결합/키 정렬/다른 네 출력 문자열 보존/불일치 거부 테스트 8개 통과.

원자료: `reports/wpt_slot_v62/anchor_verification/{predictions.csv,verification.json}`.
이 결과는 기존 계산을 재현한 것이지 새 모델 성능 향상이 아니다.

## 기존 expert와 전체 앙상블의 역할 차이 (개발셋 진단)

같은 authorized codec v4에서 기존 WPT 단독과 최고 제출의 최종 File 출력을 비교했다.
새 v60/v62 후보의 점수는 사용하지 않았다. RR은 진짜 음성+진짜 음악이며 FR/RF/FF
각각을 RR과 비교한 EER다. 이는 Voice/Music 성분 EER 또는 공식 test EER가 아니다.

| 조건 | 기존 전체 File EER | 기존 WPT 단독 File EER |
|---|---:|---:|
| 전체 600행 | 0.23333 | 0.28000 |
| 가짜 음성+진짜 음악 vs RR | 0.26000 | 0.27333 |
| 진짜 음성+가짜 음악 vs RR | 0.26667 | 0.35333 |
| 둘 다 가짜 vs RR | 0.18000 | 0.23333 |
| clean | 0.10000 | 0.16667 |
| G.711 | 0.20000 | 0.23333 |
| G.722 | 0.16667 | 0.23333 |
| Opus NB | 0.33333 | 0.33889 |
| G.711→Opus | 0.30000 | 0.33333 |

다른 expert를 유지할 근거는 특히 RF에서 크다. 기존 전체도 Opus에서 clean보다
훨씬 나쁘므로, 평균 하나만 보고 새 File expert를 채택하면 안 된다. 이 결과는
해당 개발 bank의 관찰이며 비공개 test의 구성비나 오류 원인 비율을 역산하지 않는다.

## 선택·승격 조건

`configs/wpt_slot_v62.yaml`을 후보 slot 점수 확인 전에 동결했다. 기존 전체 File과
같은 codec v4 600행에서 File EER 1%p 이상 개선, 채널/도메인 최대 악화 2.5%p 이하,
평균 도메인 개선 및 최악 도메인 비악화를 요구한다. 통과하더라도 파일별 독립 추론
재현, 새로운 all-type prospective, 긴 음성 평가, 전체 L4 runtime, 오프라인 패키지
검증이 남는다. 자동 제출은 하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/audit_wpt_anchor_v62.py \
  --output reports/wpt_slot_v62/anchor_verification
python scripts/evaluate_wpt_slot_v62.py --wait-for-training \
  --output reports/wpt_slot_v62/development_audit
```

두 run의 실제 PID와 명령이 일치하는 동안에만 기다린다. 완료 marker, checkpoint,
history, prediction 및 해시 검증 후 평가하며, 미완료 모델을 성능 후보로 다루지 않는다.

## 최종 결과

사전 전체-development selection에 따라 `paired_consistency` epoch 8이 선택됐다.
고정 0.4 slot 교체 후 600행 File EER은 **0.23333333 → 0.23333333**으로 동률이다.
G.711/G.722/G.711→Opus의 EER이 각각 3.333%p 악화해 2.5%p 보호 기준도 위반했다.
Opus 단독은 0.33333333 → 0.30555556으로 좋아졌으나, 이 부분만으로 승격하지 않는다.

최종 `pass_gate=false`. 다른 네 출력은 문자열까지 유지됐다. 새 ZIP이나 API 제출은
하지 않았고, locked bank를 이 후보 선택에 사용하지 않았다. 원자료는
`reports/wpt_slot_v62/development_audit/{comparison.json,decision.json}`이다.
