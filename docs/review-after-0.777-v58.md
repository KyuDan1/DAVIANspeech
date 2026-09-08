# 공식 0.77743 이후 방법론 재검토

2026-09-05. 목표는 실제 리더보드 1위 초과다. 사용자가 알려준 약 0.839를
계획 기준으로 쓰며, 최종 달성 여부는 당시 공식 1위와 비교해야 한다.

## 공식 결과와 식별 가능한 오차

| 제출 ID | 제출 | Total | ADS | CPS | 실행 시간 |
|---|---|---:|---:|---:|---:|
| 81325 | v18 | 0.7616379312 | 0.7363412698 | 0.9893078836 | 40:53 |
| 82673 | v18 Music constant probe | 0.7168950741 | 0.6866269841 | 0.9893078836 | 40:37 |
| 82658 | v50 | 0.7720307884 | 0.7478888889 | 0.9893078836 | 42:38 |
| 82697 | v50 + v57 Music | 0.7774307884 | 0.7538888889 | 0.9893078836 | 54:05 |

v18 probe는 Music만 상수 0.5로 만들었다. 따라서 다음이 식별된다.

```text
Music_EER_v18 = 0.5 - (0.7363412698 - 0.6866269841)/0.3
              = 0.3342857143
0.5*File_EER_v18 + 0.2*Voice_EER_v18 = 0.1633730159

Music_EER_current - Music_EER_v50
    = -(0.7538888889 - 0.7478888889)/0.3 = -0.0200000000
```

현재 Music EER이 0.31429라는 결론은 **도출할 수 없다**. v18과 v50의 ZIP
entrypoint를 대조하면 EAT patch graph와 component-query가 Music을 추가 변경했다.
File과 Voice의 현재 절대 EER 역시 미식별이다. 초기 twin의 EER을 현재 모델에
대입하지 않는다. 최신 anchor의 한 열 상수 probe를 추가하면 그 시점 절대값은
측정할 수 있지만, 이번 턴은 API 자동 제출을 설정하지 않는다.

같은 CPS에서 Total 0.8을 넘기려면 ADS > 0.7789657907, 0.839를 넘기려면
ADS > 0.8222991240이 필요하다. 현재 대비 각각 +0.0250769018,
+0.0684102351이며, 후자는 가중 EER의 약 27.80% 감소다.
CPS를 완벽하게 만들어도 Total 상승 한도는 0.0010692116이다.

## 실제로 드러난 병목

1. **학습 및 선택과 배포의 불일치.** v57은 세 task로 학습하고 ADS로 체크포인트를
   고른 뒤 Music만 배포했다. v2는 File 덕분에 전체 개발 ADS가 높지만 실제 사용되는
   Music이 좋아졌다는 보장은 없다. 배포되는 task EER로 checkpoint를 선택하는
   통제 실험이 우선이다.
2. **평가 분포 차이.** v57 Music은 pooled development에서 악화했지만 codec
   development와 공식 제출에서는 개선했다. pooled 수치만 최적화하지 않고
   원천/생성기/채널/단독/혼합별 결과와 표본 수를 함께 검토한다. 공식 테스트 구성이나
   혼합 비율을 이 세 점수로 역추정해 확정하지 않는다.
3. **학습 데이터의 원천 재사용.** 이전 v57에서 발견한 5개 identity에 해당하는
   train 6행을 제외한 strict split을 유지한다. 평가셋은 학습으로 옮기지 않는다.
   v9은 이미 노출됐으므로 다음 모델 선택에 사용하지 않는다.
4. **표현 학습의 한계.** 최근 residual들은 frozen 특징을 읽는 작은 head다.
   크기를 늘리는 것만으로 원래 표현이 놓친 흔적을 복원한다고 보장할 수 없다.
   기존 WPT는 prompt를 통해 backbone 표현을 바꾸므로, '우리는 전혀 표현 학습을
   안 했다'도 잘못이다. 다음 큰 변경은 WPT/adapter의 실제 mixed/codec 재학습을
   기존 head 학습과 분리해 검증해야 한다.
5. **실행 시간 여유.** 최신 공식 실행은 54:05로 여유가 5:55뿐이다. 원본 XLS-R
   캐시는 기존 음성/music **stem** 추론과 입력이 다르다. 코드상 같은 원본을 중복
   계산한 것은 아니므로 단순 캐시 재사용으로 삭제할 수 없다. 이 원본 stream을
   teacher로 사용하고 EAT/SPEAR student에 지식을 옮기는 방법은 검증할 가치가 있다.

## 연구와 연결

- [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090):
  speech용 사후학습의 근거다. 공개 speech 모델이 음악에도 동일하게 일반화한다고
  가정할 근거는 아니며 원본 mixture에 대한 별도 평가가 필요하다.
- [All-Type ADD / WPT](https://arxiv.org/html/2504.06753v1):
  speech, singing, music, sound 간 분포 차이를 다루며 SSL의 prompt를 학습한다.
  우리가 참고할 부분은 다유형 공동학습과 표현 적응이다. 논문 평균 EER 3.58%를
  전화 혼합 대회의 예상 EER로 사용하지 않는다.
- [MixFake](https://arxiv.org/html/2605.23201v1):
  전경과 배경의 진위를 교차 구성하고 신호 특징을 SSL prompt로 주입한다.
  상대 성분과 SNR에 관계없이 해당 성분의 진위를 구별하도록 학습하는 설계를
  참고한다. 이 저장소의 cached EAT/SPEAR/XLS-R head는 논문의 prompt 방법을
  그대로 재현한 모델이 아니다.

## 이번 실험: v58

`configs/music_specialist_v58.yaml`을 학습 전 고정했다. 같은 seed, 18,738개
strict train, 4,577개 development, 동일 특징/구조를 사용한다.

| 변형 | 학습 loss Voice/Music/File | checkpoint 선택 |
|---|---|---|
| joint_music_selection | 0.2 / 0.3 / 0.5 | Music 전용 |
| music_only_training | 0 / 1 / 0 | Music 전용 |

선택 점수는 `1 - (0.5*pooled Music EER + 0.25*mean-domain Music EER +
0.25*worst-domain Music EER)`이다. Music이 없는 domain은 해당 EER에서 제외한다.
30 epoch 상한, 2 epoch마다 평가, patience 6을 사전 고정했다.
두 결과는 기존 v57 Music과 같은 개발 행에서 비교한다. 배포할 경우 v57 head를
교체하므로 backbone 수를 늘리지 않는다. v58의 File/Voice 출력을 배포하지 않는다.

최소 pooled Music EER 1%p 개선, mean domain 개선, worst domain 비악화,
각 domain/channel Music EER 악화 2.5%p 이내를 후보 승격 조건으로 고정한다.
통과하지 않으면 기준을 사후 변경하지 않고 기각한다. 다음 prospective bank는
새 source뿐 아니라 가능한 범위에서 새 generator family를 분리하여 별도로 만든다.

### 실행 중 발견한 구현 결함과 수정

첫 `music_only_training`은 첫 epoch에서 중단됐다. Group-DRO가 Music 가중치
`[0,1,0]`에서 음악 없는 environment에도 Voice/File loss를 모아 가중치 합 0으로
나누고 있었다. 가중치 0인 task를 environment loss에서 제외하도록 수정했다.
음악 없는 전체 batch는 미분 가능한 0을 반환하며, 정상 joint loss는 바뀌지 않는다.
전용 회귀 테스트 2개로 finite loss/gradient와 비활성 task gradient 0을 검증했다.
앞으로는 non-finite loss/gradient가 optimizer state를 손상시키기 전에 중단한다.

실패 로그는 `reports/music_specialist_v58/`에 보존하고, 같은 seed/config로 재시작한
Music-only 결과는 `reports/music_specialist_v58_repaired/`에 별도 기록한다.
이는 하이퍼파라미터 탐색이 아니라 확인된 zero-weight 계산 버그의 수정이다.
원래 joint run은 수정 전 코드를 이미 load했지만 모든 task weight가 양수이므로
해당 분기 수정은 joint run의 수학적 계산을 바꾸지 않는다.

`scripts/evaluate_music_specialist_v58.py`는 등록된 development만 읽고, 정확히 같은
DATASET/ID의 Music 확률만 비교한다. RF/FF 등 단일 class 그룹에 억지로 EER을
만들지 않고, real/fake 표본 수와 고정 threshold 0.5의 FPR/FNR을 별도 보고한다.
이 threshold는 official EER을 최적화하거나 추정하는 threshold가 아니다.

### 논문 재검토에 따른 다음 큰 변경의 기준

[Hidden-Domain Routing](https://arxiv.org/html/2608.00493v1)은 유형별 전문가와
branch별 score 해석이 유효했던 AT-ADD 사례다. 하지만 그 유형별 binary Macro-F1
문제와 이 대회의 혼합 성분별 global EER는 다르다. 논문 1위 구조를 그대로 붙이는
것만으로 여기서 1위가 된다고 볼 수 없으며, 특히 채널 간 score offset도 검사해야 한다.

v55는 EAT-base adapter를 실제 학습했지만 새 출처에서 개선/중립/악화가 갈렸고,
v56도 EAT-large adapter+AASIST feasibility run이 이미 존재한다. 따라서 이것들을
새로운 미시도 방법처럼 다시 제안하지 않는다. 다음 표현 학습 변경은 **동일 시간
crop의 원음/실제 codec pair, 상대 성분 교체 전후의 component target 유지, 다양한
SNR에서의 mixed-input prompt/adapter 학습**으로 기존 시도와 차이를 명시해야 한다.
최신 제출의 추가 원본 XLS-R 연산을 없애기 위한 distillation은 성능 개선 실험과
별도 축으로 측정한다. 아직 구현/완료된 성과가 아니다.

[MixFake 공식 저장소](https://github.com/saltfish233/MixFake)는 재확인 시 학습 코드가
공개되어 있었으나, 이 확인에서 pretrained detector 배포 또는 코드 사용 라이선스는
확인되지 않았다. 소스를 무조건 복사하거나 pretrained 성과가 확보된 것처럼 주장하지
않고 논문 설계를 참고한다.

## v58 완료 결과와 다음 실험 v59

동일 development 4,577행 중 Music-present 4,261행(real 2,131/fake 2,130)을
비교했다. 아래 EER는 공식 제출 점수가 아니라 Music 전용 개발 지표다.

| 모델 | pooled Music EER | 최악 domain Music EER | codec v4 Music EER | 판정 |
|---|---:|---:|---:|---|
| 현 제출의 v57 Music head | 0.148791 | 0.260000 | 0.253333 | 유지 |
| v58 joint, Music 선택(epoch 12) | 0.134241 | 0.253333 | 0.253333 | 채널 보호 기준 미달 |
| v58 Music-only, cold start(epoch 0 선택) | 0.140812 | 0.310000 | 0.273333 | 개선 기준 미달 |

joint는 평균 domain EER도 1.734%p 개선했지만, clean 120행에서
`0.11667 -> 0.15000`, G.722 120행에서 `0.08333 -> 0.13333`으로 악화했다.
각각 real/fake 60/60행이다. 반대로 telephone8k 600행은 `0.19667 -> 0.16000`,
Opus NB 120행은 `0.36667 -> 0.33333`으로 개선했다. **전화라는 한 범주 안에서도
방향이 엇갈린다.** 전체 codec v4 EER이 같다고 모든 전화 변형에 안전한 것은 아니다.

Music-only는 12 epoch까지 Music 선택 점수가 초기와 동일해서 초기 identity head가
선택됐다. 따라서 표의 0.140812는 새 학습이 얻은 개선이 아니라 v57 residual 이전
v47/v50 Music 출력으로 돌아간 수치다. 학습 residual loss가 약 4/3까지 올라간 점은
Music 보정의 ±2 경계 포화와 부합하지만, 이 로그만으로 모든 보정이 동일한 상수였다고
단정하지 않는다. 다음 run에는 residual 평균/표준편차/포화 비율을 직접 기록한다.
이 최적화 실패를 '음악 전용 모델은 원천적으로 불가능하다'는 증거로 사용하지 않는다.

결과 파일:

- `reports/music_specialist_v58/audit_joint.json`
- `reports/music_specialist_v58/audit_music_only.json`
- `reports/music_specialist_v58/audit_incumbent_identity.json` (동일 모델 대조: 모든 차이 0)

후속 `configs/music_retention_v59.yaml`은 위 개발 결과를 본 뒤, v59 학습 전에
별도로 동결했다. 기존 v57 strict checkpoint에서 Music-only로 이어 학습하며,
learning rate를 1e-3에서 1e-4로 낮춘다. 두 변형 간 차이는 초기 모델의 **train 행**
logit을 유지하는 Smooth-L1 loss의 가중치 `0 / 0.2`뿐이다. 참조 logit은 학습 행에서
한 번 계산하고 detach하며, 개발/평가 행은 teacher 학습 target으로 쓰지 않는다.
추론은 기존 head 하나이므로 teacher나 backbone이 추가되지 않는다.

v58과 같은 pooled/domain/channel 승격 기준을 유지하고, 20 epoch 상한/patience 4,
같은 18,738 train/4,577 development를 사용한다. 이것은 개발셋 기반의 후속 실험이지
미관측 generalization 검증이 아니다. 두 후보 모두 실패하면 현재 최고 제출을 보존한다.

### v59 완료: 두 후보 모두 교체하지 않음

| 변형 | 선택 epoch | pooled Music EER | 최악 domain EER | 판정 |
|---|---:|---:|---:|---|
| warm_music (보존 loss 없음) | 0 | 0.148791 | 0.260000 | 기존 checkpoint 그대로, 개선 없음 |
| warm_music_retention (보존 loss 0.2) | 6 | 0.150199 | 0.253333 | pooled 악화 및 채널 기준 미달 |

보존 loss는 최악 domain의 악화를 줄였지만 pooled EER은 0.141%p 나빠졌다.
평균 domain 변화는 수치 오차를 제외하면 0이며, 최대 채널 악화는 3.399%p다.
개선이 `1e-18` 수준의 부동소수점 오차인 경우를 실제 개선으로 세지 않도록
evaluator의 양의 개선 판정에 `1e-12` 허용 오차를 적용하고 회귀 테스트를 추가했다.
이는 gate 실패를 통과로 바꾸는 예외가 아니며, 원래 결과도 후보 기각이었다.

결론: **4개 학습 실험 모두 제출 교체 기준을 통과하지 못했다.** 학습 손실을 더
낮추거나 기존 logit을 유지하는 제약만으로 새 성능을 얻지 못했다. v58 joint의
평균 개선과 채널 악화라는 trade-off를 확인했지만, 0.839 또는 새로운 공식 최고점은
달성하지 않았다. 현재 0.7774307884 ZIP은 수정하지 않았고 API 제출도 하지 않았다.
모든 학습 process는 정상 종료됐으며, 최초 v58 cold-start의 실패 로그만 별도로 남긴다.

검증 결과와 재현 파일은 `reports/music_retention_v59/`의 matrix manifest,
각 variant history/summary/checkpoint, `audit_warm.json`,
`audit_retention_verified.json`에 있다.

최종 회귀 검증: specialist selection, zero-weight Group-DRO, 초기 모델 보존 loss,
Music evaluator, 기존 three-stream head, matrix runner의 **60 tests passed**.
`git diff --check`도 통과했다. 기존 제출 ZIP과 사용자가 수정한
`src/artifactnet_detector.py`는 이번 작업에서 변경하지 않았다.

## 전체 방법론의 다음 우선순위

Music이 유일한 병목이라고 결론내리지 않는다. 기존에 계산된 정확한 v50+v57 Music의
codec v4 결과는 File/Voice/Music EER `0.23333 / 0.20667 / 0.25333`이다. 이 개발셋의
가중 오류 기여는 각각 `0.11667 / 0.04133 / 0.07600`으로 File이 가장 크다. 단,
이를 공식 test의 성분 EER로 바꿔 말할 수는 없다.

후속 우선순위는 다음과 같다.

1. 현재 최고 모델의 File 판단을 우선 대상으로 삼고, 실제로 개선을 보였던 WPT
   branch의 원본 mixed-input 표현 학습을 검토한다. 새 head 수나 mixing weight를
   늘리는 것과 다른 실험이다.
2. raw training에서는 하나의 시간 crop을 먼저 뽑은 다음 실제 codec을 적용한다.
   원음과 전화 음성을 별도로 random crop해서 내용 차이까지 consistency loss로
   지우는 문제를 피한다. narrowband뿐 아니라 G.722 같은 wideband도 포함한다.
3. 순차/희소 혼합에서는 해당 crop 안에 fake 성분이 실제로 들어있는지 확인한다.
   파일 전체 label을 fake 구간 없는 짧은 crop에 그대로 붙이는 방식은 오염된
   supervision이 될 수 있다. 합성 metadata의 구간 label 또는 전체 파일 MIL이 필요하다.
4. 현재 고정 평가셋으로 더 개선을 보여도 이를 새 일반화 증거로 부르지 않는다.
   후보를 고정한 뒤 새 source/가능한 새 generator로 구성한 bank에서 최종 비교한다.

위 raw-training 변경은 다음 구현 과제이며 이번 v58/v59의 달성 결과가 아니다.
공식 최신 ZIP의 기본 separator는 HTDemucs이고, 추가 v57 stream은 원본 mixture를
읽는다. 따라서 현재 최고 모델 전체가 '분리 없는 모델'이라고 표현해서는 안 된다.

## 이전 one-shot 운영의 정정 (상세)

v9에서 v57 Music의 ADS 개선은 +0.00125로, 사전 최소 +0.0025 기준을
통과하지 못했다. 이전 턴은 이후 탐색 목적으로 제출했으므로 **사전 게이트 예외**였다.
공식 점수가 올랐다는 사실이 이 절차상의 예외를 없애지는 않는다. 문서에서
'고정 게이트 통과 후 제출'로 기록하면 부정확하다. v9 채점기는 처음 absent
component의 결측 fake label을 거부했고, label 스키마 처리만 수정 후 채점했다.

v58의 체크포인트, 게이트, prospective 승인 조건은 결과를 보기 전에 기록한다.
공식 0.77743은 검증된 기준이고, 로컬 개선을 공식 1위 달성으로 부르지 않는다.
