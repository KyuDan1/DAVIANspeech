# v60: 전체 시간 구간을 보는 WPT File 학습

상태: 두 GPU의 전체 학습 및 사전 development 비교 완료. 두 직접 교체 후보 모두 탈락.
공식 성능 개선을 달성한 모델이 아니다.

## 공식 페이지 확인과 적용

2026-09-05 사용자 제공 [공식 개요](https://dacon.io/competitions/official/236749/overview/description),
[평가](https://dacon.io/competitions/official/236749/overview/evaluation),
[규칙](https://dacon.io/competitions/official/236749/overview/rules)을 직접 확인했다.

- 전화 채널은 일부 샘플이며 그 비율이나 실제 범죄 녹음 여부는 공개되어 있지 않다.
  phone-only 최적화나 평가셋의 균형 분포를 전제하지 않는다.
- 동시·순차 혼합 모두 대상이며, 반주가 있는 보컬 노래는 혼합이다.
  단순 후처리가 적용된 실제 음원은 Fake로 바꾸지 않는다.
- File EER의 총점 계수는 0.45, Music은 0.27, Voice는 0.18이다.
  성분 EER는 해당 성분이 존재할 때만 평가한다.
- 파일 안의 segment 결합은 허용되지만, 다른 평가 파일의 통계·예측을 이용하거나
  비공개 데이터로 추가 학습하는 것은 금지된다.
- 60분/L4 22.4 GiB, 설치 10분, ZIP 10GB/해제 32GB, 하루 3회 제한을 유지한다.
- 2차 평가에는 재현 가능한 학습 코드와 학습데이터 일체, HWP 보고서 2개가 요구된다.
  연구용 Markdown은 최종 HWP 제출물을 대신하지 않는다.

## v58/v59 이후의 변경

앞선 4개 frozen-feature Music head 실험은 교체 기준을 통과하지 못했다.
이번에는 **raw original mixture → WPT-XLSR/Spectra-AASIST → File logit**을 실제로
학습한다. 다른 head를 누적하는 실험이 아니라 File expert 교체를 위한 연구다.
기존 외부 Spectra checkpoint(로컬 모델 카드 Apache-2.0)에서 시작하고, 이전 대회
학습 WPT checkpoint는 가져오지 않는다. 같은 source exclusion을 보증하지 못하는
이전 가중치에서 시작해 새 strict split이라고 주장하지 않는다.

### 1. 파일 전체 coverage

기존 짧은 crop 일부에는 파일 안의 fake 구간이 들어있지 않을 수 있다. 이를 피하려고
64,600-sample 창을 겹치거나 맞닿도록 배치하여 모든 원본 sample을 적어도 한 번
포함한다. 60초에서는 15개 창이 된다. 짧은 파일만 repeat padding한다.

창별 예측을 LME(T=2)로 결합한 **파일 출력에만 File label**을 적용한다.
가짜가 없는 창에 파일의 fake label을 강제로 붙이지 않는다. 시간 구간별 정답을
추측하거나 평가 데이터에서 얻는 과정은 없다. batch padding 창은 encoder에 넣지
않고 pooling에서도 제외한다.

### 2. 원음/실제 codec pair

전체 파일의 창 위치를 먼저 결정한 뒤, 정확히 같은 창에 G.711/G.722/Opus/
double-transcode 중 하나를 적용한다. 두 view를 별도로 random crop하지 않는다.
이는 별도의 내용 차이를 channel 차이로 잘못 학습하는 것을 방지한다. codec frame
padding만 끝부분에서 정리하며, 생성형 source separation은 사용하지 않는다.

두 후보 모두 동일한 pair와 File BCE, pairwise ranking을 학습한다.

| 후보 | 원음+codec BCE | ranking | channel→clean logit consistency |
|---|---:|---:|---:|
| paired_bce | 동일 | 0.10 | 0 |
| paired_consistency | 동일 | 0.10 | 0.20 |

따라서 두 후보의 차이는 augmentation 유무가 아니라 consistency loss 하나다.
clean target logit은 detach하고 양쪽 view 모두 정답 BCE로 학습한다. Presence,
Voice/Music 출력의 개선을 주장하지 않으며 최종 배포한다면 File 하나만 바꾼다.

### 3. 데이터와 선택

v58의 등록된 strict train 18,738/development 4,577행을 재사용하고 원본 identity
제외 6행을 유지한다. raw loader도 role manifest를 직접 해석하고 voice/music/parent
및 확장 identity 중복 검사를 수행한다. Suno holdout, OOD/stress, 이미 노출된 v9
이전 bank는 학습이나 checkpoint 선택의 입력이 아니다.

train sampling은 corpus → File real/fake → 성분 조합의 순서로 균형화한다.
실제 test class ratio의 추정은 사용하지 않는다. checkpoint는 File 기준
`1-(.5 pooled EER + .25 mean-domain EER + .25 worst-domain EER)`로 고른다.
8 epoch 상한, epoch당 4,096 draws, 2 epoch마다 평가, patience 3을 사전 동결했다.

## 승격 조건과 한계

paired_bce보다 pooled File EER 1%p 이상 감소, mean-domain 개선, worst-domain
비악화, domain/channel EER 악화 2.5%p 이하가 1차 기준이다. 이를 통과해도 실제
최고 v50+v57 전체 파이프라인과 File 교체 비교를 해야 한다. 개발 개선만으로 제출하지
않고 후보 고정 후 prospective 및 실제 runtime 검증이 추가로 필요하다.

Full coverage는 긴 파일의 연산량을 늘린다. 최신 공식 runtime이 54:05이므로 기존
WPT와 중복 실행하거나 runtime 측정 없이 ZIP에 더할 수 없다. 필요하면 학습된
teacher를 저비용 student에 옮기는 별도 실험이 필요하다. 아직 구현한 성과는 아니다.

## 재현

설정: `configs/paired_wpt_file_v60.yaml`

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_paired_wpt_file_v60.py \
  --variant paired_bce --output reports/paired_wpt_file_v60/paired_bce
CUDA_VISIBLE_DEVICES=7 python scripts/train_paired_wpt_file_v60.py \
  --variant paired_consistency --output reports/paired_wpt_file_v60/paired_consistency
```

`--smoke`는 full manifest 누수 검증 후 8개 train/8개 dev만 사용한다. smoke checkpoint는
명시적으로 `smoke=true`이며 학습 성능이나 제출 후보로 사용하지 않는다.
ffmpeg PATH 자동 발견이 실패한 초기 smoke는 학습 전에 종료했으며, Python 환경의
ffmpeg fallback을 추가한 새 디렉터리에서 검증을 재시작했다. 실제 G.711/G.722/Opus
roundtrip까지 포함한 window/codec tests 11개는 모두 통과했다.

## 실행 확인

- CUDA smoke는 실제 gradient update, compact checkpoint 저장, 재현 manifest 및
  8개 development 추론을 완료했다. smoke EER 0은 작은 진단 표본의 값이며 성능 근거가 아니다.
- window/실제 codec, paired collation/sampling, 기존 WPT, component evaluator를
  포함한 **24 tests passed**. deprecated `TRANSFORMERS_CACHE` 경고만 남아 있다.
- 전체 run은 `scripts/run_paired_wpt_v60.py --output reports/paired_wpt_file_v60 --gpus 0,7`
  로 시작했다. 두 run 모두 18,738 train/4,577 development와 strict identity overlap 0을
  확인하고 1 epoch step 200까지 gradient update를 정상 수행했다.
- PID/명령은 `reports/paired_wpt_file_v60/launch.json`, 실시간 로그는 각 variant의
  `.log`에 기록된다. 프로세스 완료 여부는 실제 PID와 supervisor 상태로 재확인해야 한다.
- 종료 후 비교 명령(두 예측 파일이 완성된 다음 실행):

```bash
python scripts/evaluate_music_specialist_v58.py \
  --task file --matrix configs/paired_wpt_file_v60.yaml \
  --incumbent reports/paired_wpt_file_v60/paired_bce/dev_predictions.csv \
  --candidate reports/paired_wpt_file_v60/paired_consistency/dev_predictions.csv \
  --output reports/paired_wpt_file_v60/file_comparison.json
```

evaluator는 역사적인 파일명을 유지하되 File/Voice/Music task를 지원한다. File은
성분 presence로 샘플을 제외하지 않는다. 현재 새 ZIP 생성이나 API 제출은 하지 않았다.

## 추론 구현 및 실측 (성능 평가 아님)

`src/full_coverage_wpt_inference.py`는 파일마다 독립 호출한다. 다른 파일의 예측,
길이, 통계 또는 batch 구성을 참조하지 않는다. `smoke=true` checkpoint는 기본적으로
거부하며 진단 옵션을 명시한 경우에만 사용할 수 있다.

CUDA smoke checkpoint와 synthetic waveform으로 NVIDIA B200에서 측정했다.
모델 load 약 9.91초 이후 warm-up을 하고, 길이별 forward 3회 평균을 기록했다.
디코딩/전체 앵커 실행/패키지 설치는 포함하지 않았다.

| 길이 | 창 수 | 평균 forward/파일 | 최대 할당 VRAM |
|---|---:|---:|---:|
| 4초 | 1 | 0.0259초 | 1.275 GiB |
| 30초 | 8 | 0.0456초 | 1.883 GiB |
| 60초 | 15 | 0.0430초 | 2.492 GiB |

반복 수가 적은 microbenchmark라 길이별 시간은 잡음과 batching 효율에 영향을 받는다.
이를 L4 처리량이나 전체 제출의 60분 통과 증거로 사용하지 않는다. 순서를
4/30/60에서 60/30/4로 뒤집었을 때 각 File 확률은 bit-exact였다.
`reports/paired_wpt_file_v60_smoke_repaired/runtime_independence.json`에 원자료가 있다.
추론/codec/학습 데이터 collation/evaluator 테스트 21개가 추가 실행에서 통과했다.

## 후보 판단 순서 명확화 (v60 개발 결과 확인 전)

`paired_consistency` 대 `paired_bce`는 loss 하나의 효과를 묻는 ablation이다.
이 비교에서 이기지 못했다고 해서 대조 모델을 자동 폐기하지 않는다. 두 모델 모두
기존 최고 제출의 File 출력과 별도 비교하며, **실제로 앵커를 이기는 모델**이 후보가
될 수 있다. 기존 최고의 File과 Voice는 v50 이후 v57 Music-only 추가로 바뀌지 않았다.

정확한 v50 File 출력이 있는 authorized codec v4 600행에서 먼저 동일 키로 비교한다.
이 부분집합에서도 사전 1%p File EER 개선과 domain/channel 2.5%p 보호 기준을
적용한다. 두 후보가 모두 통과하면 전체 4,577행에서 사전 정의한 File 선택 점수가
높은 모델 하나를 동결한다. 둘 다 실패하면 baseline ZIP을 유지한다.
이후 prospective/runtime 관문은 별도로 남는다. 혼합 weight sweep이나 probe 제출은
이번 비교에 포함하지 않는다.

```bash
python scripts/evaluate_music_specialist_v58.py \
  --task file --matrix configs/paired_wpt_file_v60.yaml \
  --datasets codec_mixed_dev_v4 \
  --incumbent reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv \
  --candidate reports/paired_wpt_file_v60/paired_bce/dev_predictions.csv \
  --output reports/paired_wpt_file_v60/bce_vs_official_anchor_dev.json
```

`paired_consistency`도 별도 output 경로로 같은 비교를 한다. 이 600행은 development이며
공식 test 또는 새로운 blind 성과가 아니다.

## 공식 길이 조건과 개발셋의 빈틈: 실제 헤더 감사

등록된 development 4,577행의 실제 오디오 헤더를 모두 읽어 확인했다. 모델 예측이나
보호된 평가셋을 읽은 결과가 아니다. 재현 스크립트는
`scripts/audit_development_durations_v60.py`, 원자료는
`reports/paired_wpt_file_v60/development_duration_audit_v2/{headers.csv,summary.json}`이다.

| 항목 | 현재 development |
|---|---:|
| 길이 중앙값 | 5.543초 |
| 최장 파일 | 20.2초 |
| 4초 미만 | 1,232/4,577행 (26.9%) |
| 30초 이상 | 0행 |
| 기존 앵커 비교용 codec v4 최장 | 17.446초 |

공식 입력은 **4~60초**이므로, 현재 개발셋은 긴 파일의 검증을 놓치고 대회 범위보다
짧은 파일도 상당수 포함한다. 4초 미만 중 1,122행은 MixFake Music 개발 코퍼스이고,
나머지 110행은 external mixed 원음/전화 변형이다. 원음과 전화 변형은 독립적인 원천
자료 개수로 세지 않는다. 이 숫자로 실제 test 길이 분포나 실패 원인을 확정할 수 없다.

길이 4.0375초인 창 5개는 최대 20.1875초만 볼 수 있어, 60초 파일에서는 적어도
39.8125초를 보지 못한다. 이는 crop의 시간 예산에 따른 하한이며 **탐지 실패율이
66.4%라는 뜻은 아니다**. 가짜 구간의 위치와 길이에 따라 결과는 달라진다.
현재 개발 파일에서는 이 5-window 예산 초과가 단 2행뿐이므로, 짧은 개발셋 성능만으로
full coverage의 긴 파일 이점을 입증할 수 없다.

진행 중인 두 v60 run의 학습 데이터, checkpoint 선택, 사전 기준은 변경하지 않는다.
특히 지금 짧은 파일을 제거하고 이전 결과와 직접 비교하거나 development를 train으로
옮기지 않는다. 다음 평가 bank에는 source identity가 겹치지 않는 30/45/60초 자료,
동시/순차 혼합, 짧은 fake 구간의 초반/중간/후반 배치와 동일 편집의 real 대조군이
필요하다. clean/전화 조건을 짝지어 검사하되, 길이별·성분별 결과를 별도로 보고한다.
이 bank는 아직 구축 완료가 아니며, 기존 개발 음원을 늘여 만든 실험은 새로운
source-disjoint prospective 성과로 부르지 않는다.

## 첫 중간 평가 (epoch 2, 최종 후보 아님)

| run | pooled File EER | 사전 정의 selection |
|---|---:|---:|
| paired_bce | 0.23229825 | 0.75763407 |
| paired_consistency | 0.23767876 | 0.75723227 |

각 run의 첫 평가까지 학습 타이머 기준 약 1,286초가 걸렸으며 이후 epoch 3이 진행 중이다.
이 시점에서는 consistency가 더 좋다는 증거가 없다. 이는 현재 개발 4,577행의 중간
결과이며, 기존 최고 제출의 공식 EER 또는 별도 600행 EER과 직접 비교하지 않는다.
중간 결과를 보고 weight나 선택 기준을 수정하지 않고, 완료된 checkpoint로 사전
비교를 수행한다. 추가 테스트 실행은 21 passed, `git diff --check`도 통과했다.

epoch 4 중간 결과는 paired_bce File EER 0.21693168 / selection 0.76724064,
paired_consistency 0.21844683 / selection 0.76378654이다. 둘 다 epoch 5로 진행했다.
이는 여전히 최종 후보/공식 성과가 아니다.

`scripts/finalize_paired_wpt_v60.py`가 실제 launch PID의 명령을 검증하며 종료를 기다린다.
두 run의 complete/checkpoint/history/예측 행 수를 확인한 뒤, 두 후보 ablation 및
각 후보 대 실제 최고 File의 codec v4 600행 비교를 실행한다. 비교 도중 artifact hash가
바뀌면 실패하며, 고정된 개발 관문을 통과한 후보만 전체 개발 selection으로 하나를
고른다. 이 결과는 prospective/전체 runtime 확인이나 제출 자체가 아니다.

```bash
python scripts/finalize_paired_wpt_v60.py --wait-for-training \
  --output reports/paired_wpt_file_v60/final_development_audit
```

## 최종 v60 결과

두 run 모두 8 epoch를 완료했다. 학습 타이머 기준 약 4,644초이며, 완료 marker와
checkpoint/history/예측 hash 검증까지 끝났다. BCE는 epoch 6, consistency는 epoch 8이
각 사전 selection의 최적 checkpoint였다.

| run | 전체 개발 4,577행 File EER | selection | 앵커 비교 600행 File EER |
|---|---:|---:|---:|
| paired_bce | 0.20784081 | 0.78012548 | 0.30111111 |
| paired_consistency | 0.21776675 | 0.78071506 | 0.26777778 |
| 기존 최고 전체 pipeline | 이 실험에서 전체 4,577행 재평가 안 함 | 해당 없음 | 0.23333333 |

전체 개발 수치와 600행 수치는 다른 모집단이다. 새 모델의 전체 개발 EER이 낮다고
기존 최고를 이겼다고 해석할 수 없다. Consistency는 최악 domain을 줄여 selection이
조금 더 높았지만 pooled/평균 domain을 개선하지 못했다. 600행에서도 두 후보 모두
기존 최고보다 나쁘고 채널 보호 기준을 위반했다. `frozen_candidate=null`이 최종 결정이다.
긴 음성 locked bank는 이 실패를 보고 다시 선택하는 데 사용하지 않았으며 아직 unscored다.
