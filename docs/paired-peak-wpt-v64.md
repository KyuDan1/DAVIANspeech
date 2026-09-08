# v64: peak normalization을 포함한 matched 재학습

상태: 구현·19개 단위 테스트·CUDA smoke 완료. GPU 0에서 전체 학습 진행 중.
공식 성능 개선이나 제출 후보 승격을 달성한 모델이 아니다.

## 근거와 비교 대상

v63의 추론-only normalization은 음량 변화에 대한 민감도를 크게 줄였다. Codec v4
600행의 Opus/재압축 내부 EER은 좋아졌지만 pooled File EER은 0.23333 → 0.24000으로
나빠져 탈락했다. 안정성과 정확도를 혼동하지 않는다.

v64는 학습·평가·배포 추론 모두 **preemphasis 이후, 각 4.0375초 창별 peak normalization**을
동일하게 적용한다. 원래 Spectra의 실제 설정도 normalization=False이므로 이것은
원본 전처리 누락을 고치는 작업이 아니라 새로운 학습 조건 실험이다.

대조군은 완료된 v60 `paired_bce`다. 다음은 모두 유지한다.

- 동일한 pretrained Spectra checkpoint에서 cold start; seed 20260905.
- strict train 18,738 / development 4,577과 동일한 source exclusion.
- 원음과 G.711/G.722/Opus/재압축의 같은 구간 pair, 동일한 sampler/gain augmentation.
- 전체 시간 coverage, File MIL의 LME T=2, File BCE+ranking, consistency loss 없음.
- batch/optimizer/LR, epoch당 4,096 draws, 최대 8 epoch, 2 epoch마다 평가, 같은 selection.

변경은 normalization 하나다. 기존 대조군을 다른 seed나 학습량으로 다시 돌리지 않는다.
새 long-voice 원천 및 2,880개 파생 파일은 모두 locked이며 train/checkpoint 선택에 쓰지 않는다.

## 구현과 재현성

`forward_bags(..., peak_normalize=False)`의 기본 경로는 유지한다. True일 때만
preemphasis 직후 각 유효 창을 `max(abs(window))+1e-8`로 나눈다. 패딩 창은 encoder에
들어가지 않고, 다른 파일·다른 창의 통계도 사용하지 않는다. config flag를 train/eval과
`src/full_coverage_wpt_inference.py` 모두 전달한다. 예전 checkpoint는 flag가 없으면 False다.

변경 전 v60 코드 4개를 `reports/paired_wpt_file_v60/source_snapshot/`에 보존했다.
trainer와 full-coverage source의 SHA-256은 v60 manifest에 기록된 값과 일치한다.
회귀 테스트는 별도 fixture의 v60 경로와 새 기본 경로의 출력을 bit-exact로 비교한다.
또한 두 YAML의 numerical/data/optimizer 설정이 metadata와 normalization 외에는
같다는 검사와 inference가 checkpoint flag를 실제 사용한다는 검사가 포함되어 있다.

## 평가 기준

학습 효과는 v60 BCE와 같은 4,577행에서 비교해 보고한다. 실제 승격의 주 비교는
기존 최고 전체 File에 대한 **고정 0.4 WPT slot 교체**다. 다른 네 출력은 변경하지 않는다.
같은 codec v4 600행에서 File EER 최소 1%p 개선, domain/channel 최대 악화 2.5%p 이하,
mean domain 개선과 worst domain 비악화를 요구한다. 비중 sweep은 하지 않는다.
직접 File 전체 교체 결과는 별도 진단이며, 서로 다른 비교의 성공 여부를 혼용하지 않는다.

개발 gate를 통과해도 새 all-type prospective, long-voice, 독립 추론 재현 및 전체 L4
runtime/오프라인 ZIP 확인이 남는다. 이 한 실험 통과가 SoTA나 공식 1위를 의미하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_paired_wpt_file_v60.py \
  --config configs/paired_peak_wpt_v64.yaml --variant paired_peak \
  --output reports/paired_peak_wpt_v64_smoke --smoke
python scripts/run_paired_wpt_v60.py --config configs/paired_peak_wpt_v64.yaml \
  --output reports/paired_peak_wpt_v64 --gpus 0
```

`--smoke`는 8 train/8 dev의 실행 진단이며 checkpoint를 제출에 사용하지 않는다.

## 실행 확인

CUDA smoke는 gradient update, checkpoint 저장 및 8개 dev 추론을 완료했다. 약 22.4초의
진단 실행이며 smoke EER 0은 성능 근거가 아니다. 실제 decoder/codec을 포함한 경로에서
nonfinite loss/gradient 오류는 없었다.

그 후 전체 run을 시작했고 18,738 train/4,577 development, strict identity overlap 0,
`smoke=false`와 첫 gradient step을 확인했다. 실행 PID와 명령은
`reports/paired_peak_wpt_v64/launch.json`, 로그는 `paired_peak.log`에 저장된다.
변경 후 학습/추론 source와 config도 이 run의 `source_snapshot/`에 보존했다.

Smoke checkpoint의 4/30/60초 synthetic inference와 파일 순서 반전도 검사했다.
B200 forward 평균은 각각 약 0.0243/0.0596/0.0375초이고 최대 할당 VRAM은 60초에서
약 2.49 GiB였다. 반복 수가 작아 시간은 비단조적이며, 디코딩/전체 pipeline/L4 시간의
증거가 아니다. 순서를 뒤집은 File 확률은 bit-exact였다.

종료 후 비교 작업도 실제 launch PID를 검증하며 기다리도록 시작했다.

```bash
python scripts/evaluate_paired_peak_v64.py --wait-for-training \
  --output reports/paired_peak_wpt_v64/development_audit
```

완료 checkpoint의 normalization flag 및 대조군과의 학습 조건 일치를 확인한다.
같은 4,577행의 학습 ablation, 600행 직접 교체 진단, **주 판정인 0.4 slot 교체**를
구분해 저장한다. 아직 새 후보 점수나 공식 성능 개선은 없고, API 제출도 하지 않았다.

## 중간 관찰 (학습 완료 아님)

Epoch 2: 전체 development 4,577행 File EER 0.251685, selection 0.737846.
같은 epoch의 v60 BCE는 0.232298/0.757634이므로 아직 개선하지 못했다.
이 값은 현재 최고 전체 pipeline의 600행 EER과 직접 비교할 값이 아니다.
남은 epoch와 사전 completion gate를 유지하며 중간 checkpoint로 locked 평가를 하지 않는다.

Epoch 4: File EER 0.226548, selection 0.757683으로 epoch 2보다는 개선했다.
대조군의 동일 epoch는 0.216932/0.767241이므로 여전히 뒤처진다. 완료 전 최종 판정은 없다.

## 완료 판정: 기각

8 epoch 학습과 자동 development audit가 종료됐다. 선택 checkpoint는 epoch 6이며
전체 4,577행 File EER은 0.228743이다. 같은 학습 조건의 v60 BCE 대조군
0.207841보다 2.09%p 나쁘다.

주 판정인 최고 pipeline의 WPT 0.4 slot 교체에서는 codec development 600행
File EER이 **0.233333 → 0.254444**로 2.11%p 악화했고, 최대 채널 악화는
3.89%p다. 다른 네 출력은 text-exact로 유지됐다. 따라서 제출에 채택하지 않는다.
peak normalization이 gain 민감도를 줄일 수 있다는 진단과 실제 판별 성능 개선은
다르다. 이 결과로 사전 승격 기준을 바꾸거나 locked 데이터를 추가로 열지 않는다.

근거: `reports/paired_peak_wpt_v64/development_audit/decision.json`,
`slot_vs_anchor.json`, `matched_training_ablation.json`. 최고 ZIP은 그대로 보존한다.
