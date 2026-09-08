# v67: 저대역 magnitude/phase Music-only MIL

상태: 두 전체 run과 고정 개발 판정을 완료했다. **현재 제출에 채택하지 않음**.
v64/v65 File 실험과 별개의 Music 표현 학습이다. 아래 학습 중 설명은 당시 이력이다.

## 최종 결과 — 2026-09-05

단독 Music EER은 magnitude 0.331378(best epoch 8), magnitude+phase
0.338888(best epoch 6)였다. 사전 기준으로 magnitude가 선택됐다.
최고 제출 Music logit에 고정 10%로 결합하면 codec dev Music EER은
0.253333 → 0.246667로 개선됐으나, 사전 최소 개선량 1%p에 미달했다.
채널별 비악화와 다른 네 출력의 text-exact 보존은 통과했다.
고정 gate 실패이므로 비중을 추가 탐색하거나 locked/API 평가를 진행하지 않는다.
근거: `reports/lowband_music_v67/development_audit/decision.json`.

## 왜 이 실험인가

v66 진단에서 최고 제출 Music EER은 clean 0.1167, Opus 0.3667,
G.711→Opus 0.4000이었다. 같은 가짜 음악의 점수가 Opus 후 약 0.30 떨어졌고,
진짜 음악은 오히려 약 0.08 상승했다. 전체 threshold만 옮겨 해결할 상황이 아니다.

기존 Music-selected WPT와 frozen EAT/SPEAR head 실험의 실패 기록을 인정한다.
v67은 해당 점수를 재조합하는 것이 아니라 **원본 혼합의 저대역에서 작은 CNN 전체를
직접 학습**한다. 위상 변화가 추가적인 구분 정보를 제공할 수 있다는 것은 우리의
검증 전 가설이지, 검증된 전화음악 SoTA 방법이라는 주장이 아니다.

## 구조

- 입력: 원본 16kHz mono 혼합. waveform separation/생성/복원 없음.
- 전체 구간 coverage를 보장하는 약 4.04초 창. 빈 패딩 창은 모델에 넣지 않는다.
- STFT 512/hop 160, 250–3,500Hz만 사용한다. 전화 대역에서 남는 정보에 집중하되
  3,500Hz까지 모든 코덱이 보존한다고 가정하지 않는다.
- 각 창 저대역 최대 power 대비 log-power. 파일 간/배치 간 통계는 사용하지 않는다.
- phase 채널: 이웃 시간 frame의 복소 cross-product를 unit magnitude로 정규화한
  실수·허수. 상대 에너지가 매우 낮은 bin은 0으로 마스킹한다.
- CNN 4층(16/32/64/128), GroupNorm, GELU, 평균+최대 pooling, linear Music head.
- 여러 창은 File-local LME T=2로 결합한다. Music 파일 수준 target만 사용하고,
  일부 구간에만 fake가 있는 경우 모든 창에 강제로 fake label을 붙이지 않는다.

두 모델은 같은 구조, 같은 seed/초기 가중치, 같은 학습 조건을 사용한다.
`magnitude`는 phase 두 채널이 항상 0, `magnitude_phase`만 실제 위상 변화를 넣는다.
비교 중 layer 수나 parameter 수를 바꾸지 않는다. 사전학습 weight는 없다.

## 학습과 누수 통제

기존 strict train 18,738행 중 Music-present만 학습에 사용한다. 성분이 없는 파일을
Music-real로 학습하지 않는다. 모든 source/speaker/song 제외 검사는 필터 전 전체
train/development에서도 수행한다. 개발 예측은 동일 4,577행을 저장하되 Music EER은
Music-present 행만 계산한다. v61 locked와 과거 retired blind는 사용하지 않는다.

corpus→Music class→component cell 순으로 sampler를 균형화한다. 즉 File의
real/fake 균형과 Music의 real/fake 균형을 혼동하지 않는다. FR과 RF도 학습에 포함하여
음성의 Fake 흔적만으로 Music을 판정하는 shortcut을 억제할 기회를 준다. 이것만으로
shortcut이 사라진다고 보장하지 않는다.

원음과 같은 구간의 G.711/G.722/Opus/재압축 pair에 Music BCE+0.1 ranking loss를
적용한다. 별도 consistency loss는 없다. gain [0.7,1.3], seed 20260905,
8 epoch/epoch당 4,096 draws, batch 8, AdamW 3e-4, 2 epoch마다 개발 평가다.
혼합 waveform을 codec 처리하며 반주와 음성을 별도 codec으로 처리하지 않는다.

## 사전 고정 판정

1. 두 모델의 학습이 끝난 뒤 전체 development의
   `1-(.5*pooled Music EER+.25*mean-domain EER+.25*worst-domain EER)`로 하나를 고른다.
   동률이면 단순한 magnitude 모델을 선택한다.
2. 선택한 모델만 현재 최고 Music logit에 **고정 10%** 결합하여 codec v4 600행에서
   주 판정한다. 나머지 네 출력은 문자열까지 유지한다. 비중 sweep은 하지 않는다.
3. Music EER 최소 1%p 개선, domain/channel 최대 악화 2.5%p 이하, mean domain 개선,
   worst domain 비악화를 요구한다. 단독 Music 결과는 별도 진단이다.
4. 통과해도 source/generator prospective, 파일 독립 추론 재현, 실제 전체 L4 runtime,
   오프라인 ZIP 검사 전에는 제출하지 않는다. 자동 API 제출은 없다.

이 모델은 기존 WPT File slot 교체가 아니다. 배포하면 새 작은 CNN 연산이 추가되므로,
현재 54:05짜리 전체 pipeline의 제한을 다시 측정해야 한다. B200 단독 속도로 60분
통과를 주장하지 않는다. 조건을 통과하지 못하면 기존 최고를 유지한다.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_lowband_music_v67.py \
  --variant magnitude_phase --output reports/lowband_music_v67_smoke --smoke
python scripts/run_lowband_music_v67.py --output reports/lowband_music_v67 --gpus 2,4
python scripts/evaluate_lowband_music_v67.py --wait-for-training \
  --output reports/lowband_music_v67/development_audit
```

기본 테스트는 저대역 shape, 음량/극성 변화, 무음 finite gradient, 패딩과 파일 순서,
두 ablation의 초기화 동일성, Music sampler의 target 균형을 검사한다. 이들은 성능
근거가 아니며, 실제 통신환경에서 위상이 유효한지는 개발·prospective 결과로 판단한다.

## 실행 검증

Smoke는 전체 source overlap 검사 후 8 train/8 dev에서 gradient update, 저장,
Music 추론을 완료했다. 학습/평가 구간은 약 15.66초, smoke Music EER 0.25다.
이 작은 진단 표본으로 성능을 주장하거나 config를 바꾸지 않았다.

완료된 smoke checkpoint를 독립 추론 코드에서 읽어 4/30/60초 synthetic 입력을 검사했다.
모델은 97,937 parameters다. B200의 파일당 평균 forward 시간은 각각 약
0.00225/0.01166/0.00455초, 최대 allocated VRAM은 60초에서 약 0.056 GiB였다.
파일 순서를 바꿔도 확률이 bit-exact였다. 반복 수가 적어 시간이 비단조적이고,
디코딩/전체 pipeline/실제 L4 실행 검증은 아니다.

`src/lowband_music_inference_v67.py`는 기본적으로 smoke checkpoint를 거부한다.
학습된 모델의 phase/config 일치 및 state shape를 검증하고, 각 파일 내부의 모든
창만 사용한다. 평가 후 사용할 경우에도 배포용 checkpoint로 다시 검증해야 한다.

전체 run PID/명령은 `reports/lowband_music_v67/launch.json`, variant별 로그는
`magnitude.log`, `magnitude_phase.log`다. 실제 PID와 명령을 확인하며 기다리는
평가 작업도 시작했다. 코드·설정은 같은 run의 `source_snapshot/`에 보존했다.
현재 locked 추론, ZIP 제작, API 제출은 하지 않았다.

두 전체 run 모두 Music-present train 18,010 / development 4,577,
strict identity overlap 0 및 `smoke=false`를 확인했다. 첫 step loss는 magnitude
0.81485, magnitude+phase 0.79441로 finite다. 이 첫 loss 차이는 모델 우열 근거가 아니다.
