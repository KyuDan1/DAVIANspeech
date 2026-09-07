# v39: WPT all-type expert와 router 대 고정 MoE 검증

## 결론

v39는 v38을 그대로 앵커로 두고, source separation을 거치지 않은 원본 오디오를
보는 `WPT-XLS-R-300M + Spectra-AASIST` 전문가를 추가한다. 새 전문가는
Voice/Music/File을 함께 학습했지만 최종 배포에서는 검증상 일관된 **Voice와
File 출력만** 쓴다. Music은 v38의 unified EAT/SPEAR 결과를 그대로 유지하고,
두 presence 열도 전혀 바꾸지 않는다.

중간 latent router, phone-aware router, hard router, 고정 task별 MoE를 동일 분할로
비교했다. 학습형 soft router는 개발점수가 고정 MoE보다 `+0.00188` 높았지만
phone locked ADS가 `-0.00693` 낮았다. hard router는 개발 선택점수부터
`-0.052~-0.081` 하락했다. 따라서 hidden domain 일반화를 우선해 고정 MoE를
채택했다.

## 1. 문제 정의와 모델

### 왜 별도 전문가가 필요한가

현재 hidden 성능의 병목은 CPS가 아니라 File/Music EER이며, 특히 실제·가짜
음성과 음악이 섞인 교차 성분 조건에서 한 전문가의 강점이 모든 축에 일치하지
않는다. source separator는 생성 흔적을 지우거나 자체 흔적을 추가할 수 있으므로,
새 전문가는 stem이 아닌 원본 waveform만 사용한다.

새 모델은 [Wavelet Prompt Tuning 기반 all-type 탐지 연구](https://arxiv.org/abs/2504.06753)를
독립 재구현했다.

- backbone: frozen XLS-R-300M
- 각 transformer layer: 일반 prompt 6개 + 학습 가능한 2-D Haar WPT prompt 4개
- backend: Spectra-AASIST graph attention
- 입력: 파일 전체에서 균등하게 뽑은 4.04초 view 3개
- pooling: 온도 5의 log-mean-exp
- 출력: Voice, Music, File authenticity 3개
- 학습 파라미터: 820,399개, compact checkpoint 약 3.3MB

prompt와 AASIST backend는 공동 학습하되 XLS-R backbone은 고정했다. Voice/Music
loss는 해당 성분이 존재하는 행에서만 계산하고, File loss와 component-OR
consistency, EER를 근사하는 pairwise ranking loss를 함께 사용했다.

### 학습 및 평가 분리

학습에는 다음 8개 bank, 총 18,936개 메타데이터 행만 사용했다.

- `forensic_call_train_v1`
- `external_mixed_train_v1`
- `mixed_devvoice_train_v1`
- `mixed_fmc_music_train_v1`
- `mixfake_music_train_v1`
- `telephone_mixed_train_v1`
- `temporal_mixed_train_v2`
- `channel_invariant_factorial_train_v1`

checkpoint 선택에는 source/generator가 분리된 개발 bank 8개를 사용했다.
`factorial holdout`, `phone factorial`, `YuE cross-component`, Suno audit는 학습과
선택에서 잠갔다. data guard로 train 및 router-train 경로가 locked source를
참조하지 않음을 다시 검사했다.

## 2. WPT standalone 결과

개발 선택 seed `20260906`, epoch 7의 결과다. 제출과 동일한 batch=8에서도 EER가
batch=12 결과와 같았다.

| 평가축 | Voice EER | Music EER | File EER | ADS |
|---|---:|---:|---:|---:|
| factorial dev | 0.2114 | 0.2686 | 0.2153 | 0.7695 |
| factorial holdout | 0.1829 | 0.2514 | 0.2018 | 0.7871 |
| phone factorial | 0.1575 | 0.1325 | 0.1183 | **0.8696** |
| YuE cross-component | 0.2125 | 0.1935 | 0.1277 | 0.8356 |

Suno 보컬 음악에서는 File과 Music 모두 원본 13/13, codec stress 65/65가 0.5를
넘었다. 반면 Voice는 원본 8/13, stress 35/65만 0.5를 넘었다. 즉 Suno 파일 전체와
음악 성분 탐지는 강하지만, 생성된 노래의 보컬을 모든 조건에서 독립적인 fake
voice로 식별한다고 과장할 수는 없다.

## 3. 중간 router 실험

router에는 단순 최종 확률만 넣지 않았다.

- unified EAT/SPEAR task별 pre-classifier latent: 864차원
- WPT-AASIST view mean/std/max latent: 480차원
- Voice/Music presence와 상호작용: 5차원
- 선택 실험에서 phone probability와 불확실도: 3차원
- expert token 두 개에 attention을 적용한 뒤 task별 결합 가중치 출력

학습은 별도 router-train bank 두 개에서만 하고, 7개 dev bank로 선택한 뒤 세
locked bank를 한 번 평가했다. 모델이 내는 가중치는 검증된 고정 prior 주변으로
제한했다. prior의 `(unified, WPT)` 비율은 Voice `(0.10, 0.90)`, Music
`(0.25, 0.75)`, File `(0.20, 0.80)`이다.

### 개발 비교

| 방법 | 선택점수 | 평균 ADS | 최저 ADS | 고정 대비 |
|---|---:|---:|---:|---:|
| 고정 task별 MoE | 0.84773 | **0.87796** | **0.81790** | - |
| bounded soft router 0.10 | **0.84960** | 0.87783 | 0.81738 | +0.00188 |
| bounded soft router 1.00 | 0.84761 | 0.86915 | 0.80951 | -0.00012 |
| hard router 최선 | 0.79543 | 0.83856 | 0.77636 | -0.05230 |

### locked standalone 비교

| 평가축 | 고정 MoE ADS | soft router ADS | router - 고정 |
|---|---:|---:|---:|
| factorial holdout | 0.82517 | 0.82556 | +0.00039 |
| phone factorial | **0.88757** | 0.88064 | **-0.00693** |
| YuE cross-component | 0.86231 | **0.86611** | +0.00381 |

phone feature를 제거해도 결론은 거의 같았다. 학습된 평균 WPT weight도 Voice
0.826, Music 0.682, File 0.730으로 샘플 간 변화가 작았다. phone 여부는 채널을
잘 알려주지만, 같은 phone 채널 안에서도 generator와 성분 조합에 따라 어느
expert가 옳은지는 달랐다. 그리고 대회 샘플은 speech/music이 동시에 혹은
순차적으로 공존할 수 있으므로, 상호 배타적 domain hard routing과도 맞지 않는다.
이는 [hidden-domain hard routing 연구](https://arxiv.org/abs/2608.00493)의 설정과
이번 문제의 중요한 차이다.

## 4. 추가 seed와 ensemble ablation

Music loss를 키운 seed `20260907`도 별도로 학습했다.

| 모델 | factorial dev ADS | holdout ADS | phone ADS | YuE ADS |
|---|---:|---:|---:|---:|
| 일반 seed | **0.7695** | 0.7871 | **0.8696** | 0.8356 |
| Music 특화 seed | 0.7596 | 0.8048 | 0.8487 | 0.8537 |
| 두 출력의 60:40 ensemble | 0.7768 | **0.8155** | **0.8743** | **0.8612** |
| 60:40 weight soup | 0.7506 | 0.7532 | 0.8059 | 0.8431 |

출력 ensemble은 WPT standalone에는 좋았지만 v38+unified와 최종 결합했을 때
locked 평균 ADS가 일반 seed 단독 `0.86960`에서 `0.86754`로 낮아졌다. 20%만
결합하면 `0.87051`로 아주 작게 올랐지만 두 번째 전체 backbone pass의 시간과
hidden 위험을 정당화할 정도가 아니다. 서로 다른 초기 prompt를 직접 평균한
weight soup도 실패했다. 따라서 배포는 일반 seed 한 개만 쓴다.

## 5. 최종 고정 MoE

최종 결합은 logit 공간에서 다음 순서로 수행한다.

```text
VoiceExpert = 0.10 * UnifiedVoice + 0.90 * WPTVoice
FileExpert  = 0.20 * UnifiedFile  + 0.80 * WPTFile

FinalVoice = 0.90 * v38Voice + 0.10 * VoiceExpert
FinalFile  = 0.40 * v38File  + 0.60 * FileExpert
FinalMusic = v38Music
```

여기서 덧셈은 모두 확률이 아니라 logit에 적용한다. dev 최적치는 Voice/File
0.60/0.70 이상으로 더 공격적이었지만, 이미 강한 hidden Voice 축을 보호하고
YuE Voice 회귀를 피하기 위해 Voice 0.10, File 0.60을 잠갔다.

| 평가축 | v38 ADS | v39 ADS | 변화 | v39 File / Voice / Music EER |
|---|---:|---:|---:|---:|
| factorial dev | 0.74034 | **0.79392** | +0.05358 | 0.1847 / 0.2514 / 0.2114 |
| factorial holdout | 0.76821 | **0.81636** | +0.04816 | 0.1753 / 0.2057 / 0.1829 |
| phone factorial | 0.87021 | **0.89843** | +0.02821 | 0.0941 / 0.1600 / 0.0750 |
| YuE cross-component | 0.86649 | **0.89401** | +0.02753 | 0.1012 / 0.0833 / 0.1290 |

네 aggregate 축이 모두 개선됐고, factorial dev의 codec/channel subgroup별 ADS도
최소 변화가 0으로 음수 회귀가 없었다. Music은 v38과 동일하므로 v38이 확보한
Music 이득을 잃지 않는다.

## 6. 제출 구현 및 검증

- 원본 오디오만 사용하며 separator 결과를 입력으로 쓰지 않는다.
- MP3/WAV/FLAC 등은 기존 `librosa`/ffmpeg loader를 그대로 사용한다.
- WPT는 파일당 균등 view 3개, 파일 batch 8로 실행한다.
- B200에서 batch 8 WPT peak는 allocated 3.26GiB, reserved 3.30GiB였다.
- clean/전화/동시 혼합 3파일 전체 CUDA smoke가 성공했다.
- smoke에서 v38 대비 Music과 두 presence 열은 정확히 동일하고 Voice/File만
  변경됐다.
- 관련 단위 및 회귀 테스트 14개가 통과했다.
- 평가 서버 기본 패키지를 재설치하지 않으며, 기존 v38과 동일하게
  `onnxruntime-gpu==1.23.2`만 설치한다.

최종 후보는 `wpt_fixed_moe_v39.zip`이다.

- 압축 크기: 8,286,763,669 bytes
- 압축 해제 크기: 9,189,597,377 bytes
- ZIP entry: 134개, 중복 0개, bytecode/cache 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 최대 단일 member: 2,387,980,808 bytes
- 전체 CRC: 오류 없음
- SHA-256: `6313eee3d73c366e83e8c378e87a53445ea7f138c7561642f992c4bd71e523db`

## 7. 해석

이번 실험에서 router가 원리적으로 쓸모없다는 결론은 아니다. 서로 다른 전문가의
신뢰도를 알려주는 충분히 큰 OOD router 학습셋이 생기면 attention gate를 다시
검토할 수 있다. 현재 데이터에서는 domain 분류와 expert correctness가 같지 않았고,
작은 dev 이득이 phone 회귀로 뒤집혔다. 그래서 이번 제출에는 [attention MoE
연구](https://arxiv.org/abs/2509.17585)가 강조하는 표현 다양성은 활용하되,
gate 자체는 학습하지 않은 task별 고정 soft MoE가 더 일반적이고 재현 가능한
선택이다.
