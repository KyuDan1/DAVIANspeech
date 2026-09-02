# Temporal MIL + CPS v16 연구·배포 기록

2026-09-02 기준. 이번 후보의 핵심은 실제 최고점 `lme_spear_v1`을 버리지 않고,
**원본 혼합 오디오의 시간별 SSL 표현을 보는 작은 expert를 5%만 결합**하는 것이다.
전화 전용 보정은 전체 순위를 깨뜨릴 수 있어 최종 ADS 경로에는 추가하지 않았고,
전화 router는 검증된 CPS 보정에만 제한했다.

## 실제 제출 결과

공개 리더보드의 팀 최고 행은 제출 횟수 24에서 Total `0.75058`, ADS `0.72406`,
CPS `0.98930`으로 갱신됐다. ZIP 생성·제출 시각과 직전 제출 횟수를 함께 보면 이
행은 v16 결과로 판단된다. DACON 제출 목록에서 얻은 전체 정밀도 행은 아니므로 아래
차이는 공개 행의 표시 정밀도 기준이다.

| 항목 | `lme_spear_v1` | v16 공개 행 | 변화 |
|---|---:|---:|---:|
| Total | 0.7444743 | **0.75058** | 약 +0.00611 |
| ADS | 0.7172857 | **0.72406** | 약 +0.00677 |
| CPS | 0.9891721 | **0.98930** | 약 +0.00013 |

따라서 저비중 temporal expert의 ADS 방향은 실제 평가에서도 재현됐다. 반면 로컬
CPS `0.998887`의 큰 상승은 실제로 거의 전이되지 않았다. 이후 제출은 v16의
temporal 구조를 유지하고 ADS 전문가를 보수적으로 추가하며, CPS는 별도 분포 불일치
문제로 다시 다룬다.

## 1. 출발점과 목표

| 항목 | 실제 리더보드 |
|---|---:|
| 현재 최고 `lme_spear_v1` Total | 0.7444743 |
| 현재 최고 ADS | 0.7172857 |
| 현재 최고 CPS | 0.9891721 |
| 1위 ADS | 0.80354 |
| 1위 CPS | 0.99707 |

기존 probe로 역산한 주 병목은 File EER 약 `0.2741`, Voice EER 약 `0.2156`,
Music EER 약 `0.3714`다. Music이 가장 어렵고, RF(진짜 음성 + 가짜 음악)와
부분 중첩/순차 혼합에서 한 성분의 점수가 다른 성분에 끌려가는 문제가 중요하다.

기존 LME+SPEAR의 실제 점수는 유지할 가치가 있지만, 로컬 제출 버전 순위와 실제
순위의 Spearman 상관은 기존 mixed 전체에서 `-0.2`, 순차/동시 subset에서 `0.4`에
불과했다. 따라서 로컬 절대 최고 weight를 고르지 않고, 여러 미관측 평가군에서 같은
방향으로 움직인 expert를 낮은 비중으로만 결합했다.

## 2. 데이터: 시간 구조가 있는 누수 방지 bank

`scripts/build_temporal_mixed_train.py`로 두 개의 3,200-file bank를 만들었다.
둘 다 RR/RF/FR/FF가 균형이며 다음 시간 배치를 동일 비율로 포함한다.

- 완전 동시(`concurrent`)
- 부분 중첩(`partial_overlap`)
- 앞뒤 순차(`sequential`)
- 짧은 음성 구간(`sparse_voice`)
- 짧은 음악 구간(`sparse_music`)

각 행에는 `VOICE_START/END`, `MUSIC_START/END`가 있어 6초 EAT view와 10초
SPEAR view별 정답을 정확히 만들 수 있다. 파일 전체가 fake라는 약한 label만 쓰지
않고, 해당 crop 안에 실제로 들어온 fake 성분만 local positive로 학습한다.

누수 방지는 다음과 같이 확인했다.

- train 2,560 / internal dev 640
- 동일 speaker, 음악 group, 원본 source는 한 split에만 존재
- 평가용 component ID는 학습 bank에 미포함
- 생성 파일 3,200개, manifest 3,200행, 구간 범위와 ID가 모두 일치
- `data_guard` 통과

v1은 다양한 독립 조합을 만들고, v2는 같은 voice/music/layout/SNR을 고정한 정확한
2×2 counterfactual quartet 800개를 만든다. v2는 nuisance component를 고정한 채
Voice 또는 Music의 real/fake만 바꾸므로 cross-component shortcut을 줄이는 목적이다.

## 3. 방법: 분리 없는 원본-mixture temporal MIL

새 branch는 HTDemucs나 diffusion separator를 사용하지 않는다. 원본 오디오에서
start/middle/end crop을 뽑고 다음 frozen SSL latent를 사용한다.

- EAT: 6초 view, acoustic/event 표현
- SPEAR: 10초 view와 13개 layer, speech와 music을 함께 배운 표현
- token 통계: mean, standard deviation, temporal absolute delta,
  Teager--Kaiser texture energy

작은 head는 view별 local logit, attentive global statistics와
`logmeanexp(temperature≈5)` MIL을 함께 사용한다. Voice task는 주로 SPEAR 중간층
5--9, Music task는 얕은층 0--3을 선택했다. 이는 waveform을 물리적으로 분리하지
않고 모델 내부의 task-specific attention으로 두 성분을 다르게 읽는 방식이다.

RR/RF/FR/FF joint head도 함께 학습해 `File = Voice OR Music` 관계를 보조한다.
학습이 끝난 뒤에는 여섯 checkpoint를 사용한다.

- v1 3-seed ensemble: File과 Voice
- paired v2 3-seed ensemble: Music

기존 LME+SPEAR와는 probability 평균이 아닌 logit 공간에서 각 task 5%만 결합한다.
CPS는 `cps_v13`의 EAT/PANNs presence와 전화 Voice presence head를 그대로 쓴다.

## 4. ablation 결과

### 4.1 정확한 view supervision

동일 seed에서 crop 구간 label을 사용한 모델이 사용하지 않은 모델보다 세 독립
audit 모두 좋아졌다.

| 모델 | Factorial ADS | Phone ADS | YuE ADS |
|---|---:|---:|---:|
| view auxiliary weight 0 | 0.71738 | 0.69157 | 0.75650 |
| view auxiliary weight 0.5 | **0.73849** | **0.69418** | **0.77162** |

auxiliary weight 1.0은 phone은 `0.72339`로 올랐지만 YuE가 `0.71353`으로 떨어져
채택하지 않았다. width 64 축소 모델도 세 audit에서 일관되지 않아 제외했다.

### 4.2 v1과 paired v2

| standalone expert | Factorial | Phone | YuE | source-disjoint Music EER | multigen Music EER |
|---|---:|---:|---:|---:|---:|
| v1 3 seeds | 0.75294 | 0.70943 | 0.74867 | 0.1900 | 0.1708 |
| paired v2 3 seeds | 0.74878 | 0.70275 | **0.79258** | **0.1500** | **0.1375** |

v1은 Factorial/Phone의 File·Voice가 더 안정적이고, v2는 미관측 음악 생성기에서
훨씬 강했다. 따라서 전부 평균하지 않고 task별 hybrid를 선택했다.

사용자가 추가한 Suno vocal 음악 13개에서는 v2가 다음을 만족했다.

- File fake 13/13, 최소 확률 `0.9953`
- Music fake 13/13, 최소 확률 `0.9837`
- Voice fake 11/13

대회 정의상 이 파일들은 AI 음악 성분 하나만 fake여도 파일 전체 fake이므로,
요구했던 File/Music 판정은 전부 통과했다. 다만 Suno 13개만으로 다른 생성기
일반화를 주장하지 않고 source-disjoint와 YuE 결과를 함께 선택 기준으로 썼다.

### 4.3 실제 최고 anchor에 5% 결합

| 평가군 | LME+SPEAR anchor | hybrid 5% | 변화 |
|---|---:|---:|---:|
| 여러 dev bank union | 0.66123 | **0.70261** | +0.04138 |
| Factorial holdout | 0.67618 | **0.72366** | +0.04748 |
| Phone factorial | 0.68936 | **0.70339** | +0.01404 |
| YuE generator audit | 0.74091 | **0.80873** | +0.06783 |

Factorial의 23개 세부 contrast 중 16개 개선, 6개 동일, 1개 악화였다. 남은 유일한
회귀는 `partial_overlap / voice fake + music real`에서 Music EER가
`0.28 → 0.32`가 된 경우다. Music expert를 빼면 이 회귀는 없어지지만 실제 최대
병목인 Music EER 개선도 사라져, 낮은 5%를 유지했다.

## 5. 전화 데이터의 현재 결론

전화 평가셋 `phone_factorial_1200_v1`은 동일한 300개 clean 원천에 다음 네 채널을
각각 적용한 1,200개다.

- 8 kHz resampling
- G.711 μ-law
- G.726 24 kbps
- Opus narrowband 8 kHz

구성은 mixed 400, music-only 400, voice-only 400이며 train에는 들어가지 않는다.
전화 router는 1,200/1,200을 전화로 잡고 대응 clean 300개는 0/300만 전화로
오탐했다.

기존 전화 anchor는 Voice EER `0.1825`가 아니라 Music EER `0.4175`, 특히
Opus-NB mixed File이 문제였다. 새 temporal hybrid는 전화 ADS를
`0.68936 → 0.70339`로 올렸다. 따라서 전화에서도 speech detector 교체보다
원본-mixture Music/File evidence가 실제 병목에 더 직접적으로 작용했다.

기존 phone-only dual-domain 보정을 temporal 뒤에 강하게 더하면 phone subset은
`0.73575`까지 올랐지만 일반 Factorial 전체 ADS가 temporal-only `0.72538`에서
`0.71564`로 하락했다. EER은 모든 전화/일반 파일의 상대 순위를 함께 보기 때문에,
전화 subset 내부 calibration만 맞춘 hard route가 전체 순위를 어긋나게 만든다.
그래서 v16 ADS에는 별도 phone route를 넣지 않았다. router는 CPS의 좁은 Voice
presence 10% 보정에만 쓴다.

## 6. CPS 경로

v16은 `cps_v13.zip`을 기준 패키지로 사용한다. 로컬 1,200 union에서 이 경로는
Voice Presence AUC `0.999043`, Music Presence AUC `0.998730`, CPS
`0.998887`이었다. 같은 평가의 기존 CPS `0.988383`은 실제 `0.989172`와
`0.000789` 차이여서, 현재 로컬 지표 중 실제와 가장 잘 맞는 축이다.

CPS가 실제 목표 `0.99707`까지 도달하더라도 Total 기여는 약 `+0.00079`이므로,
주요 승부는 여전히 ADS다. Presence를 fake probability gate로 쓰지 않아 CPS 오류가
ADS로 전파되지 않게 했다.

## 7. 배포 검증

후보는 `temporal_mil_cps_v16.zip`이다.

- 기반: 실제 최고 LME pooling + SPEAR 0.10, CPS v13
- 새 ADS 변경: v1 File/Voice + v2 Music temporal expert를 logit 0.05 결합
- 큰 모델: 기존 XLS-R-2B, EAT, SPEAR를 오프라인 local model에서만 로드
- requirements: `onnxruntime-gpu==1.23.2`
- clean 1개 + 전화 1개 end-to-end smoke 성공
- 전화 router: 1/2만 선택
- 출력 5개 확률: 모두 finite, `[0, 1]` 범위
- 임시 EAT/SPEAR 통계: 실행 종료 후 삭제
- ZIP member 90개, 중복 0개, CRC 오류 없음
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- ZIP 7.00837 GiB, 압축 해제 7.00835 GiB, 최대 member 2.22398 GiB
- SHA-256: `1a4b3289d394a85c046f7cc8c314dded169eacd2327268a1b05c86faad722d97`

스모크 중 두 가지 배포 결함을 실제 제출 전에 발견해 수정했다.

1. 기준 ZIP의 구버전 `spear_detector.py`를 남겨 통계 메서드가 없던 문제
2. 검증된 `onnxruntime-gpu` requirement를 빈 파일로 덮어쓰던 문제

## 8. 해석과 다음 실험

이번 후보는 네 독립 평가군에서 모두 양의 방향이지만 실제 test 분포와 로컬 버전
순위 정렬이 약하므로 실제 점수 상승 폭은 단정할 수 없다. 특히 과거 SPEAR
`0.10 → 0.30`은 로컬에서 크게 올랐지만 실제에서는 하락했다. 따라서 먼저 5%
후보 한 개로 실제 인과를 확인해야 한다.

실제 결과 이후의 우선순위는 다음과 같다.

1. v16 ADS/CPS가 모두 오르면 이 제출을 새 anchor로 고정한다.
2. Music EER 이득을 probe 제출로 분리하고, paired v2 Music weight만 0.025/0.05
   사이에서 좁게 확인한다.
3. 전화는 Opus-NB mixed File과 부분 중첩 RF를 별도 고정 audit으로 유지하되,
   전화 subset만 좋아지는 hard route는 사용하지 않는다.
4. 남은 `partial_overlap / fake voice + real music` 회귀는 더 많은 생성기가 아니라
   nuisance-matched quartet과 local component consistency loss로 해결한다.
5. 실제 리더보드 확인 전에는 큰 weight, 새 separator, 학습형 cross-file calibration을
   추가하지 않는다.

재현에 필요한 핵심 구현은 `src/temporal_dual_domain_head.py`,
`src/temporal_dual_domain_inference.py`, `scripts/build_temporal_mixed_train.py`,
`scripts/train_temporal_dual_domain_head.py`, `scripts/evaluate_temporal_dual_domain.py`,
`scripts/sweep_anchor_expert_fusion.py`, `scripts/build_temporal_cps_submission.py`에 있다.
