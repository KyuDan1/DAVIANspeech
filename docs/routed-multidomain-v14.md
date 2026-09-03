# v14: 유형·전화채널 routing을 포함한 다중 도메인 파이프라인

## 최종 구조

v14는 모든 모델을 투표에 넣는 방식이 아니라, 검증된 역할만 맡기는 sparse MoE다.

| 입력 경로 | 유형 판단 | Voice fake | Music fake | File fake |
|---|---|---|---|---|
| speech-only | SPEAR single + PANNs | 공개 NII XLS-R | 평가에서 제외 | XLS-R |
| music-only | SPEAR single + PANNs | 평가에서 제외 | Fourier | Music fake |
| concurrent mixed | SPEAR mixed | XLS-R mixed head | Fourier temporal head | max(Voice, Music) |
| cascaded mixed | SPEAR mixed | XLS-R mixed head | 4초 multiple-instance Fourier | max(Voice, Music) |
| telephone mixed/music | 4.2 kHz 이상 에너지 | XLS-R phone blend | SONICS 80% + EAT 20% | 유형별 OR |

source separation과 stem은 어느 경로에서도 사용하지 않는다. SPEAR의 fake head도 사용하지 않고 single/mixed routing에만 사용한다.

## 파일 단위 독립 routing

다른 평가 파일의 통계는 사용하지 않는다. 한 파일에서 다음 순서로 결정한다.

1. `high_band_energy_ratio < 3e-6`이면 전화대역으로 분류한다.
2. SPEAR mixture score가 `0.8` 이상이면 mixed로 분류한다.
3. single이면서 PANNs voice `>=0.4`, music `<=0.075`이면 speech-only다.
4. 나머지 single은 music-only다. 보컬곡은 이 경로로 들어간다.

이 규칙은 개발 speech의 98.9%를 공개 XLS-R 단독 경로로 보냈다. 정상 mixed 6조건 ADS는 v13보다 하나도 낮아지지 않았고, source-disjoint music alignment File EER은 0.02에서 0으로 감소했다.

## 정상 채널 결과

| 데이터 | 조건 | ADS |
|---|---|---:|
| external mixed | cascaded | 0.8593 |
| external mixed | concurrent | 0.7860 |
| source-disjoint mixed | cascaded | 0.9120 |
| source-disjoint mixed | concurrent | 0.7840 |
| equal-duration mixed | cascaded | 0.8880 |
| equal-duration mixed | concurrent | 0.8647 |

- source-disjoint speech-only: File EER 0.000, Voice EER 0.000
- source-disjoint music: Music EER 0.020
- music prospective: Music EER 0.040
- 사용자 Suno 보컬곡: 13/13 fake
- WAV/FLAC/MP3/OGG Suno: 각 13/13 fake

## 전화채널

### 채널 검출

8 kHz를 거쳐 16 kHz로 복원한 전화 오디오 400개 중 97.8%를 검출했다. 임계값 `3e-6`에서 정상 source-disjoint music과 speech 오탐은 0%, 정상 external mixed 오탐은 0.2%였다.

### 음악 전문가 선택

| 후보 | 전화 source-disjoint Music EER |
|---|---:|
| 기존 5–8 kHz Fourier | 0.500 |
| EAT-Echoes | 0.420 |
| SONICS | 0.325 |
| SONICS 80% + EAT 20% | **0.295** |
| 저대역 Fourier | 0.215 |

저대역 Fourier는 순수 전화 음악 ranking은 좋았지만 혼합 selection에 추가했을 때 held-out equal-duration sequential ADS를 0.833에서 0.793으로 낮춰 최종에서 제외했다. 순수 음악 숫자 하나만 보고 채택하지 않고 모든 유형의 최악 ADS를 우선한 결정이다.

전화 전용 최종 결과:

| 데이터 | 조건 | 기존 v13 ADS | v14 ADS |
|---|---|---:|---:|
| external mixed | cascaded | 0.673 | 0.724 |
| external mixed | concurrent | 0.569 | 0.697 |
| source-disjoint mixed | cascaded | 0.780 | 0.827 |
| source-disjoint mixed | concurrent | 0.667 | 0.699 |
| equal-duration holdout | cascaded | 0.763 | 0.833 |
| equal-duration holdout | concurrent | 0.701 | 0.738 |

- 전화 speech-only: File/Voice EER 0.000/0.000
- 전화 music prospective Music EER: 0.300

전화 변환 Suno는 1/13만 0.5 이상이다. bias를 올리면 전부 fake로 만들 수 있지만 전화 mixed 최악 ADS가 하락하므로 채택하지 않았다. 원본 및 일반 코덱 Suno는 모두 13/13 fake를 유지한다.

## 기각한 방법

- source separation: detector artifact를 바꾸고 기존 ablation에서 악화
- SPEAR fake head: generator holdout에서 chance 수준
- 전화 증강 EAT 재학습: prospective EER 0.44, 전화 Suno 0/13
- 저대역 Fourier 단독: 확률이 0.51 부근에 몰리고 mixed holdout 악화
- 전화 Suno 전용 bias: Suno recall은 올리지만 mixed File EER 악화
- PANNs 단독 hard routing: mixed의 2–5%를 single로 오판

## 누수 통제

- `source_disjoint_mixed_locked_v1`은 v14 학습·선택·평가에서 사용하지 않았다.
- 전화 speech stress는 locked에 쓰인 24개 voice source를 제외하고 만들었다.
- Suno와 prospective split은 hyperparameter 선택에 사용하지 않았다.
- 모든 학습 스크립트는 `src/data_guard.py` 검사를 먼저 통과한다.

## 실행비용

400개 정상 mixed는 모델 초기화 후 약 44초, 400개 전화 mixed는 약 47초였다. 400개 12초 music은 약 80초였다. 1,200개 평가 제한 60분보다 충분히 작다. 전화 전문가 EAT/SONICS는 전화 파일이 처음 나타날 때만 lazy-load한다.

## 주요 파일

- `src/simple_pipeline.py`
- `scripts/experiment_type_routing.py`
- `scripts/build_codec_stress_eval.py`
- `scripts/build_simple_submission.py`
- `tests/test_simple_routing.py`

## 제출 산출물

- `submit_routed_v14_fixed.zip`
- 압축 크기: 6.31 GiB
- 압축 해제 크기: 6.96 GiB
- `requirements.txt`: 0 bytes (평가 서버 기본 torch/torchaudio 보존)
- 정상 mixed와 전화 mixed를 패키지 내부 모델만으로 end-to-end 재실행
- ZIP 무결성 검사 통과

최초 ZIP은 fp16 XLS-R checkpoint 하나가 4 GiB를 31 MB 정도 넘었다. ZIP 자체와
압축 해제 크기는 정상이어도 DACON 업로드 검사기가 이 ZIP64 단일 엔트리를 손상 또는
32 GB 초과로 판정했다. 최종본은 모델 값은 바꾸지 않고 XLS-R를 각각 2 GiB 이하인
safetensors 세 개로 나눴다. 815개 tensor가 최초 checkpoint와 bit-exact함을
확인했고, shard loader로 실제 GPU 추론한 뒤 DACON API 제출 성공 응답을 확인했다.

- 최대 ZIP 엔트리: 2,387,980,808 bytes (SPEAR)
- SHA-256: `d3749c4e417bd9c8fee87ca451bbee574dec8bba2b0a8ca6dc2a023fca227209`
