# v10 제출 방법론 및 연구 결과 보고서

## 1. 제출 요약

이번 제출은 한 모델의 점수에 의존하지 않고, 서로 다른 생성 흔적을 보는 전문가를
파일 단위로 결합하는 Mixture-of-Experts(MoE) 방식이다. 핵심 변경은 다음과 같다.

1. 음원 분리를 최종 판단의 전제로 사용하지 않고 원본 오디오 전문가를 주 경로로
   사용한다.
2. 음성은 anti-deepfake XLS-R의 공개 head와 EchoFake 적응 head를 결합한다.
3. 음악은 XLS-R, EAT, SPEAR의 SSL 표현과 고주파 Fourier fakeprint를 결합한다.
4. SPEAR로 혼합 오디오를 직접 판별하고, 혼합일 때만 혼합 전용 head를 제한적으로
   사용한다.
5. 각 평가 파일은 완전히 독립적으로 처리하며 평가 데이터 간 보정이나 적응을 하지
   않는다.

실제 리더보드 기준 출발점은 Score `0.7356957778`, ADS `0.707531746`, CPS
`0.9891720635`였다. CPS가 유지된다고 가정하면 Score 0.8에 필요한 ADS는
`0.778980882`이다. v10의 가장 어려운 외부 혼합 진단셋 ADS는 `0.86200`이며,
같은 CPS를 대입한 예상 Score는 `0.8747`이다. 이 값은 로컬 추정이며 최종 성능은
실제 비공개 평가 결과로만 확정한다.

## 2. 평가 데이터 구축

공개 데이터 한 종류에 맞춘 성능을 피하기 위해 서로 다른 역할의 평가셋을 만들었다.

### competition_v2: 대회 형식과 통신 변형 재현

- 총 1,200개 파일
- 원본 380개: 음성 200개, 음악 100개, 혼합 80개
- FLAC, 64kbps MP3, 전화 대역 WAV, 15dB 잡음 OGG 변형
- 동일 원본에서 파생된 모든 파일을 같은 split에 묶어 원본 누수를 차단
- calibration, validation, holdout으로 분리

### competition_v3: 독립 시드 음악 중심 재검증

- 총 1,200개 파일
- MusicCaps/FakeMusicCaps 의미쌍 200개 중 독립적인 100쌍 사용
- 음성 120개와 혼합 80개 추가
- clean, MP3, telephone 변형 적용
- v2와 다른 시드 및 원본 구성을 사용

### external_mixed_v1: 실제 병목을 겨냥한 혼합 평가

- 총 400개 파일
- real/fake 음성 × real/fake 음악의 네 조합을 균형화
- 동시 혼합 200개, 순차 혼합 200개
- 구성 성분의 원본 ID가 학습셋과 겹치지 않도록 분리
- File, Voice, Music EER을 모두 계산

일반 holdout 외에도 fake 생성기 하나를 통째로 제외하는
leave-one-generator-out(LOO) 평가를 사용했다. 이 검증을 통과하지 못한 모델은
일반 holdout 점수가 높더라도 비중을 낮추거나 제외했다.

## 3. 최종 추론 방법

### 3.1 구성 요소 존재 여부: PANNs

PANNs Cnn14로 `VOICE_PRESENT_PROB`와 `MUSIC_PRESENT_PROB`를 예측한다. 기존 실제
CPS가 `0.98917`로 이미 높기 때문에 presence 경로는 크게 변경하지 않았다.

### 3.2 음성 fake MoE

- `nii-yamagishilab/xls-r-2b-anti-deepfake` 공개 speech fake head
- 동일 XLS-R 원본 embedding에 학습한 EchoFake voice head
- HTDemucs voice stem의 공개 head와 EchoFake 적응 head

원본과 voice stem을 모두 사용하되 stem을 유일한 증거로 사용하지 않는다. 분리기는
생성 artifact를 지울 수 있지만, 혼합에서 음성을 드러내는 보완 효과도 있기 때문이다.
v9에서 stem embedding에 이미 학습된 EchoFake head를 재사용해 추가 encoder pass
없이 외부 혼합 Voice EER을 `0.260 → 0.185`로 낮췄다.

### 3.3 음악 fake MoE

원본 오디오에 다음 전문가를 적용한다.

| 전문가 | 역할 | 최종 기본 가중치 |
| --- | --- | ---: |
| EAT/FakeMusicCaps head | 일반 오디오 SSL 기반 음악 생성 탐지 | 0.225 |
| XLS-R/FakeMusicCaps head | anti-deepfake 표현의 음악 적응 | 0.090 |
| EAT/Echoes head | 새로운 음악 생성기 보완 | 0.225 |
| XLS-R/Echoes head | 새로운 음악 생성기 보완 | 0.360 |
| SPEAR music head | speech+audio SSL의 상보적 표현 | 0.100 |

이 합성 점수에 Fourier 전문가를 최종 `10%` 결합한다.

```text
MUSIC_FAKE = 0.90 × neural_music_MoE + 0.10 × Fourier_fakeprint
```

Fourier 전문가는 16kHz 원본 오디오의 5--7.99kHz 평균 스펙트럼에서 local lower
hull을 제거한 고주파 잔차를 사용한다. 분리하지 않은 원본을 보기 때문에 분리 과정의
왜곡과 독립적이고, 신경망 SSL 전문가와도 다른 종류의 증거를 제공한다.

### 3.4 SPEAR 혼합 라우터

`marcoyang/spear-xlarge-speech-audio-v2`의 원본 오디오 embedding으로 음성과
음악이 함께 존재하는지를 판별한다. 독립 원본 검증의 mixture EER은 `0.0050`,
competition_v2/v3 교차 도메인 EER은 `0.0082/0.0042`였다.

라우터 점수가 `0.8` 이상인 경우에만 SPEAR 혼합 전용 voice/music head를 각각
`20%` 결합한다. 생성기 LOO에서 혼합 전용 fake head 자체는 거의 chance였기
때문에, v7의 공격적인 60% 가중치를 폐기하고 보수적인 20%만 유지했다.

혼합으로 판정된 파일은 대회 정의인 “한 성분이라도 fake이면 파일 fake”를 직접
반영한다.

```text
if mixture_probability >= 0.8:
    VOICE_FAKE = 0.80 × voice_MoE + 0.20 × SPEAR_mixed_voice
    MUSIC_FAKE = 0.80 × music_MoE + 0.20 × SPEAR_mixed_music
    FILE_FAKE  = max(VOICE_FAKE, MUSIC_FAKE)
```

단일 성분 파일은 PANNs presence gate `0.7`을 사용해 존재하는 성분만 File score에
반영한다.

## 4. 결과 중심 실험 요약

### 4.1 실제 제출과 주요 버전 변화

| 단계 | 핵심 변경 | 외부 혼합 ADS | 실제 리더보드 ADS |
| --- | --- | ---: | ---: |
| 기존 제출 | speech detector 중심 | - | 0.70753 |
| v8 | SPEAR 혼합 라우터 + 보수적 혼합 head | 0.80950 | 미제출 |
| v9 | voice-stem EchoFake 적응 head 재사용 | 0.84533 | 미제출 |
| v10 | Fourier 음악 전문가 10% 추가 | **0.86200** | 제출 후 확인 필요 |

### 4.2 v10의 외부 혼합 상세 결과

| 조건 | File EER | Voice EER | Music EER | ADS |
| --- | ---: | ---: | ---: | ---: |
| 전체 400개 | 0.13000 | 0.185 | 0.120 | **0.86200** |
| 순차 혼합 200개 | 0.08333 | 0.160 | 0.090 | **0.89933** |
| 동시 혼합 200개 | 0.10333 | 0.210 | 0.120 | **0.87033** |

v9 대비 File EER은 `0.14833 → 0.13000`, Music EER은 `0.145 → 0.120`으로
개선됐고 Voice EER은 `0.185`로 유지됐다.

### 4.3 세 진단 도메인 비교

| 평가셋 | v9 ADS | v10 ADS | 변화 |
| --- | ---: | ---: | ---: |
| competition_v2 재구성 | 0.94227 | 0.93953 | -0.00274 |
| competition_v3 재구성 | 0.94060 | 0.93976 | -0.00084 |
| external_mixed_v1 | 0.84533 | 0.86200 | **+0.01667** |

Fourier 비중을 더 높이면 v2/v3의 FakeMusicCaps 계열 성능이 악화됐다. 따라서
한 평가셋 최고점보다 세 도메인의 최악 성능을 높이는 maximin 기준으로 10%를
선택했다.

### 4.4 생성기 일반화 결과

- Echoes 12개 음악 생성기 Fourier LOO 평균 Music EER: `0.0356`
- Echoes Fourier LOO 최악 Music EER: `0.1188`
- 외부 혼합 Fourier 단독 Music EER: `0.105`
- SPEAR 혼합 voice/music head LOO 평균 EER: `0.457/0.498`

마지막 결과 때문에 SPEAR의 강한 mixture representation과 약한 generator
일반화를 구분했다. SPEAR는 라우팅에는 적극 사용하고 fake head에는 낮은 가중치만
부여했다.

## 5. 기각하거나 축소한 방법

### music stem 중심 탐지

HTDemucs music stem의 XLS-R Music EER은 `0.325`, EAT 최적 결합은 `0.25`였고
원본 music MoE의 `0.21`보다 나빴다. 분리로 artifact가 손실될 수 있다는 우려가
실험으로 확인되어 music stem detector는 제거했다.

### SONICS

일부 내부 split에서는 개선됐지만 독립 도메인을 포함한 maximin 탐색에서 최적
가중치가 0이었다. 최종 제출에서 제외했다.

### Orphea

Echoes에서는 score 방향을 뒤집었을 때 EER `0.055`였지만 FakeMusicCaps v2/v3는
각각 `0.744/0.753`으로 도메인 방향 자체가 뒤집혔다. 일반화 불가능으로 판단해
제외했다.

### 학습형 file-level 결합기

calibration에서는 좋아졌지만 validation에서 악화됐다. 파일 score는 별도
classifier 대신 대회 규칙과 동일한 component score의 논리 OR를 유지했다.

## 6. 제출 패키지 검증

- 산출물: `submit_moe_v10.zip`
- 압축 크기: 약 `6.32 GiB` (제한 10GB 이하)
- 압축 해제 크기: 약 `6.97 GiB` (제한 32GB 이하)
- 최상위 구조: `model/`, `src/`, `script.py`, `requirements.txt`
- 오프라인 환경 변수 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 적용
- 패키지 자체 모델 경로만 사용
- 실제 FLAC 혼합 샘플 1개 end-to-end 추론 성공
- 스모크 실행시간: `81.1초` (모델 초기화 포함)
- 단위/회귀 테스트: `11 passed`

평가 서버에서는 모델을 한 번 초기화한 뒤 1,200개를 연속 처리한다. 기존보다 무거운
구조도 시간 제한을 통과한 이력이 있고, 현재 구조는 XLS-R embedding을 여러 head가
공유해 추가 2B encoder pass를 만들지 않는다.

## 7. 출처와 라이선스 메모

- XLS-R anti-deepfake 모델 및 논문:
  <https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake>,
  <https://arxiv.org/abs/2506.21090>
- SPEAR XLarge speech-audio-v2:
  <https://huggingface.co/marcoyang/spear-xlarge-speech-audio-v2>
- Fourier fakeprint 논문 및 공식 구현:
  <https://arxiv.org/abs/2506.19108>,
  <https://github.com/deezer/ismir25-ai-music-detector>
- Fourier 공식 구현 라이선스: CC BY-NC 4.0

2차 평가 보고서에는 사용한 모든 공개 모델·데이터의 버전, URL, 라이선스와 실제
사용 범위를 다시 명시해야 한다.

## 8. 남은 확인 사항

1. `submit_moe_v10.zip`을 실제 대회 서버에 제출한다.
2. 실제 Score, ADS, CPS를 각각 기록한다.
3. Score가 0.8 미만이면 공개되지 않는 세 EER을 직접 복원할 수 없으므로, ADS 변화와
   로컬 failure slice를 함께 보고 음성/음악/혼합 중 다음 병목을 판단한다.
4. 목표 완료 조건은 로컬 예상치가 아니라 실제 제출 Score `> 0.8`이다.
