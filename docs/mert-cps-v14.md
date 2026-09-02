# MERT + CPS v14 실험 결과

## 결론

`mert_cps_v14`는 실제 최고 `lme_spear_v1`을 기준으로 세 가지 원칙을 지킨
보수적 ensemble 후보이다.

1. 검증된 LME + SPEAR `weight=0.10` 경로는 그대로 둔다.
2. CPS는 fake 점수의 gate로 쓰지 않고 별도 출력으로만 개선한다.
3. 원본 mixture를 보는 SOFIA/MERT 음악 expert를 File 0.10, Music 0.05의 낮은
   비중으로만 결합한다. Voice fake는 바꾸지 않는다.

MERT expert에도 source separation을 사용하지 않는다.

## 실제 점수에서 확인된 병목

실제 anchor와 component 상수 probe를 역산하면 대략 File EER 0.274,
Voice EER 0.216, Music EER 0.371이다. ADS 손실 기여도는 각각 약 0.137,
0.043, 0.111이므로 File과 Music이 주된 병목이다. SPEAR 비중을 0.10에서
0.30으로 높인 실제 제출은 ADS가 `0.717286 -> 0.715571`로 하락했기 때문에,
새 expert도 큰 비중을 주지 않았다.

## SOFIA/MERT expert

2026년 SOFIA의 공개 G1-MERT checkpoint와 MERT-v1-95M encoder를 사용했다.
MERT의 모든 hidden layer를 시간·layer 방향으로 평균한 뒤 공개 projection과
classifier를 적용한다.

- SOFIA: <https://github.com/homura23/SOFIA>
- 논문: <https://arxiv.org/abs/2606.16612>
- MERT-v1-95M: <https://huggingface.co/m-a-p/MERT-v1-95M>
- MERT license: CC-BY-NC-4.0

공개 구현은 `torchaudio`를 import하지만 평가 서버에서 `libtorchaudio.so` 오류가
이미 발생한 적이 있다. 제출 구현은 같은 Hann-window sinc kernel을 순수 PyTorch로
재현하고 원본 sample rate -> 44.1 kHz -> 24 kHz의 공개 전처리 순서를 유지한다.
dev 400개에서 공개 경로와 제출 경로의 score Spearman 상관은 0.999975,
평균 절대 차이는 0.000816이었다.

40초 강제 pad와 원본 길이 추론도 분리해 평가했다. 이 대회에서는 pad 없는
원본 길이 추론이 더 좋아 해당 경로를 고정했다.

## 독립 평가 결과

아래 값은 가중치 선택 후 바꾸지 않은 File 0.10 / Music 0.05 설정이다.

| 평가군 | LME+SPEAR ADS | + MERT ADS | 변화 |
|---|---:|---:|---:|
| factorial dev | 0.63390 | 0.64668 | +0.01278 |
| factorial holdout | 0.67787 | 0.69964 | +0.02177 |
| phone factorial | 0.68936 | 0.69511 | +0.00575 |
| YuE cross-component | 0.74091 | 0.75687 | +0.01596 |

개발·source holdout·전화 변형·unseen YuE에서 방향이 모두 같았다. 특히 MERT
단독을 쓰는 것이 아니라 기존 expert와 오류가 다른 부분만 낮은 비중으로 사용한다.

사용자가 추가한 보컬 포함 Suno 13곡에서는 MERT 단독 확률 중앙값이 0.993이고
12/13이 0.5 이상이었다. 전체 제출 파이프라인에서는 File/Music/Voice fake가
모두 13/13에서 0.5 이상이었다.

## CPS와 error propagation

CPS-only v13과 동일한 presence 경로를 사용한다.

| 지표 | 기존 경로 | 새 presence |
|---|---:|---:|
| Voice Presence AUC | 0.992400 | 0.999043 |
| Music Presence AUC | 0.984365 | 0.998730 |
| CPS | 0.988383 | 0.998887 |

presence 결과는 File/Voice/Music fake 확률을 threshold하거나 gate하지 않는다.
따라서 presence 오분류가 ADS로 전파되지 않는다. 실제 CPS가 1등 수준 0.99707에
도달하더라도 총점 이득은 약 0.00079이므로, CPS 개선과 별개로 ADS 개선이 계속
필요하다.

## 채택하지 않은 학습 실험

12,800개 학습 bank에서 같은 음악에 다른 voice authenticity를 붙인 4,674쌍,
같은 음성에 다른 music authenticity를 붙인 5,454쌍, 원본-전화 변형 4,800쌍을
만들었다. component invariance, bank-balanced sampling, joint posterior 분리 강도를
비교했다.

| 모델 | factorial holdout ADS | phone ADS | YuE ADS |
|---|---:|---:|---:|
| 기존 dual-domain 3-seed | 0.6977 | 0.7155 | 0.7991 |
| paired invariance | 0.6647 | 0.7090 | 0.8005 |
| decoupled, no pair loss | 0.6737 | 0.7144 | 0.7743 |
| decoupled + pair loss | 0.6829 | 0.7121 | 0.7641 |

훈련 데이터에서 성분 간 간섭은 줄일 수 있었지만 source-disjoint holdout 성능이
기존 head를 넘지 못했다. 따라서 이 모델들은 제출에 포함하지 않았다. 이는
현재 synthetic bank 규모에서 더 복잡한 학습 head보다 공개 대규모 음악 표현의
저비중 결합이 일반성이 높다는 증거로 해석한다.

MusicDET의 real-only normalizing flow도 공개 구조로 재학습했지만 혼합음에서는
factorial dev/holdout Music EER가 각각 0.491/0.440, phone 0.535로 일반화되지
않았다. EAT 통계의 Gaussian residual·radius·kNN real-only density 역시 평균
Music EER 0.494~0.577이었다. 음악-only를 가정한 likelihood detector를 mixture에
그대로 적용하는 방식은 채택하지 않았다.

## 실행 검증

13개 60초 Suno 파일로 제출 `script.py` 전체를 실행해 다음을 확인했다.

- 외부 다운로드 없이 PANNs, Demucs, XLS-R, EAT, SPEAR, MERT 모두 로드 성공
- `onnxruntime`, custom MERT 코드, 순수 PyTorch resampler 실행 성공
- `output/submission.csv` 13행/6열 생성
- NaN 0개, 모든 확률 0~1 범위
- File/Voice/Music fake 13/13이 0.5 이상

최종 archive 검증값은 다음과 같다.

- 파일: `mert_cps_v14.zip`
- ZIP 크기: 7,884,711,020 bytes
- 압축 해제 크기: 7,884,699,466 bytes
- 최대 member: 2,387,980,808 bytes
- 최상위 항목: `model/`, `script.py`, `requirements.txt`
- member 73개, 중복 0개, CRC 전체 통과
- SHA-256: `4ab6c9432a264a0b9d050bfe9bc7317c2e9ca8ba59b98202dc68fbe3fc753d68`

리더보드 분포 차이 때문에 로컬 상승을 실제 점수로 보장하지 않는다. 특히 기존
factorial bank가 과거 제출 순서를 한 번 뒤집은 이력이 있으므로, 실제 제출은
MERT와 CPS의 결합 전이를 확인하는 단일 후보로 취급한다.
