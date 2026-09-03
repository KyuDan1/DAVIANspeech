# Channel-invariant 3-seed ensemble v19 연구·배포 기록

2026-09-02 기준. v19는 v17의 원본 오디오 전문가 뒤에 동일 구조의
channel/component-invariant head 세 개를 logit 평균한 뒤 File/Voice/Music에 10%로
결합한다. v18의 단일 seed 5%에서 바뀐 것은 head의 seed ensemble과 그에 맞춘 결합
비중뿐이다. 추가 backbone이나 separator는 실행하지 않고 이미 계산한 EAT/SPEAR
통계를 재사용한다.

v18의 실제 공개 결과가 Total 약 `0.76163`, ADS 약 `0.73634`, CPS 약
`0.98930`으로 v16보다 상승해 이 학습 방향의 전이를 확인했다. 이에 따라 사전에
정한 승격 기준을 만족한 v19를 `inv3_v19.zip`이라는 짧은 파일명으로 DACON API에
제출했고 `Success` 응답을 받았다.

## 1. 학습 원칙

세 head는 같은 구조와 학습 데이터를 사용하고 seed만 다르다.

- 학습 샘플 12,800개, natural sampling
- clean↔telephone 대응 pair 4,800개
- voice가 같고 music만 다른 pair 5,454개
- music이 같고 voice만 다른 pair 4,674개
- component consistency `1.0`, channel consistency `0.5`
- RR/RF/FR/FF auxiliary 분류는 학습에만 사용
- Factorial holdout, phone factorial, YuE, 사용자 Suno는 학습·선택에서 제외

목표는 전화 여부를 hard route하는 것이 아니라 같은 원천의 clean/전화 score 순서를
보존하고, RF/FR에서 반대 성분의 label이 component score로 새지 않게 하는 것이다.

| seed | best epoch | selection | mean dev ADS | worst dev ADS |
|---|---:|---:|---:|---:|
| 00 | 5 | 0.76859 | 0.81568 | 0.72151 |
| 01 | 14 | **0.77130** | 0.81517 | **0.72743** |
| 02 | 16 | 0.76821 | **0.81944** | 0.71699 |

세 seed의 서로 다른 오차를 probability가 아니라 logit에서 평균한다. 단독 ensemble
audit에서는 Factorial ADS가 단일 seed `0.68343`에서 `0.69673`, YuE가
`0.78825`에서 `0.80226`으로 올랐다. Phone은 `0.68807`에서 `0.68689`로 소폭
하락했으므로 단독 head 수치만으로 채택하지 않고 전체 v17 뒤 결합 결과를 확인했다.

## 2. 전체 파이프라인 결합 결과

v17 뒤에서 0--10%를 sweep했을 때 세 고정 audit이 10%에서 모두 상승했다.

| audit | v17 | v19 10% | 변화 |
|---|---:|---:|---:|
| Factorial holdout ADS | 0.72457 | **0.74704** | +0.02247 |
| Phone factorial ADS | 0.69982 | **0.73111** | +0.03129 |
| YuE ADS | 0.77197 | **0.83600** | +0.06404 |

세 task별 EER도 동시에 감소했다.

| audit | File EER | Voice EER | Music EER |
|---|---:|---:|---:|
| Factorial v17 → v19 | 0.2800 → **0.2476** | 0.2400 → **0.2343** | 0.2914 → **0.2743** |
| Phone v17 → v19 | 0.2959 → **0.2583** | 0.1725 → **0.1550** | 0.3925 → **0.3625** |
| YuE v17 → v19 | 0.2819 → **0.1807** | 0.1208 → **0.1021** | 0.2097 → **0.1774** |

## 3. 전화·혼합 세부 검증

전화 codec 네 종류가 모두 개선됐다. 가장 어려운 Opus-NB도 양의 방향이다.

| phone codec | v17 ADS | v19 ADS |
|---|---:|---:|
| G.711 μ-law | 0.75200 | **0.76486** |
| G.726 24 kbps | 0.78157 | **0.81371** |
| Opus-NB 8 kHz | 0.60329 | **0.61714** |
| 8 kHz resampling | 0.74700 | **0.76429** |

Factorial의 핵심 혼합 조건도 모두 유지·개선됐다.

| 혼합 방식 | v17 ADS | v19 ADS |
|---|---:|---:|
| concurrent | 0.62600 | **0.66133** |
| partial overlap | 0.68000 | **0.69000** |
| sequential | 0.73000 | **0.76400** |

남은 회귀도 있다. clean FLAC의 Music EER은 `0.2583→0.2929`, MP3 64k는
`0.2929→0.3274`, telephone FLAC의 Voice EER은 `0.2374→0.2713`으로 나빠졌다.
YuE concurrent의 Voice EER도 `0.1875→0.2500`으로 올랐다. 전체 File/Music 개선이
이를 상쇄하지만, 실제 리더보드 검증 없이 10%보다 큰 비중을 사용하면 안 된다.

## 4. 재현과 배포 검증

체크포인트는 `models/channel-invariant-v2/seed_00.pt`부터 `seed_02.pt`까지다.
빌더는 다음처럼 실행한다.

```bash
python scripts/build_channel_invariant_submission.py \
  --base temporal_mert_fakeprint_v17 \
  --checkpoint models/channel-invariant-v2/seed_00.pt \
               models/channel-invariant-v2/seed_01.pt \
               models/channel-invariant-v2/seed_02.pt \
  --invariant-weight 0.10 \
  --output channel_invariant_ensemble_v19
```

체크포인트 SHA-256:

- seed 00: `8b689e2103716ddd23d56a9f345317dad63ffe2209a91e886aa495fea115bd4c`
- seed 01: `6d21f0fab7b763ba663399e5ec393b44791704911eda79dc336eb5716dce5c4a`
- seed 02: `bfb1ecb8045e31abfafe5ff0341a9235060ca34064d0b037bc454822985d9900`

clean mixed 1개와 Opus-NB mixed 1개의 전체 entrypoint smoke를 통과했다. 모든 출력은
finite `[0, 1]`이고 전화 router는 1/2만 선택했으며 임시 통계 파일은 삭제됐다.

- ZIP member 109개, 중복 0개, CRC 오류 없음
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- ZIP `7,914,144,900` bytes, 압축 해제 `7,914,122,982` bytes
- 최대 member `2,387,980,808` bytes
- ZIP SHA-256: `0d23e12fc0aca2fc5b403072f32ac8b5a0982f9f0e3a4e276f14add36080fe58`

## 5. 제출 의사결정과 상태

v19는 로컬 기준으로 v18보다 더 강하지만 같은 학습 계열이므로, v18 실제 ADS가
v16보다 상승한 경우에만 제출한다는 기준을 먼저 고정했다. 실제로 v18 ADS가 약
`0.72406→0.73634`로 상승해 2026-09-02 API 제출을 실행했다. 제출 파일
`inv3_v19.zip`은 위에 기록한 원본 ZIP과 hard-link 관계라 SHA-256과 내용이 완전히
같고, 파일명 길이 12자로 API의 30자 제한도 만족한다. 현재 v19 채점 결과를 기다리는
중이다.
