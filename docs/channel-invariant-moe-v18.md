# Channel-invariant MoE v18 연구·배포 기록

2026-09-02 기준. v18은 `temporal_mert_fakeprint_v17` 뒤에 clean↔telephone 및
voice↔music counterfactual consistency로 학습한 작은 EAT/SPEAR head를 logit 5%로
결합한다. 추가 backbone 추론 없이 v17이 만든 통계를 재사용한다.

`channel_invariant_moe_v18.zip`은 2026-09-02 DACON API 제출에 성공했다. 현재 실제
채점 결과를 기다리고 있으며, 결과 확인 전에는 새 anchor로 간주하지 않는다.

## 1. 왜 telephone router 대신 invariance인가

전화 router는 phone 1,200/1,200, 대응 clean 0/300으로 정확했지만 phone subset에만
강한 score correction을 적용하면 clean/phone 사이의 전역 순위가 어긋났다. 실제
EER은 평가 파일 전체 순위로 계산되기 때문에 router 정확도와 최종 ADS 개선은 같은
문제가 아니다.

v18의 head는 route 이후 점수를 바꾸지 않고 학습 시 다음 제약을 건다.

- 동일 원천의 clean/전화 버전은 authenticity score를 가깝게 유지
- voice만 바뀐 counterfactual pair는 Music score를 유지
- music만 바뀐 pair는 Voice score를 유지
- RR/RF/FR/FF auxiliary head는 학습에만 사용하고 component marginal에 직접 혼합하지 않음

학습에는 4,800 channel pair, 5,454 voice-invariant pair, 4,674 music-invariant pair를
사용했다. 평가용 Factorial holdout, phone 1,200, YuE, 사용자 Suno는 학습에 넣지
않았다.

## 2. 설정 선택

네 후보를 v16 뒤에 1--10%로 결합했다. 세 audit에서 가장 균형적인
`dec_n1`의 5%를 선택하고, 같은 설정을 MERT/fakeprint까지 들어간 v17 뒤에서 다시
검증했다.

| audit | v17 | v18 (`dec_n1` 5%) | 변화 |
|---|---:|---:|---:|
| Factorial holdout ADS | 0.72457 | **0.74551** | +0.02094 |
| Phone factorial ADS | 0.69982 | **0.71589** | +0.01607 |
| YuE ADS | 0.77197 | **0.82845** | +0.05648 |

세 task EER도 같은 방향이었다.

| audit | File EER | Voice EER | Music EER |
|---|---:|---:|---:|
| Factorial v17 → v18 | 0.2800 → **0.2553** | 0.2400 → **0.2229** | 0.2914 → **0.2743** |
| Phone v17 → v18 | 0.2959 → **0.2717** | 0.1725 → **0.1675** | 0.3925 → **0.3825** |
| YuE v17 → v18 | 0.2819 → **0.1861** | 0.1208 → **0.1021** | 0.2097 → **0.1935** |

10%는 phone ADS `0.73036`까지 올랐지만 Factorial이 `0.73483`으로 5%보다 낮고
music-only EER도 `0.20 → 0.24`로 악화됐다. 실제 리더보드에서 큰 local weight가
역전된 전례까지 고려해 5%를 고정했다.

## 3. 전화·혼합 세부 결과

| phone codec | v17 ADS | v18 ADS |
|---|---:|---:|
| G.711 μ-law | 0.75200 | **0.76529** |
| G.726 24 kbps | 0.78157 | **0.79943** |
| Opus-NB 8 kHz | 0.60329 | **0.60629** |
| 8 kHz resampling | 0.74700 | **0.75186** |

Opus-NB가 여전히 가장 큰 병목이지만 네 codec 모두 비악화다. 전화 audio type별로도
mixed `0.6370 → 0.6445`, music-only File/Music EER `0.32/0.33 → 0.30/0.31`,
voice-only EER `0.065 → 0.065`로 유지·개선됐다.

일반 Factorial에서 mixed ADS는 `0.66133 → 0.67800`이다. concurrent
`0.626 → 0.642`, partial overlap `0.680 → 0.684`, sequential
`0.730 → 0.751`로 모두 상승했다. 다만 clean FLAC 66개에서는 ADS가
`0.73747 → 0.72022`로 내려갔다. 이 작은 subgroup 회귀 때문에 weight를 더 키우지
않았고, 실제 점수 검증 전에는 v18을 확정 anchor로 간주하지 않는다.

## 4. 배포 구조

`channel_invariant_moe_v18`의 순서는 다음과 같다.

1. legacy HTDemucs stem + XLS-R-2B LME anchor
2. 원본 EAT/PANNs CPS, 원본 SPEAR
3. 원본 temporal MIL 5%
4. 원본 MERT 2.5%/1.25%
5. 원본 modern music fakeprint 2.5%
6. 재사용한 EAT/SPEAR 통계의 invariant head 5%

따라서 legacy separator가 약화시킬 수 있는 생성 artifact를 네 original-audio
branch가 보완하는 구조다. invariant head는 약 3.1MB이며 별도 backbone 실행이 없어
평가 시간 증가가 거의 없다.

배포 checkpoint SHA-256은
`8b689e2103716ddd23d56a9f345317dad63ffe2209a91e886aa495fea115bd4c`다.

clean mixed 1개와 Opus-NB mixed 1개의 전체 entrypoint smoke를 통과했다. 전화
router는 1/2만 선택했고 다섯 출력 확률은 모두 finite `[0, 1]`이었다. 임시 통계도
종료 후 삭제됐다.

- 전체 unit/integration test: `48 passed`
- ZIP member 107개, 중복 0개, CRC 오류 없음
- 최상위: `model/`, `script.py`, `requirements.txt`만 존재
- ZIP `7,907,816,613` bytes, 압축 해제 `7,907,795,069` bytes
- 최대 member `2,387,980,808` bytes
- SHA-256: `b3f7082a268acf08bbc1bfc36a25ced1be6232d81c7cf79be7ab9f890ae2d845`
