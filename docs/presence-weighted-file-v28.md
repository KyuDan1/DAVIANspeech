# Presence-weighted File consistency v28

2026-09-03 기준. v28은 `routed_long_horizon_v27.zip` 뒤에서
`FILE_FAKE_PROB`만 바꾸는 단일변수 후보다. Voice/Music fake와 두 Presence 출력은
그대로 유지하므로 ADS의 File 항만 변하고 CPS에는 영향이 없다.

## 1. 왜 File을 고치는가

실제 리더보드의 `twin_v1`, Voice constant probe, Music constant probe를 역산하면
v18 이전 anchor의 EER은 대략 다음과 같다.

| 항목 | 실제 EER |
|---|---:|
| File | 0.2741 |
| Voice | 0.2156 |
| Music | 0.3714 |

v27은 Music을 직접 개선하지만 File head는 각 성분의 최종 개선을 충분히 전달받지
못한다. 실제 대회 라벨은 “존재하는 Voice 또는 Music 중 하나라도 fake면 File fake”인
OR 관계이므로, 마지막에 이 논리를 soft하게 복구할 수 있다.

## 2. hard gate 대신 presence-weighted soft OR

단순 `max(VoiceFake, MusicFake)`는 존재하지 않는 성분의 잘못된 fake score까지 File로
전파할 수 있다. 반대로 Presence 0.5 hard gate는 CPS의 작은 오차가 출력을 갑자기
바꾼다. v28은 각 성분의 fake logit에 presence logit의 절반을 더한다.

```text
voice_evidence = logit(VOICE_FAKE_PROB) + 0.5 * logit(VOICE_PRESENT_PROB)
music_evidence = logit(MUSIC_FAKE_PROB) + 0.5 * logit(MUSIC_PRESENT_PROB)
component_or   = sigmoid(max(voice_evidence, music_evidence))

new_File = sigmoid(0.5 * logit(old_File) + 0.5 * logit(component_or))
```

각 파일의 다섯 확률만 사용하므로 평가 파일 간 정보 공유가 없고, 새 모델 pass나
source separation도 없다.

## 3. 선택과 독립 확인

`file_weight`는 Factorial dev에서만 선택했다. `0.5`와 `0.7`이 File EER
`0.2247`로 동률이어서 더 보수적인 `0.5`를 고정했다. 이후 holdout, Phone, YuE는
선택에 사용하지 않고 한 번 확인했다.

| 평가군 | v27 File EER | v28 File EER | v27 ADS | v28 ADS |
|---|---:|---:|---:|---:|
| Factorial dev | 0.2782 | **0.2247** | 0.73862 | **0.76535** |
| Factorial holdout | 0.2553 | **0.2382** | 0.76094 | **0.76948** |
| Phone factorial | 0.2017 | **0.1759** | 0.81314 | **0.82607** |
| YuE | 0.1277 | **0.1012** | 0.86734 | **0.88059** |
| Factorial+Phone 통합 순위 | 0.2162 | **0.1906** | 0.79346 | **0.80627** |

세부적으로 Factorial concurrent File EER은 `0.4000 → 0.3600`, Phone mixed는
`0.2700 → 0.2400`, YuE concurrent는 `0.2500 → 0.1458`로 개선됐다. Phone
Opus-NB는 `0.4400 → 0.4234`로 여전히 가장 어려운 조건이며 다음 병목으로 남는다.

개발셋에서 raw component max는 더 높은 최고점을 냈지만, absent component 오류를
그대로 전달하므로 선택하지 않았다. Presence product와 hard gate도 비교했지만
dev/holdout 사이의 일관성이 soft logit 방식보다 낮았다.

## 4. 구현과 검증

- 추론: `src/presence_weighted_file_fusion.py`
- 재현 평가: `scripts/evaluate_presence_weighted_file_fusion.py`
- builder: `scripts/build_presence_weighted_file_v28_submission.py`
- v27과 비교해 `script.py`와 새 3KB 추론 모듈만 다름
- 3파일 GPU end-to-end smoke 성공
- 네 개의 비-File 출력은 v27과 bit-for-bit 동일
- 모든 출력은 finite이며 `[0, 1]` 범위
- 전체 테스트 `72 passed`
- ZIP `7,053,718,220` bytes, 압축 해제 `7,909,206,795` bytes
- ZIP member `111`, 중복 `0`, CRC 오류 `0`, 최대 member `2,387,980,808` bytes
- SHA-256: `38f5279cc787a8a74c0247768e5d6f71299cfd78a3a82cda37755cb086e5a8e7`

v27의 실제 단일변수 점수를 먼저 확인하는 것이 가장 해석력이 높다. 제출 quota가
허용하면 v27 뒤에 v28을 제출해 Music route 효과와 File consistency 효과를 분리한다.
