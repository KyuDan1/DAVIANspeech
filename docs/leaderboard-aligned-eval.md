# 리더보드 정렬 평가셋 재구성 결과

작성일: 2026-08-29

## 결론

기존 로컬 평가셋 하나의 총점으로 제출 버전을 선택하면 안 된다. 기존 셋은 대부분
학습에 사용한 FakeMusicCaps, Echoes, EchoFake 계열과 가까워서 v6의 실제 하락을
반대로 평가했다. 실제 제출 네 건을 기준점으로 삼아 다시 검사한 결과, 다음 두 축을
분리해야 한다.

1. **source-disjoint 음악 단독 평가**: 새로운 생성기와 실제 음악에 대한 음악
   detector 자체의 일반화를 측정한다.
2. **source-disjoint 혼합 평가**: 새로운 음악을 음성과 동시에 섞거나 순서대로
   연결해, component detector 사이의 간섭과 file-level OR/router를 측정한다.

두 번째 축을 `CONDITION`별로 보면 실제 리더보드의 버전 변화 방향을 재현한다.
따라서 이후 후보는 단일 평균이 아니라 음악 단독, 동시 혼합, 순차 혼합의 세 조건을
모두 통과해야 한다.

## 실제 제출 기준점

| 버전 | 실제 ADS | 실제 CPS | 실제 총점 |
|---|---:|---:|---:|
| v3/reference | 0.707532 | 0.989172 | 0.735696 |
| v6 | 0.649960 | 0.989172 | 0.683881 |
| v9 | 0.675087 | 0.989172 | 0.706496 |
| v10 | 0.676302 | 0.989172 | 0.707589 |

CPS는 네 제출에서 동일하다. 현재 병목과 버전 차이는 ADS이며, 실제 변화는
`v3 > v6 < v9 < v10`이다.

## 평가 데이터 구성

### source_disjoint_music_v1

- fake 200개: SONICS `part_01`의 공식 test split에서 Suno chirp-v3.5 100개,
  Udio-120s 100개
- real 200개: FMA-small에서 200개
- 원본 song ID는 한 번만 사용한다.
- 양쪽 클래스를 모두 mono, 16 kHz, PCM16 FLAC, 12초로 동일하게 변환한다.
- crop 위치는 ID의 SHA-256으로 정해 재현 가능하다.
- 200개 alignment / 200개 prospective로 나눈다. prospective 절반은 앞으로
  weight 선택에 사용하지 않고 최종 방향 검증에만 사용한다.

생성 명령:

```bash
python scripts/build_source_disjoint_music_eval.py \
  --sonics-zip data/external/sonics_eval/fake_songs/part_01.zip \
  --sonics-metadata data/sources/sonics_metadata/fake_songs.csv \
  --fma-zip data/external/sonics_eval/fma_small.zip \
  --output-dir data/eval/source_disjoint_music_v1
```

FMA-small ZIP의 검증된 SHA-1은
`ade154f733639d52e35e32f5593efe5be76c6d70`이다.

### source_disjoint_mixed_v1

위 음악과 `asvspoof_voice_v1`의 음성을 사용해 200개를 만들었다.

- voice real/fake × music real/fake 네 조합 균등
- simultaneous 100개: -6/0/+6 dB voice-to-music 비율로 겹쳐서 혼합
- sequential 100개: 4초 음성과 4초 음악을 순서대로 연결
- 모든 결과는 16 kHz FLAC

생성 명령:

```bash
python scripts/build_eval_mixtures.py \
  --voice-dir data/eval/asvspoof_voice_v1 \
  --music-dir data/eval/source_disjoint_music_v1 \
  --output-dir data/eval/source_disjoint_mixed_v1 \
  --per-combination 25 --seed 20260829 --id-prefix sdxmix
```

이 데이터는 자연 녹음 혼합의 완전한 대용물이 아니라 detector 간 간섭을 통제해
측정하는 스트레스 테스트다. 절대 ADS를 private ADS의 추정치로 사용하지 않는다.

## 결과

### 음악 단독: 외부 일반화

| 버전 | File EER | Music EER |
|---|---:|---:|
| v3 | 0.440 | 0.450 |
| v6 | 0.365 | 0.365 |
| v9 | 0.365 | 0.365 |

v6의 새 음악 ensemble은 v3보다 분명히 좋아졌다. 그러나 EER 0.365는 여전히
매우 높다. 동시에 이 순서는 실제 제출의 v3→v6 하락과 반대이므로, 이 셋만으로
제출 버전을 고르면 안 된다. 이 결과는 v6의 실제 하락 원인이 음악 head 자체의
외부 일반화 하나가 아니라는 증거다.

alignment에서 v3/v6/v9 Music EER은 각각 0.44/0.37/0.37이고, 한 번도 선택에
쓰지 않은 prospective에서는 0.47/0.35/0.35였다. v6 개선 방향은 holdout에서도
유지됐다.

### 혼합 상호작용: 리더보드 변화 방향 재현

| 조건 | v3 ADS | v6 ADS | v9 ADS | v10 ADS |
|---|---:|---:|---:|---:|
| 전체 | 0.6710 | 0.6793 | 0.7350 | 0.7453 |
| sequential | 0.7913 | 0.7780 | 0.8260 | 0.8580 |
| simultaneous | 0.6227 | 0.6160 | 0.6520 | 0.6533 |

전체 평균은 v6의 단일-component 개선 때문에 v3보다 높아져 실제 순서를
희석한다. 반면 두 혼합 조건은 모두 `v3 > v6 < v9 < v10`으로 실제 리더보드와
같은 방향이다. 특히 v9의 mixture-aware head/router가 v6 손실을 회복한다는
해석과 일치한다.

로컬 상승 폭은 실제보다 크다. 예를 들어 v9→v10은 로컬 전체에서 +0.0103이지만
실제는 +0.00121이다. 따라서 이 평가셋은 **방향 필터**로 사용하고, 수치를 private
점수로 선형 환산하지 않는다.

## v3→v6 하락에 대한 현재 해석

v3→v6에서는 한 번에 다음이 바뀌었다.

- ArtifactNet 제거
- EchoFake voice head 추가
- Echoes 기반 XLS-R/EAT music head 추가
- SPEAR music head 추가
- ensemble weight 대폭 변경

음악 단독에서는 이 묶음이 개선됐지만 두 혼합 조건에서는 악화됐다. 가장 가능성이
큰 문제는 raw mixture에서 학습된 여러 음악 점수가 음성 성분에 반응하거나,
component별 점수의 scale이 달라진 상태에서 presence 기반 `combine`을 적용한
것이다. v9에서 raw-mixture 전용 head와 router를 넣자 회복된 것도 이 설명을
지지한다. 다만 v6 변경이 묶여 있으므로 아직 특정 head 하나가 원인이라고 단정할
수는 없다.

SAM-Audio/HTDemucs가 생성 흔적을 지울 수 있다는 우려도 타당하지만, 현재 파이프라인의
music head 대부분은 분리 stem이 아니라 raw audio를 본다. 분리 영향은 주로 voice
stem 경로에 남는다. 다음 ablation은 separator 전체 교체보다 `raw-only voice`,
`stem-only voice`, `raw+stem`을 같은 혼합셋에서 비교하는 것이 먼저다.

## 이후 실험 규칙

1. CPS는 회귀 방지용으로만 검사하고, 연구 예산은 ADS에 집중한다.
2. 새 head는 학습 source 내부 점수와 source-disjoint 점수를 함께 기록한다.
3. 후보가 음악 단독을 개선해도 simultaneous 또는 sequential 중 하나를 악화시키면
   바로 제출하지 않는다.
4. leaderboard 네 점에 고차원 weight를 fitting하지 않는다. 관측값이 너무 적어
   과적합된다.
5. alignment로 소수의 coarse choice만 하고 prospective는 최종 한 번만 확인한다.
6. 다음 구현은 v10에서 전문가별 raw score를 저장하는 diagnostic mode를 추가하고,
   v6 변경 요소를 한 번에 하나씩 되돌리는 post-hoc ablation이다. 이후
   source-conditioned gate 또는 작은 OOF stacking head를 학습한다.

## 바로 이어서 할 실험

- v10 diagnostic dump: legacy XLS-R, EchoFake voice, EAT/FakeMusicCaps,
  EAT/Echoes, XLS-R/Echoes, SPEAR, mixture heads, router score를 파일별 저장
- v6 counterfactual: ArtifactNet 복원, EchoFake voice 제거, 각 새 music head 제거,
  v3 weights 복원
- raw/stem voice ablation으로 separator 손실 측정
- generator/source group 단위 OOF로 gate 학습; 파일 간 통계는 추론에서 사용하지 않음
- 후보 선택 기준은 세 축의 worst-case ADS 개선과 실제 실행시간 60분 이내

