# ArtifactBench music generalization v108

## 결론

v108은 v83/v101의 원본 오디오 기반 구조를 유지하면서 Music prompt expert만
다양한 공개 AI 음악으로 확장한 후보이다. 소스 분리는 사용하지 않는다. 최종 후보는
ArtifactBench 학습 소스와 기존 학습 자료로 학습한 뒤 exact telephone codec으로 낮은
학습률 재적응을 거쳤다.

- ZIP: `artifactbench_codec_music_moe_v108.zip`
- base: `best_eat_music_v83.zip`
- v101 joint File/Voice expert는 그대로 유지
- Music expert: codec-refined ArtifactBench prompt
- Music logit weight: 0.37
- Presence 두 출력은 변경하지 않음
- 공식 제출: 아직 하지 않음

## 데이터 경계

ArtifactBench v1의 embedded AI audio 4,400곡을 16 kHz PCM16 FLAC으로
물질화했다. 전체 곡을 저장하고 학습 때마다 임의 4초 crop을 선택한다.

| 역할 | 소스 | 곡 수 |
|---|---|---:|
| train | AIME 9종 + MoM 4종 | 2,600 |
| OOD holdout | SONICS 5종 + 최신 Suno/Udio 4종 | 1,800 |

학습 내부의 동일 PCM 3곡은 제외했다. train/holdout을 가로지르는 동일 해시는 없고,
repository의 protected identity guard도 통과했다. 기존 development에 SONICS 200곡이
있으므로 SONICS holdout을 완전 독립 근거로 과장하지 않는다. 최신
`suno_cdn_latest`, `suno_extra`, `udio_cdn_latest`, `udio_extra`가 더 중요한
새 버전 감사이다.

ArtifactBench의 라이선스는 CC BY-NC 4.0이다. 대회 2차 보고서에는 데이터셋과
원 논문 출처를 반드시 기록해야 한다.

## 학습 방법

모델은 frozen XLS-R/Spectra backbone 위의 base/frequency/texture prompt와 작은
AASIST backend를 사용한다. 원본 혼합 오디오만 입력하므로 separator가 새로운
artifact를 만드는 문제를 피한다. 학습 중 같은 crop에 clean 및 deterministic
telephone 변형을 만들고 두 예측에 동일 정답과 consistency loss를 적용한다.

세 설정을 같은 seed로 비교했다.

1. ArtifactBench case mass 1.0, 기본 학습률
2. ArtifactBench case mass 0.5, 기본 학습률
3. ArtifactBench case mass 0.5, 보수적 학습률

세 설정 모두 단독 broad EER는 낮지 않았지만 기존 v83과 오류 상관이 약해 soft
fusion에서는 개선됐다. full-mass 모델은 broad EER가 가장 좋았으나 codec bank가
크게 악화되어 채택하지 않았다.

그 후 full-mass checkpoint를 exact G.711/G.722/narrowband Opus/G.711→Opus
paired bank에 두 학습률로 재적응했다. development와 codec 결과만 보고 5e-5 prompt
학습률 후보를 고정했다.

## 결과

### Broad development 4,261 music-present files

| 방법 | Music EER |
|---|---:|
| v83 Music | 0.1291 |
| v101 Music | 0.1202 |
| full-mass replacement | 0.1131 |
| **v108 codec-refined replacement** | **0.1164** |

v108은 full-mass보다 pooled EER가 0.0033 높지만 채널별 회귀가 작다.

| corpus | v83 | v101 | v108 |
|---|---:|---:|---:|
| codec_mixed_dev_v4 | 0.2133 | 0.2033 | 0.2100 |
| external_mixed_v1 | 0.1400 | 0.1300 | 0.1350 |
| external_mixed_v1_telephone | 0.1550 | 0.1550 | 0.1450 |
| factorial_eval | 0.1829 | 0.2114 | 0.1714 |
| source_disjoint_mixed | 0.0800 | 0.0800 | 0.0700 |
| source_disjoint_mixed_telephone | 0.1500 | 0.1300 | 0.1400 |

leave-one-dataset-out에서는 v83 대비 9개 중 5개 corpus가 개선됐다. 이는 완전한
일관 개선이 아니므로 공식 성능을 보장하지 않는다.

### Frozen ArtifactBench OOD audit

아래 값은 모두 fake인 파일에서 score > 0.5인 비율이다. threshold는 대회 EER
threshold가 아니므로 절대 정확도가 아니라 최신 generator recall 진단으로만 본다.

| source | old seed35 | full-mass | v108 refined |
|---|---:|---:|---:|
| Suno CDN latest | 0.170 | 0.700 | 0.515 |
| Suno extra | 0.045 | 0.765 | 0.505 |
| Udio CDN latest | 0.780 | 1.000 | 0.980 |
| Udio extra | 0.825 | 0.995 | 0.980 |
| all 9 sources | 0.751 | 0.936 | **0.877** |

codec 재적응 뒤에도 old seed35 대비 새 generator recall이 유지됐다. full-mass가
OOD recall은 더 높지만 telephone/codec 회귀가 커서 제출 후보에서 제외했다.

## ArtifactNet 감사

공개 ArtifactNet v9.4는 최신 Suno holdout에서 강했지만 broad Music EER는 0.3656으로
낮았다. v101에 global logit fusion하면 최적 weight가 0.02에 불과했고 9개
leave-one-dataset-out 중 1개만 개선됐다. 따라서 v108에는 추가하지 않았다. 향후
generator-unknown router 근거가 생기기 전에는 high-score hard routing도 사용하지
않는다.

## 예상과 한계

v101의 broad File/Voice 수치와 v108 Music 수치를 합친 proxy ADS는 약 0.852이다.
이는 공식 점수 예상치가 아니다. 기존 로컬/공식 괴리를 고려한 현실적 총점 예상은
약 0.79~0.82이며, 공식 데이터에 최신 Suno/Udio 비중이 클 때 상단을 기대할 수 있다.

공식 0.85를 달성하려면 Music 개선만으로 부족할 수 있다. 공식 probe로 추정한
현재 EER는 File 0.2317, Voice 0.1756, Music 0.3171이므로, 다음 반복의 우선순위는
v108 Music 검증 후 File EER, 그 다음 Voice EER이다.

## 재현 산출물

- materializer: `scripts/prepare_artifactbench_external_v106.py`
- train manifest: `data/external/artifactbench_v1/prepared_v106/train_augmented.csv`
- holdout manifest: `data/external/artifactbench_v1/prepared_v106/holdout_fake_eval.csv`
- selected checkpoint: `reports/artifactbench_music_prompt_v106/exactcodec_refine_lr5e5_seed108/multistream_prompt.pt`
- robustness audit: `reports/artifactbench_music_prompt_v106/exactcodec_refine_lr5e5_seed108/robustness_audit/`
- package report: `artifactbench_codec_music_moe_v108.report.json`
- ZIP SHA256: `f5330d223dca915279b51f49bd58de31374b98b2ac9a39ffec77966acef474f5`
