# Strict all-type three-stream v57 결과

작성일: 2026-09-04. 최종 학습·선택 근거는 명시된 train/development와
`codec_mixed_dev_v4`의 exact-v50 prediction만 사용했다. 비선택 role의 feature,
prediction, truth, score는 최종 strict run의 입력이 아니다.

## 결론

2026-09-05 공식 결과 추가: 제출 82697은 Total `0.7774307884`, ADS
`0.7538888889`, CPS `0.9893078836`, 실행 `54:05`를 기록했다. v50 대비
Music EER이 정확히 2%p 감소했다. 아래는 당시 개발 결과로 내렸던 판단의 기록이며,
공식 일반화가 불가능하다는 확정 결론으로 해석하면 안 된다. v9의 ADS +0.00125는
사전 +0.0025 게이트를 통과하지 못했지만 탐색 목적으로 제출한 예외였다.
최신 해석과 절차 정정은 `docs/review-after-0.777-v58.md`에 기록했다.

일반성만 보면 **v57을 exact v50에 붙이지 않는 것이 정답**이다. 4,577행 전체에서
가장 강한 v2 full은 ADS를 `0.831538 → 0.857405`로 크게 올렸지만 voice-only
layout이 `-0.05143` 퇴행했다. task를 분리하면 v2 File-only가 ADS `+0.01957`,
Voice-only가 `+0.00632`였지만 각각 Opus와 voice-only layout의 사전 비회귀 문턱을
넘지 못했다.

다만 exact v50 위의 허용된 codec development 600행에서는 **v1 Music-only**만
ADS를 `0.760 → 0.766`, Music EER을 `0.27333 → 0.25333`으로 개선했다. 따라서 이
한 후보만 residual scale 1.0의 공격적 one-shot challenger로 고정했다. broader
development에서는 ADS가 `-0.002394`이므로, 별도의 이미 고정된 one-shot gate를
이기지 못하면 제출하지 않는 후보이다. 다른 full/Voice/File/v2 후보는 폐기한다.

## 데이터 무결성 재구축

처음 component-specific column만 검사했을 때는 overlap이 0이었다. 이후
`FMA_TRACK_ID`, `ORIGINAL_AUDIO`, archive member, generic speaker까지 all-token
검사를 넓히자 다음 5개 source identity가 train과 development에 동시에 있는 것을
발견했다.

- FMA track `116870`, `121346`, `136994`, `145761`
- `The Dawn - The One They Fear`

성능과 무관하게 해당 train 6행을 제거하고 처음부터 재학습했다. 최종 split은 다음과
같다.

| 항목 | Train | Development | 교집합 |
|---|---:|---:|---:|
| rows | 18,738 | 4,577 | - |
| unique rendered-audio SHA-256 | 18,701 | 4,568 | **0** |
| expanded identity token | 43,714 | 11,091 | **0** |
| Voice/Music/Parent identity | 6,706/8,634/13,420 | 1,971/2,523/3,012 | **0/0/0** |

최종 trainer는 `--selected-split-only`로 선언된 두 role만 읽고, exclusion mask를 truth
row와 모든 cached feature stream에 똑같이 적용한다. 실제 감사 결과는
`reports/three_stream_all_type_v57_strict/split_audit.json`이다.

## 학습 방법

- exact-v47의 5개 출력은 anchor로 동결했다.
- 원본 혼합 오디오에서 미리 추출한 ordered XLS-R window, EAT temporal/spectral
  patch graph, SPEAR component-bin을 함께 읽는다. source separation을 새로 하지 않는다.
- 935,809-parameter head가 Voice, Music, direct File logit residual 세 개를 낸다.
- residual limit은 세 task 모두 `±2.0`; authenticity loss weight는 공식 ADS와 같은
  Voice/Music/File `0.2/0.3/0.5`다.
- dataset/type/RR·RF·FR·FF/layout/channel/generator-aware Group-DRO와 paired-channel
  consistency를 사용한다.
- v1은 기본 robust loss, v2는 quartet component/latent invariance와 AUC ranking을
  추가한다. 두 설정은 점수를 보기 전에 matrix에 고정했다.
- checkpoint 선택식은 `0.50 pooled ADS + 0.25 mean-domain ADS + 0.25 worst-domain ADS`다.

## 전체 task EER와 독립 residual

아래 delta는 같은 4,577행 exact-v47 identity 대비이며, 음수 EER delta가 개선이다.
Presence는 모든 후보가 bit-exact라 CPS가 모두 `0.990837`이다.

| 후보 | File EER | Voice EER | Music EER | ADS | ΔADS |
|---|---:|---:|---:|---:|---:|
| identity | 0.179828 | 0.181523 | 0.140812 | 0.831538 | - |
| v1 full | 0.150980 | 0.159363 | 0.148791 | 0.848000 | +0.016462 |
| v1 Voice-only | 0.179828 | 0.159363 | 0.140812 | 0.835970 | +0.004432 |
| **v1 Music-only challenger** | 0.179828 | 0.181523 | 0.148791 | 0.829144 | **-0.002394** |
| v1 File-only | 0.160751 | 0.181523 | 0.140812 | 0.841076 | +0.009538 |
| v2 full | **0.138489** | **0.149900** | 0.144567 | **0.857405** | **+0.025868** |
| v2 Voice-only | 0.179828 | 0.149900 | 0.140812 | 0.837862 | +0.006325 |
| v2 Music-only | 0.179828 | 0.181523 | 0.144567 | 0.830411 | -0.001126 |
| v2 File-only | 0.140684 | 0.181523 | 0.140812 | 0.851110 | +0.019572 |

v2 full의 domain-normalized ADS는 11개 중 9개에서 같거나 개선했다. 하락은
`echoes_fma_paired_v3 -0.01126`, `source_disjoint_music_v1 -0.01125`였다. 채널은
telephone8k `+0.04611`, G.711 μ-law `+0.05556`, telephone FLAC `+0.08372`로 강했지만,
voice-only layout `-0.05143` 때문에 일반 배포 후보로는 기각했다. 모든 domain,
channel, layout delta와 task EER delta는
`reports/three_stream_all_type_v57_strict/evaluation_v3/regressions.csv`에 있다.

RR/RF/FR/FF 각 cell은 label이 한 class라 cell 내부 EER 자체는 정의되지 않는다.
대신 전체 development에서 얻은 task별 EER operating threshold를 각 cell에 고정하여
error rate를 냈다. v1 Music-only의 broader-development weighted error delta는
RR `+0.00227`, RF `+0.00065`, FR `+0.00162`, FF `+0.00227`로 모두 소폭 악화했다.
완전한 표는 `evaluation_v3/cell_operating_points.csv`에 있다.

## exact v50 위의 사후 적용

v50은 기존 Voice와 WPT File을 이미 개선하므로 v47 결과만 보고 residual을 붙이면 안
된다. 허용된 `codec_mixed_dev_v4` 600행의 exact-v50 prediction에 v57 residual을
다시 적용했다. Voice-only/Music-only/File-only는 이름 그대로 한 열만 바꾸고, full만
고정된 70% direct File + 30% component noisy-OR를 사용했다.

| 후보 | File EER | Voice EER | Music EER | ADS | v50 대비 |
|---|---:|---:|---:|---:|---:|
| exact v50 | 0.23333 | 0.20667 | 0.27333 | 0.76000 | - |
| v1 full | 0.23333 | 0.27667 | 0.25333 | 0.75200 | -0.00800 |
| v1 Voice-only | 0.23333 | 0.27667 | 0.27333 | 0.74600 | -0.01400 |
| **v1 Music-only** | 0.23333 | 0.20667 | **0.25333** | **0.76600** | **+0.00600** |
| v1 File-only | 0.23333 | 0.20667 | 0.27333 | 0.76000 | 0 |
| v2 full | 0.26000 | 0.22333 | 0.27667 | 0.74233 | -0.01767 |
| v2 Voice-only | 0.23333 | 0.22333 | 0.27333 | 0.75667 | -0.00333 |
| v2 Music-only | 0.23333 | 0.20667 | 0.27667 | 0.75900 | -0.00100 |
| v2 File-only | 0.26667 | 0.20667 | 0.27333 | 0.74333 | -0.01667 |

선택 Music residual은 exact-v50 cell error에서 FF `-0.040`, FR `-0.0733`, RF `0`으로
fake music 쪽을 개선했지만 RR false positive error는 `+0.0333` 악화했다. 즉 이
challenger의 핵심 tradeoff는 “가짜 음악 recall 증가 vs 진짜 음악 false alarm 증가”다.

## frozen 산출물

- matrix: `configs/three_stream_all_type_v57.yaml`
- exclusion list: `configs/three_stream_all_type_v57_exclusions.txt`
- selected challenger: `configs/three_stream_all_type_v57_selected.yaml`
- clean submission archive: `v50_v57m_challenger_v2.zip`
  (`a5730fde4e0c49100ee9aca5cc373eb550bf02948517072f23afe02fd62b91d6`)
  - compressed 8,283,677,531 bytes, expanded 9,185,965,040 bytes, 110 members
  - roots exactly `model/`, `script.py`, `requirements.txt`; no duplicate, unsafe,
    or symlink member; full CRC passed
  - offline clean/G.711/G.722 entrypoint smoke passed. The runtime audit confirmed
    byte-for-byte string preservation of ID, File, Voice, and both Presence fields,
    while Music alone changed in all 3/3 rows.
- v1 checkpoint SHA-256:
  `1f7904ee37f4fb16b0d699945e8898a0476c42e2fffb2e6c5971211bf3423b99`
- v2 checkpoint SHA-256:
  `e79637fe113bd9b9032361a28b757eb8dbf91fe7c09b2858ed3dec89ca960b7b`
- 독립 expert logit/residual export:
  `reports/three_stream_all_type_v57_strict/evaluation_v3/*_expert_logits.csv`
- exact-v50 변경 감사:
  `reports/three_stream_all_type_v57_strict/v50_authorized_v2/`
- runtime은 exact v50의 모든 expert가 끝난 뒤 v1 Music residual만 적용하며 File,
  Voice, 두 Presence 열을 bit-exact로 보존한다.

ZIP은 fixed one-shot gate에서 이 후보가 exact v50을 이긴 경우에만 공식 제출 대상으로
승격한다.
