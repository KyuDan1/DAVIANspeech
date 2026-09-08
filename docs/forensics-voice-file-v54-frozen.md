# Forensics XLS-R Wild Voice + final File v54

2026-09-04 기준. v54는 frozen v50을 그대로 실행한 뒤 원본 혼합음의 외부 Voice
전문가를 7.5%만 더하고, 마지막에 이미 동결된 component-consistent File 20%를
적용한 실험 후보이다. prospective v8에서 Voice는 개선했지만 ADS가 하락해 최종
후보로 기각했다. ZIP은 만들지 않았고 package directory까지만 생성했다.

## 최종 실행 순서

1. `sparse_voice_wpt_file_v50_frozen`의 모든 단계를 그대로 실행한다.
2. 원본 오디오에서 처음·중간·끝에 균등 배치된 5초 window 세 개를 만든다.
3. `Forensics XLS-R Wild`가 낸 real logit의 부호를 뒤집어 fake logit으로 만들고
   세 logit을 산술 평균한다.
4. `VoiceLogit = 0.925 * v50 VoiceLogit + 0.075 * ForensicsLogit`을 적용한다.
5. 혼합 presence gate 안에서 최종 Voice/Music max를 File에 20% 전달한다.
6. 임시 통계를 삭제하고 `submission.csv`만 남긴다.

Forensics는 stem이 아니라 **original mixture**를 직접 본다. 세 window를 한 번에
batching하며, 파일 사이의 score나 통계는 절대 공유하지 않는다. Music과 두 Presence
열은 읽거나 다시 쓰지 않는 Voice-only residual이다. 이후 File 단계만 최종 Voice와
Music을 소비한다.

## 왜 7.5%인가

외부 모델은 clean/telephone pure speech에서는 매우 강하지만 음악이 섞인 직접 EER는
v50보다 약했다. 따라서 replacement나 hard route가 아니라 낮은 diversity residual로만
사용한다. `0.075`는 development에서 고정했고, v5/v6/v7 회고 결과를 보고 바꾸지
않았다.

| 평가 bank | 역할 | v50 Voice EER | +Forensics 7.5% |
|---|---|---:|---:|
| codec mixed v4 | 선택 | 0.20667 | **0.19667** |
| codec mixed v5 | 회고 전용 | 0.11667 | **0.10833** |
| codec mixed v6 | 회고 전용 | 0.08333 | **0.07500** |
| codec mixed v7 | 회고 전용 | 0.20000 | **0.19444** |

aggregate Voice EER는 네 bank에서 모두 개선됐다. 다만 v5와 v7의 Opus 단독 subgroup은
악화됐으므로 weight를 키우거나 전화 hard route로 바꾸지 않는다.

## Prospective v8 판정

후보와 비교식까지 고정한 뒤 source-disjoint `codec_mixed_blind_v8` 360개에서 v50과
v54의 패키지 entrypoint를 각각 실행했다. 두 CSV가 완성되기 전에는 truth를 열지
않았고, scoring 직후 v8을 회고 전용으로 퇴역시켰다.

| 후보 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| v50 | **0.25741** | 0.34444 | 0.24444 | **0.72907** |
| v54 | 0.27593 | **0.31667** | 0.24444 | 0.72537 |

Forensics residual은 Voice 순위를 실제로 보완했지만, 그 Voice를 component max로 File에
전달하면서 File 순위가 더 크게 나빠졌다. 따라서 v54 전체는 **REJECT**다. 이 결과로
가중치나 File 식을 재튜닝하지 않으며, package directory는 재현용으로만 보존한다.

모델 출처와 라이선스는 다음과 같다.

- Hugging Face: `eliya/forensics_0.3B_xlsr_wild_deepfake_classifier`
- revision: `aa10055181a35eb7b4916175d2d72266def78c1b`
- license: CC-BY-NC-4.0
- checkpoint SHA-256:
  `39e37fa5a958c2b46ebb6a5937874c36c39003906dcdd5d0aba6154bc6b2dc21`

대회는 최소 비영리 사용이 허용된 공개 모델을 허용하지만, 2차 보고서에는 이 모델과
CC-BY-NC-4.0 조건을 반드시 명시해야 한다.

## File stage

Forensics Voice 이후 아래 고정식을 마지막에 적용한다.

```text
mixed = VoicePresence >= 0.10 and MusicPresence >= 0.20
component = max(final VoiceFake, final MusicFake)
if mixed:
    FileLogit = 0.80 * v50 FileLogit + 0.20 * componentLogit
```

이 File 식은 v50 기준 v4/v5/v6 File EER를 각각
`0.23333→0.22000`, `0.21667→0.20278`, `0.23333→0.18611`로 개선했던 그대로다.
Forensics 적용 뒤 weight나 threshold를 다시 고르지 않았다.

## 오프라인·실행 검증

패키지 자체의 vendored `model.py`, XLS-R config와 SafeTensors checkpoint로 모델을
로드했다. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`에서 3-window forward가 finite
shape `(3,)`로 끝났으므로 외부 다운로드가 없다. 공개 checkpoint에만 남아 있는 legacy
`projection.*` 네 tensor 외의 missing/unexpected key가 있으면 실행을 중단한다.

실제 package entrypoint를 clean/G.711/G.722 3파일에 처음부터 끝까지 실행했다.

| 측정 | 결과 |
|---|---:|
| v54 full smoke | 109.441초, exit 0 |
| 동일 3파일 v50 control | 128.073초, exit 0 |
| B200 steady 24-window batch | 551.29 windows/s |
| B200 batch24 peak reserved | 5,624,561,664 bytes |

작은 smoke는 여러 대형 checkpoint를 GPFS에서 순차 로드하는 고정 초기화가 대부분이라
파일 수에 선형 외삽하면 안 된다. 실제 Forensics steady compute만 보면 1,200파일의
3,600 window는 B200 약 6.5초다. L4의 하드웨어 차이와 오디오 decode를 크게 보수적으로
잡아도 추가 단계는 약 2--5분으로 예상한다. 기존 계열 제출이 약 40분이었으므로 60분
한도 안으로 판단한다.

full smoke의 v50/v54 CSV를 문자열 단위로 비교했다. File과 Voice는 의도대로 바뀌었고,
다음은 bit-exact였다.

- `MUSIC_FAKE_PROB`
- `VOICE_PRESENT_PROB`
- `MUSIC_PRESENT_PROB`

## 크기와 구조

최종 top-level은 정확히 `model/`, `script.py`, `requirements.txt` 세 개다. data,
output, symlink, `__pycache__`, `.pyc`는 제거했다.

- expanded file bytes: `10,445,056,232` bytes
- expanded limit: `32,000,000,000` bytes
- file count: 115
- counting-only ZIP bytes: `9,470,639,888` bytes
- ZIP limit margin: `529,360,112` bytes
- requirements: 기존과 같은 `onnxruntime-gpu==1.23.2` 한 줄
- 기준 `sparse_voice_v49_base.zip`: `unzip -t` 전체 CRC 통과
- ZIP은 생성하지 않음

압축 크기는 최종 counting-only ZIP 검증값을 config에 기록했다. counting writer는 실제
모든 파일을 동일한 deflate level 1로 통과시키되 archive를 디스크에 쓰지 않으므로,
“package까지만 만들라”는 조건을 지키면서 10 GB 제한 여유를 확인한다.

## 재현 파일

- runtime: `src/forensics_xlsr_wild_fusion.py`
- builder: `scripts/build_forensics_voice_file_v54_submission.py`
- frozen config: `configs/forensics_voice_file_v54_frozen.yaml`
- focused tests: `tests/test_forensics_voice_file_v54.py`
- package: `forensics_voice_file_v54_frozen/`
- smoke comparison: `reports/forensics_voice_file_v54_smoke/`

기존 v50 package는 수정하지 않았으며 ZIP, commit, push, submit은 수행하지 않았다.
