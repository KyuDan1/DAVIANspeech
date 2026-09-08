# SOTA candidate v52: frozen and rejected on blind v7

`sota_candidate_v52_frozen`은 separation-free Music-only v51 위에 이미 동결된
component-consistent File residual을 **마지막 단계**로 결합한 패키지다.

## Prospective blind v7 결론

후보를 먼저 완전히 동결한 뒤 source/speaker/song/hash가 분리된 blind v7 360개를
한 번만 추론·채점했다.

| 모델 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| frozen v50 | **0.21111** | 0.20000 | 0.30556 | **0.76278** |
| frozen v52 | 0.21296 | 0.20000 | 0.30556 | 0.76185 |

v52의 Music residual은 전체 Music EER 순위를 바꾸지 못했고 마지막 File consistency가
File EER를 `+0.00185` 악화시켜 ADS가 `-0.00093` 내려갔다. 따라서 v5 proxy의 예상
Total `0.83843`은 새 source/generator에 일반화되지 않았으며, 이 패키지는 공식 제출
후보에서 제외한다. 결과를 본 뒤 weight나 threshold를 다시 맞추지 않았고 v7은 즉시
retrospective diagnostic으로 퇴역했다.

실행 순서는 다음과 같이 고정했다.

```text
기존 v50 experts
  → Music-only 3-head residual (MUSIC_FAKE_PROB만 변경)
  → WPT File expert
  → Spectra/XLS-R sparse Voice experts
  → component-consistent File residual (FILE_FAKE_PROB만 변경)
  → cache cleanup
```

마지막 File 식은 mixed gate `(VP >= 0.10 and MP >= 0.20)`에서만 적용하며,
`FileLogit = 0.8 * FileLogit + 0.2 * logit(max(final Voice, final Music))`이다.
따라서 Music v51과 Voice v49의 **최종** 성분 점수를 읽는다. Voice/Music authenticity와
두 presence 열은 문자열 값까지 그대로 보존하는 테스트를 포함했다.

## 로컬 결과

| bank | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| codec mixed dev v4 | 0.22000 | 0.20667 | 0.25333 | **0.77267** |
| codec mixed blind v5, retrospective | 0.20000 | 0.11667 | 0.18333 | **0.82167** |

v5 ADS에 현재 CPS 참고값 0.9893078836을 대입한 산술 Total은 0.8384307884다.
목표 0.83798보다 약 0.00045 높지만, v5는 240개 retrospective proxy이므로 실제
리더보드 우위를 주장하지 않는다. v6은 이 조합으로 inference하거나 선택에 쓰지
않았다. v7 결과는 위의 단 한 번의 prospective 확인에만 사용했으며 이후 선택에는
사용하지 않는다.

## 패키지 검증

- 디렉터리: `sota_candidate_v52_frozen/`
- top-level: `model/`, `script.py`, `requirements.txt` 정확히 3개
- 압축 전 크기: 9,191,998,925 bytes
- 파일 수: 112
- `script.py` SHA-256: `9dde04b637e6184a49d23a8ca17f733b9ef938639a6c160724807a2a1b1f5f0f`
- requirements: `onnxruntime-gpu==1.23.2`
- source 및 생성 entrypoint compile 통과
- Music/File 결합 focused tests: `17 passed`

재현 빌더 `scripts/build_sota_candidate_v52_submission.py`는 입력 Music v51의 script,
requirements, runtime, 세 head를 SHA-256으로 검사한 뒤에만 hardlink package를 만든다.
생성 뒤 top-level과 실행 순서를 다시 검증한다. 설정과 모든 hash는
`configs/sota_candidate_v52_frozen.yaml`에 기록했다. ZIP과 공식 제출은 수행하지 않았다.
