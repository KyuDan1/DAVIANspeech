# v79 결과: WavLM LoRA를 추가할 근거는 아직 없음

6 epoch의 paired 학습과 개발 음성 4,016개 평가가 완료됐다. 동일 ID/정답인지 검사한
조건별 분석도 완료했다. checkpoint는 개발 점수를 보기 전에 마지막 epoch로 고정했다.

| Voice 평가 조건 | 개수 | 기존 WavLM | head 추가 학습 | head + LoRA | 기존 XLS-R 참고 |
|---|---:|---:|---:|---:|---:|
| 전체 | 4,016 | 20.49% | **18.95%** | 19.30% | 12.62% |
| 음성만 | 316 | 23.42% | 22.15% | 22.15% | 18.03% |
| 실제 음악: RR 대 FR | 1,850 | 19.14% | 15.89% | 16.11% | 12.54% |
| 가짜 음악: RF 대 FF | 1,850 | 21.30% | 20.97% | 20.65% | 12.76% |

수치는 Voice EER이며 낮을수록 좋다. R/F 순서는 **음성, 음악**이다. 음악의 진위를
맞힌 비율이 아니라 해당 배경에서 **음성의 진위**를 구별한 결과다. 단일 RF 또는 FR
그룹만으로는 양쪽 Voice class가 없어 EER을 계산할 수 없다.

## 해석

- 추가 head 학습은 기존 WavLM보다 전체 EER을 약 1.54%p 줄였다.
- LoRA는 같은 추가 학습의 대조군보다 전체 EER이 약 0.35%p 높다.
  마지막 4개 block Q/V의 rank 8 LoRA가 이번 조건에서 효과적이라는 증거는 없다.
  WavLM의 모든 적응 방식이 실패한다는 결론은 아니다.
- 가짜 음악 조건에서는 LoRA가 대조군보다 약 0.32%p 낮지만, 실제 음악 조건에서는
  약 0.22%p 높다. 작은 일부 slice의 개선만으로 새 expert/router를 추가하지 않는다.
- 참고 XLS-R은 이 개발셋에서 여전히 더 낮은 EER이다. 이 표의 기존 encoder 행은
  별도 실행 결과이므로 LoRA 대조군처럼 완전히 동일한 학습 실험으로 간주하지 않는다.
  다음 v80에서는 기존 XLS-R global head도 동일한 native 실행에서 다시 계산한다.

0.5 고정 threshold에서 control의 RF false-positive rate는 29.08%, RR은 16.76%다.
하지만 0.5가 모델 간 최적 threshold라는 뜻은 아니고, RF의 배경 가짜 보컬 정답에
불확실성이 있어 이 차이를 모두 모델 오류라고 확정할 수도 없다.

## 판단과 다음 행동

이번 WavLM head/LoRA를 제출에 추가하지 않는다. 기존 공식 최고 점수 0.77743을
넘었다고 주장할 수 없고, 공식 제출은 하지 않았다. 다음 우선순위는 XLS-R의 Voice
표현에서 짧은 가짜 구간을 직접 감독하는 v80이다. 기존 v71 EAT 구간 학습과 구분한다.

소요 시간은 모델 준비 뒤 학습+개발 평가 약 34분 29초, peak CUDA 약 5,549MiB였다.
B200 연구 환경의 수치이며 L4 전체 제출 시간이나 학습 일반화 성능을 보증하지 않는다.
보호 Suno/v74/eval은 학습 또는 이번 checkpoint 선택에 사용하지 않았다.

- 학습 결과: `reports/wavlm_voice_lora_v79/full/report.json`
- 재현 가능한 slice 분석: `scripts/audit_voice_candidates_v79.py`
- ID/정답 일치 검증 및 결과: `reports/wavlm_voice_lora_v79/context_audit/report.json`
- 세부 EER·고정 threshold 오류율: 같은 디렉터리의 `slices.csv`

통계적 유의성 또는 비공개 테스트 점수 예측은 이 분석에서 주장하지 않는다.

## 보완성 확인: 단독 EER만으로 배제하지 않기

단독 모델이 약해도 기존 모델의 오류를 보완할 수 있으므로, 같은 4,016개에서 **고정
50:50 logit 평균**을 추가 확인했다. 확률 clip은 기존 v73과 동일한 1e-6이고,
가중치·threshold 탐색이나 추가 학습은 하지 않았다.

| 고정 Voice 조합 | 전체 EER | RR 대 FR | RF 대 FF |
|---|---:|---:|---:|
| XLS-R + SPEAR | **11.58%** | **10.92%** | **12.11%** |
| XLS-R + WavLM head 추가 학습 | 13.32% | 12.00% | 14.38% |
| XLS-R + WavLM LoRA | 13.42% | 12.00% | 14.27% |

WavLM 조합은 실제 음악 배경에서 XLS-R 단독 12.54%보다 낮지만, 전체 및 가짜 음악
배경에서는 더 높고 기존 XLS-R+SPEAR를 넘지 못했다. 이 고정 조합으로 교체할 근거는
없다. 가능한 모든 가중치/router를 검증한 결과는 아니다. v73의 Voice 조합이 여기서
좋다고 해서 v74에서 실패한 v73 **전체 파이프라인**을 다시 승격시키지는 않는다.

재현: `scripts/audit_wavlm_complement_v79.py`.
결과: `reports/wavlm_voice_lora_v79/complement_context_audit/report.json`.
혼합 출력 CSV는 Voice만 포함한 개발 분석용이며 제출 파일이 아니다.
