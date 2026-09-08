# v77: 별도 평가 서버 핵심 환경 검증

## 발견과 조치

기존 연구 환경은 Torch/torchaudio 2.8.0+cu128, NumPy 2.4.6, pandas 3.0.5,
librosa 0.11.0이었다. 이는 사용자가 제공한 평가 서버의 핵심 버전과 다르다.
진행 중인 실험을 유지하기 위해 기존 환경은 변경하지 않았다.

시스템 패키지를 상속하지 않는 별도 venv를
`/tmp/davian_grader_v77.o2Sm4v/venv`에 생성했다. Python은 기존 설치의 3.11.16이며
사용자가 제공한 서버의 3.11.15와 patch 버전 차이는 남는다. `/tmp`는 영구 보관 장소가
아니므로 환경이 없어지면 아래 설정으로 다시 만들어야 한다.

버전 설정은 `configs/requirements-grader-v77.txt`에 있다. 이 파일은 **로컬 검증 전용**이며,
평가 서버가 이미 제공하는 torch 등을 재설치하게 만들 수 있으므로 제출 requirements.txt로
복사하지 않는다.

설치에는 [PyTorch 공식 이전 버전 안내](https://pytorch.org/get-started/previous-versions/)의
2.7.1 CUDA 12.8 배포 저장소를 사용했다.

```bash
/tmp/davian_grader_v77.o2Sm4v/venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r configs/requirements-grader-v77.txt
/tmp/davian_grader_v77.o2Sm4v/venv/bin/python -m pip check
```

## 완료한 검사

`pip check`가 의존성 오류 없이 통과했다. `scripts/report_grader_environment_v77.py`로
설정한 20개 패키지 버전을 모두 확인하고, 다음 실제 실행도 완료했다.

- Torch 2.7.1+cu128 / torchaudio 2.7.1+cu128 native import 및 resample.
- NumPy 1.26.4 / pandas 2.0.3 binary import.
- CUDA BF16 행렬 연산. CUDA 12.8, cuDNN 90701로 보고됐다.

근거: `reports/grader_environment_v77/report.json`의
`status=complete_core_environment_smoke`.

## 아직 별도 확인해야 하는 것

이 환경에서 동일 v73 후보를 기존 개발 파일 80개에 실제 실행하고, 원래 예측과
최대 오차 0.0005 이하인지와 파일 순서를 뒤집어도 동일한지를 확인하는 작업을 시작했다.
출력: `reports/component_composition_v73/grader_stack_v77/`.
해당 `report.json`의 `pass_check`를 확인하기 전에는 모델 동등성 통과가 아니다.

이번 환경은 B200 하드웨어이며 CPU/RAM 제한도 적용하지 않았다. 실제 L4 성능,
6 vCPU에서의 복원·추론 시간, 설치 제한, 전체 submission ZIP 실행, 공식 점수까지
재현했다고 주장할 수 없다. 서버에서 명시하지 않은 전이 의존성 버전도 다를 수 있다.
검증의 범위를 구분해 기록하고 자동 제출하지 않는다.

## 모델 출력 대조 완료

위 80개 개발 파일에서 원래 예측과의 최대 절대 차이는 **0.000389335**, 평균 절대 차이는
0.000001602였다. 사전에 정한 허용 오차 0.0005를 통과했으며, 같은 환경 안에서 파일
순서를 뒤집은 결과는 bit-exact였다. **연구 환경과 새 환경 사이의 출력이 bit-exact인
것은 아니다.** 실제 기록된 차이를 유지하며 허용 오차를 사후에 늘리지 않았다.

정방향·역방향 총 160회 추론은 모델 로딩 이후 34.48초, GPU peak allocated는
12,179.38MiB였다. `grader_stack_v77/report.json`의 `pass_check=true`를 확인했다.
이는 B200의 짧은 개발 음원에서 검증한 구현 호환성으로, L4의 전체 시간/메모리와
실제 제출 점수를 보증하지 않는다. 보호 평가 오디오를 이 검사에 사용하지 않았다.
