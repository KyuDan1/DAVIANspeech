# v40: File 전용 multiview WPT

## 결론

v40은 [v39 고정 MoE](./wpt-fixed-moe-router-v39.md)의 모델과 학습 가중치를
그대로 사용하면서, WPT File 전문가만 원본 오디오의 3개 view에서 5개 view로
확장한다. Voice는 5개 중 시작·중앙·끝에 해당하는 index `0, 2, 4`만 사용하므로
v39 Voice 출력이 유지된다. Music과 두 presence 출력도 v39와 동일하다.

최종 File residual 비중은 0.60에서 0.75로 올렸다. factorial dev에서 0.60,
0.70, 0.75의 EER 순위가 같았고 모든 codec/channel subgroup ADS가 v38 이상이었다.
세 locked bank에서는 0.75가 가장 안정적으로 높았다.

## 1. 왜 File만 view를 늘리는가

대회 오디오는 최대 60초이며 음성과 음악이 순차적으로 존재할 수 있다. 파일 중
짧은 구간만 fake이면 시작·중앙·끝 세 지점 사이에 들어갈 수 있다. 반면 Voice
전문가는 세 view로 학습돼 view 수를 그대로 늘렸을 때 일부 locked Voice EER가
나빠졌다. 따라서 하나의 5-view forward에서 다음처럼 task별로 pooling한다.

```text
Voice/Music: view [0, 2, 4]의 LME(T=5)
File:        view [0, 1, 2, 3, 4]의 LME(T=5)
```

source separation은 여전히 사용하지 않는다. 추가 모델이나 checkpoint도 없으며,
WPT forward window 수만 파일당 3개에서 5개로 증가한다.

## 2. standalone ablation

| 설정 | dev ADS | factorial ADS | phone ADS | YuE ADS |
|---|---:|---:|---:|---:|
| 모든 task 3-view | 0.7695 | 0.7871 | **0.8696** | **0.8356** |
| 모든 task 5-view | **0.7793** | 0.7846 | 0.8681 | 0.8307 |
| Voice 3-view / File 5-view | 0.7724 | **0.7880** | 0.8689 | **0.8356** |

모든 task를 5-view로 바꾸면 dev는 좋아지지만 locked Voice/Music이 나빠진다.
File만 5-view를 사용하면 WPT File EER가 dev `0.2153→0.2095`, factorial
`0.2018→0.2000`으로 개선되고 phone/YuE는 동일하다. 제출에서는 WPT Music을
사용하지 않으므로 Music의 작은 수치 변화는 최종 출력에 전달되지 않는다.

## 3. 최종 v39 대비 결과

아래는 동일한 v38 anchor와 동일한 unified expert를 사용한 최종 결과다.

| 평가축 | v38 ADS | v39 ADS | v40 ADS | v40 - v39 |
|---|---:|---:|---:|---:|
| factorial dev | 0.74034 | 0.79392 | **0.79865** | +0.00473 |
| factorial holdout | 0.76821 | **0.81636** | **0.81636** | 0.00000 |
| phone factorial | 0.87021 | 0.89843 | **0.90257** | +0.00414 |
| YuE cross-component | 0.86649 | 0.89401 | **0.89673** | +0.00272 |

v40의 최종 EER는 다음과 같다.

| 평가축 | File EER | Voice EER | Music EER |
|---|---:|---:|---:|
| factorial dev | 0.1753 | 0.2514 | 0.2114 |
| factorial holdout | 0.1753 | 0.2057 | 0.1829 |
| phone factorial | 0.0859 | 0.1600 | 0.0750 |
| YuE cross-component | 0.0958 | 0.0833 | 0.1290 |

## 4. 실행 검증

- 제출 환경과 같은 전체 순서로 clean/전화/동시 혼합 3파일 CUDA smoke 성공
- v39와 비교해 Voice, Music, Voice Presence, Music Presence가 bit-exact
- File만 의도대로 변경
- B200, 2,124개 검증 파일, 5-view/batch 6 WPT 실행: 모델 로드 포함 약 28초
- WPT peak: allocated 3.78GiB, reserved 3.82GiB
- 관련 단위·회귀 테스트 11개 통과

최종 후보는 `wpt_multiview_file_v40.zip`이다.

- 압축 크기: 8,286,764,193 bytes
- 압축 해제 크기: 9,189,598,938 bytes
- ZIP entry: 134개, 중복 0개, bytecode/cache 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 최대 단일 member: 2,387,980,808 bytes
- 전체 CRC: 오류 없음
- SHA-256: `a43b747859067e958ab0dc6d6455b22b002dad5f81d3d590ca0ffd05536f81bd`
