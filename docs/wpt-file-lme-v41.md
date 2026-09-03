# v41: File 전용 LME 온도 완화

## 결론

v41은 [v40](./wpt-multiview-file-v40.md)의 5-view WPT File pooling 온도만
`T=5`에서 `T=2`로 낮춘다. Voice는 기존 3-view `T=5`, Music과 두 presence는
v38 출력을 그대로 유지한다. 모델, checkpoint, view 위치, MoE 가중치는 모두
v40과 동일하다.

높은 온도는 한 view의 극단적인 artifact에 민감하다. 파일당 view 수를 3개에서
5개로 늘린 뒤에는 전화 codec이나 자연 오디오의 국소 spike가 soft-max를 지배할
수 있으므로, File에만 더 부드러운 pooling을 적용했다.

## 전체 개발 bank 선택

학습에 사용하지 않은 8개 개발 bank, 총 3,976개 파일의 5-view logit을 한 번
추출하고 pooling만 바꿨다. 선택점수는 File EER의 평균과 최악값을 같은 비중으로
계산했다.

| File pooling | 평균 File EER | 최악 File EER | 선택점수 |
|---|---:|---:|---:|
| LME T=1.5 | **0.12432** | 0.21527 | 0.83020 |
| LME T=2 | 0.12697 | **0.20945** | **0.83179** |
| LME T=5 | 0.12768 | **0.20945** | 0.83143 |
| mean에 가까운 T=0.25 | 0.12677 | 0.20764 | 0.83280 |

T=0.25가 숫자상 가장 높지만 `external_mixed_v1` File EER가 T=5의 0.15에서
0.18로 악화돼 채택하지 않았다. T=2는 worst-domain을 유지하면서 평균도 T=5보다
낮고, 기존 훈련 온도에서 지나치게 멀지 않은 보수적인 선택이다.

## v40 대비 최종 결과

| 평가축 | v40 ADS | v41 ADS | 변화 | v41 File EER |
|---|---:|---:|---:|---:|
| factorial dev | **0.79865** | **0.79865** | 0.00000 | 0.17527 |
| factorial holdout | 0.81636 | **0.81927** | +0.00291 | 0.16945 |
| phone factorial | 0.90257 | **0.90343** | +0.00086 | 0.08414 |
| YuE cross-component | **0.89673** | **0.89673** | 0.00000 | 0.09579 |

네 축 모두 비회귀이며 factorial과 phone File 순위가 개선됐다. 차이가 작으므로
hidden 점수 상승을 보장하지는 않지만, 전체 개발 bank 선택과 두 독립 locked
환경이 같은 방향을 지지한다.

## 제출 검증

- clean/전화/동시 혼합 3파일 전체 CUDA smoke 성공
- v40 대비 Voice, Music, Voice Presence, Music Presence가 bit-exact
- 관련 단위·회귀 테스트 16개 통과

최종 후보는 `wpt_file_lme_v41.zip`이다.

- 압축 크기: 8,286,764,468 bytes
- 압축 해제 크기: 9,189,599,997 bytes
- ZIP entry: 134개, 중복 0개, bytecode/cache 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 최대 단일 member: 2,387,980,808 bytes
- 전체 CRC: 오류 없음
- SHA-256: `a301dad7bc53fc343d3b72f5c6c878a9d51f18014647f5462d731f50b43a3b25`
