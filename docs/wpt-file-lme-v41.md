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

## Router 및 component-OR 후처리 재검증

v41의 File 점수를 Voice/Music 전문가 출력이나 presence로 표본별 보정하는
router도 추가로 확인했다. raw OR, max, presence-product, logit gate, hard
presence gate를 여러 결합 비율로 비교했다. 강한 보정은 factorial dev나 YuE를
올릴 수 있었지만 factorial holdout 또는 phone에서 반드시 회귀했다. 예를 들어
raw OR 0.20은 YuE ADS를 `+0.01325` 올리는 대신 factorial을 `-0.00764`, phone을
`-0.00207` 낮췄다.

네 locked 축에서 모두 비회귀한 최선은 hard presence 0.3, weight 0.05였지만
phone ADS만 `+0.00121`이고 나머지는 완전히 같았다. 이 정도 차이는 표본 한두
개의 순위 변동이며, hidden에서 presence 오류를 File로 전파할 위험보다 작다.
따라서 component 기반 router도 최종 v41에는 넣지 않았다.

중간 latent router까지 포함한 앞선 비교에서도 bounded soft router는 개발
선택점수만 고정 MoE보다 `+0.00188`였고 phone locked ADS는 `-0.00693`였다.
hard router는 개발점수가 `-0.052~-0.081` 하락했다. 현재 데이터에서는
`전화/음악/음성 도메인 분류`가 `어느 전문가가 맞는지`와 같지 않다. 특히 한
파일 안에 음성과 음악이 동시에 또는 순차적으로 존재하므로 한 전문가로 보내는
방식보다, 모든 전문가를 유지한 뒤 task별 고정 logit 비율로 합치는 soft MoE가
더 일반적이었다.

최종 후보는 `wpt_file_lme_v41.zip`이다.

- 압축 크기: 8,286,764,468 bytes
- 압축 해제 크기: 9,189,599,997 bytes
- ZIP entry: 134개, 중복 0개, bytecode/cache 0개
- 최상위: `model/`, `script.py`, `requirements.txt`
- 최대 단일 member: 2,387,980,808 bytes
- 전체 CRC: 오류 없음
- SHA-256: `a301dad7bc53fc343d3b72f5c6c878a9d51f18014647f5462d731f50b43a3b25`
