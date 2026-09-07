# v18 vs EAT patch-graph: locked cell failure audit

## 결론

다음 병목은 **음악 단독 탐지가 아니라, 혼합물 안에서 어느 성분의 fake 증거가
File 판정으로 전달되는가**이다.

- EAT patch expert는 전화 혼합의 Music EER을 `0.415 → 0.085`로 크게 낮춘다.
- 이에 따라 `voice-real + music-fake`의 전화 File EER도 `0.370 → 0.200`
  (15% residual)으로 낮아진다.
- 그러나 `voice-fake + music-real` 전화 File EER은 `0.320 → 0.320`으로 그대로다.
- 일반 factorial의 동시 혼합에서는 이 셀이 더 심하다. File AUC가 v18
  `0.426`, patch `0.634`, 15% residual `0.467`이고 EER은 각각 `0.480`,
  `0.400`, `0.520`이다. patch에 보완 정보는 있지만 고정된 작은 residual로는
  충분히 전달되지 않는다.
- 따라서 다음 학습은 EAT Music head를 더 세게 섞는 실험보다, **동시 혼합의
  component-conditioned File head**, 특히 `voice-fake + music-real`을 직접
  최적화해야 한다.

이 분석은 학습이나 audio 재추론 없이 고정된 v18 및 seed01 EAT patch 예측만
사용했다. `fused05`는 v44, `fused15`는 v45와 동일하게 File/Music logit만 각각
5%/15% 결합하며 Voice는 v18 그대로다.

## 1. 전체 잠금 성능

| set | model | File EER | Voice EER | Music EER | ADS |
|---|---|---:|---:|---:|---:|
| factorial holdout | v18 | 0.2553 | 0.2229 | 0.2743 | 0.7455 |
|  | patch direct | 0.2705 | 0.3429 | **0.1943** | 0.7379 |
|  | v44, 5% | 0.2476 | 0.2229 | 0.2457 | 0.7579 |
|  | v45, 15% | **0.2305** | 0.2229 | 0.2171 | **0.7750** |
| phone factorial | v18 | 0.2583 | 0.1675 | 0.3450 | 0.7339 |
|  | patch direct | 0.1683 | 0.3000 | **0.0800** | 0.8319 |
|  | v44, 5% | 0.2041 | 0.1675 | 0.2450 | 0.7909 |
|  | v45, 15% | **0.1600** | 0.1675 | 0.1300 | **0.8475** |
| YuE cross-component | v18 | 0.1861 | 0.1021 | 0.1935 | 0.8284 |
|  | patch direct | 0.2181 | 0.4167 | 0.2097 | 0.7447 |
|  | v44, 5% | **0.1542** | 0.1021 | **0.1774** | **0.8493** |
|  | v45, 15% | **0.1542** | 0.1021 | **0.1774** | **0.8493** |

patch expert는 Music과 전화 File에는 강하지만 Voice에는 일관되게 약하다. 따라서
현재처럼 Voice 출력을 v18에 고정하는 선택은 맞다.

## 2. 요청한 핵심 셀

아래 값은 한 성분을 고정하고 나머지 성분 또는 File label만 바꾼 1:1 controlled
contrast의 EER이다. 고정 label 하나만 있는 개별 셀에서는 EER을 정의할 수 없으므로,
같은 layout의 대응 real/fake 셀을 묶었다.

### Music-only

| set | v18 | patch | v44 5% | v45 15% |
|---|---:|---:|---:|---:|
| factorial Music EER | **0.200** | 0.240 | 0.240 | **0.200** |
| phone Music EER | 0.280 | **0.075** | 0.190 | 0.125 |
| YuE Music EER | 0.000 | 0.000 | 0.000 | 0.000 |

즉 깨끗한 일반 music-only에서 patch가 항상 이기는 것은 아니다. 강점은 전화 채널과
혼합 상태이며, music-only만 더 학습하는 것은 현재의 핵심 병목을 직접 고치지 않는다.

### Voice-fake + Music-real: File 판정

| set/layout | v18 | patch | v44 5% | v45 15% |
|---|---:|---:|---:|---:|
| factorial concurrent | 0.480 | **0.400** | 0.480 | 0.520 |
| factorial partial overlap | 0.320 | 0.360 | 0.320 | **0.280** |
| factorial sequential | 0.240 | 0.320 | **0.200** | 0.240 |
| phone simultaneous | **0.320** | 0.360 | **0.320** | **0.320** |
| YuE concurrent | **0.375** | 0.500 | **0.375** | 0.500 |

가장 큰 실패는 **concurrent/simultaneous fake voice + real music**이다. 일반
factorial에서는 v18의 pair-ranking error가 `57.4%`이고 patch가 v18 오류 중
`64.1%`를 고치지만, 동시에 전체 pair의 `16.0%`를 새로 망친다. 결과적으로 patch
단독 AUC는 `0.634`로 v18 `0.426`보다 낫지만, 15% 고정 결합 AUC는 `0.467`에
그친다. 이 패턴은 단순한 global weight보다 content-conditioned fusion이 필요하다는
증거다.

### Voice-real + Music-fake: File 판정

| set/layout | v18 | patch | v44 5% | v45 15% |
|---|---:|---:|---:|---:|
| factorial concurrent | 0.400 | **0.160** | 0.320 | 0.280 |
| factorial partial overlap | 0.280 | **0.200** | **0.200** | **0.200** |
| factorial sequential | 0.360 | **0.160** | 0.360 | 0.200 |
| phone simultaneous | 0.370 | **0.110** | 0.270 | 0.200 |
| YuE concurrent | 0.375 | **0.250** | 0.375 | 0.375 |

이 축은 patch가 이미 잘 고친다. 전화 simultaneous에서 patch는 v18의 잘못 정렬된
pair 중 `96.0%`를 고치고 전체 pair의 `1.4%`만 새로 망친다. 다음 모델에서 이
Music 이득을 보존하는 것이 필수다.

## 3. Concurrent와 sequential의 차이

| set/layout | model | File EER | Voice EER | Music EER | 계산 ADS |
|---|---|---:|---:|---:|---:|
| factorial concurrent | v18 | 0.440 | 0.300 | 0.260 | 0.642 |
|  | v45 | 0.360 | 0.300 | **0.140** | **0.718** |
| factorial partial overlap | v18 | 0.320 | 0.300 | 0.320 | 0.684 |
|  | v45 | 0.247 | 0.300 | 0.240 | **0.7445** |
| factorial sequential | v18 | 0.247 | 0.180 | 0.300 | 0.7505 |
|  | v45 | **0.160** | 0.180 | **0.180** | **0.830** |
| YuE concurrent | v18 | 0.354 | 0.250 | 0.375 | 0.6605 |
|  | v45 | 0.354 | 0.250 | **0.250** | **0.698** |
| YuE sequential | v18 | **0.021** | 0.062 | 0.062 | **0.9585** |
|  | v45 | 0.104 | 0.062 | 0.062 | 0.917 |

순차 혼합은 각 component가 독립적인 시간 구간에 드러나므로 factorial v45에서 이미
ADS 약 `0.830`이다. 반면 concurrent는 Music은 좋아졌어도 File과 Voice가 겹침에
가려져 ADS `0.718`에 머문다. YuE는 표본이 layout당 32개뿐이라 EER 한 단계가 크지만,
여기서도 concurrent가 가장 어렵다는 방향은 같다. 다음 batch 구성은 sequential을
더 늘리는 대신 concurrent/partial overlap을 우선해야 한다.

## 4. 채널별 실패

### 일반 factorial, v18 → v45

| channel | File EER | Music EER | 해석 |
|---|---:|---:|---|
| clean FLAC | 0.287 → 0.195 | 0.293 → 0.207 | 개선 |
| MP3 64k | 0.236 → 0.180 | 0.327 → 0.173 | 개선 |
| noisy FLAC | 0.196 → 0.160 | 0.254 → 0.136 | 개선 |
| OGG 48k | 0.249 → 0.239 | 0.203 → 0.136 | File 개선 작음 |
| stereo WAV | 0.255 → 0.255 | 0.258 → 0.173 | File 정체 |
| telephone FLAC | 0.330 → 0.284 | 0.327 → 0.207 | 여전히 최악권 |

### 전화 factorial, v18 → v45

| channel | File EER | Voice EER | Music EER |
|---|---:|---:|---:|
| G.711 μ-law | 0.190 → 0.127 | 0.090 → 0.090 | 0.300 → 0.110 |
| G.726 24k | 0.190 → 0.097 | 0.070 → 0.070 | 0.270 → 0.110 |
| resample 8k | 0.200 → 0.130 | 0.050 → 0.050 | 0.350 → 0.140 |
| Opus narrowband | **0.447 → 0.337** | **0.240 → 0.240** | 0.360 → 0.070 |

Music artifact는 Opus에서도 복구되지만 File과 Voice는 그렇지 않다. 전화 전용 다음
목표는 모든 코덱을 평균내는 것이 아니라 **Opus narrowband File EER `≤0.20`**이다.
factorial의 `SNR=-10 dB`에서는 v45 Voice EER가 `0.416`, File EER가 `0.311`이고,
phone의 `SNR=-6 dB`에서도 각각 `0.319`, `0.278`이다. 낮은 voice-to-music SNR의
동시 혼합이 전화 병목과 같은 현상이다.

## 5. Generator별 실패

factorial holdout의 fake music generator별 v45 EER 상위는 다음과 같다. generator당
양성 수가 9~18개이므로 절대 순위보다는 목표 데이터 방향으로 사용해야 한다.

| generator | v18 EER | v45 EER | v45 conditional macro AUC |
|---|---:|---:|---:|
| ACE-Step | 0.332 | **0.268** | 0.804 |
| ElevenLabs | 0.311 | **0.250** | 0.823 |
| producer/compositional | 0.445 | **0.250** | **0.754** |
| Brev | 0.354 | **0.235** | 0.911 |
| Suno | 0.354 | **0.235** | 0.915 |
| Mubert | 0.300 | **0.216** | 0.919 |

전화 music-only에서는 Suno(chirp-v3.5)가 `0.316 → 0.090`으로 좋아지지만 Udio는
`0.403 → 0.182`로 상대적으로 어렵다. 새로운 학습/validation의 generator 목표는
각 holdout generator에서 **EER `≤0.20`, layout-conditioned macro AUC `≥0.85`**다.

Voice는 v45에서 v18 그대로다. factorial의 Qwen3-TTS custom 0.6B와 fine-tuned
1.7B EER가 각각 `0.287`, `0.314`로 CosyVoice `0.160`, F5-TTS `0.167`보다 어렵다.
전화 speech-only에서는 non-autoregressive neural vocoder EER `0.178`, unknown
vocoder `0.188`이 상대적으로 나쁘다. patch Voice는 모든 큰 vocoder 그룹에서 더
나쁘므로 Voice head로 사용하면 안 된다.

## 6. 다음 학습 목표와 합격 기준

리더보드 1위 Total `0.837980`을 CPS `0.98932`로 넘으려면 ADS가
`>0.821164`여야 한다. ADS error budget은 다음과 같다.

```text
0.5 * File_EER + 0.2 * Voice_EER + 0.3 * Music_EER < 0.178836
```

factorial의 현재 Voice EER `0.2229`를 그대로 둔다면, 여유를 둔 ADS `0.83`
목표는 대략 **File EER `≤0.15`, Music EER `≤0.165`**를 요구한다. 현재 v45는
`0.2305/0.2171`이므로 둘 다 아직 부족하다.

다음 학습의 구체적 합격 기준은 다음과 같다.

1. `concurrent voice-fake + music-real` File EER: factorial `0.52 → ≤0.20`,
   phone `0.32 → ≤0.20`; 동시에 `voice-real + music-fake`의 개선을 잃지 않는다.
2. layout별 factorial: concurrent File `0.36 → ≤0.15`, partial-overlap File
   `0.247 → ≤0.15`; sequential ADS `≥0.83`은 유지한다.
3. channel: Opus narrowband File `0.337 → ≤0.20`; 모든 channel Music EER
   `≤0.18` 및 어떠한 channel도 v18보다 악화되지 않는다.
4. generator holdout: Music EER `≤0.20`, conditional macro AUC `≥0.85`;
   특히 producer/compositional, ACE-Step, ElevenLabs, Udio를 hard group으로 둔다.
5. Voice specialist는 XLS-R 계열을 유지하고 Qwen3-TTS 두 계열 EER를 `≤0.20`으로
   낮춘다. EAT patch의 Voice logit은 결합하지 않는다.
6. 전체 잠금 gate: factorial ADS `≥0.83`, phone/YuE ADS 각각 `≥0.84`, 세 평가
   어느 것에서도 v18 대비 하락이 없어야 한다.

추천 구조는 hard audio-type router가 아니라, 원본 mixture에서 얻은
`voice evidence`, `music evidence`, overlap/SNR/channel latent를 입력으로 받는
**monotone component-conditioned File compositor**다. 학습 batch는
`RR/VF-MR/VR-MF/VF-MF × concurrent/partial × channel`을 같은 질량으로 만들고,
File loss 중 최소 절반을 두 교차 셀에 둔다. generator와 source ID 단위로 split한
GroupDRO는 이 cell-balanced batch 위에서 적용해야 한다.

단순 확률 OR도 확인했다. v45 File과 component OR를 30% 섞으면 세 잠금 ADS 평균이
`0.82392 → 0.82941`로 조금 오르지만, factorial ADS는 `0.775 → 0.786`에 불과하다.
따라서 component 논리를 명시하는 방향은 맞지만 후처리 OR만으로는 부족하고,
overlap/channel latent를 본 학습된 compositor가 필요하다.

## 7. 산출물과 재현

- `overall.csv`: 전체 File/Voice/Music EER와 ADS
- `controlled_contrasts.csv`: component를 고정한 controlled EER/AUC
- `pair_complementarity.csv`: patch가 v18의 pair 오류를 고친 비율과 새로 만든 비율
- `subgroups.csv`: layout/channel/SNR/overlap/order별 EER/AUC
- `generator_contrasts.csv`: fake generator 대 real 및 layout-conditioned macro AUC
- `phone_vocoder_contrasts.csv`: 전화 speech vocoder family별 결과
- `cell_error_rates.csv`: 전체 EER operating point에서 cell별 오류율과 score 분위수
- `component_or_sweep.csv`: component OR File compositor의 제한된 상한

재현:

```bash
/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/python \
  reports/eat_patch_graph_v1/cell_audit_v1/audit.py
```

주의: phone 1,200행은 300개 parent 각각의 네 channel 변형이다. overall EER은 대회
형식을 따라 1,200행으로 계산했지만, channel별 결론과 다음 목표는 네 channel을
각각 분리해 확인했다. YuE contrast는 셀당 8개뿐이므로 작은 차이를 확정적 개선으로
해석하지 않는다.
