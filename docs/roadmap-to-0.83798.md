# 0.837980을 넘기기 위한 현재 결론과 실험 규약

작성일: 2026-09-04

## 목표를 수치로 고정

현재 1위는 Total `0.837980`, ADS `0.821160`, CPS `0.98932`다. 팀의 실제
최고 v18은 Total `0.7616379312`, ADS `0.7363412698`, CPS
`0.9893078836`이다. 팀 CPS를 그대로 유지할 때 1위를 넘는 데 필요한 ADS는
`0.8211658`이다.

ADS는 다음의 가중 오류를 1에서 뺀 값이다.

```text
ADS = 1 - (0.5 * File EER + 0.2 * Voice EER + 0.3 * Music EER)
```

따라서 가중 EER를 `0.263659`에서 `0.178834` 미만으로, 절대
`0.084825` 또는 상대 약 **32.2%** 줄여야 한다. CPS는 1위와 사실상 같으므로
현재 병목은 CPS가 아니라 ADS다. 작은 확률 보정 하나만으로 달성할 수 있는 차이가
아니다.

## 실제 리더보드에서 배운 것

| 제출 | 핵심 변경 | ADS |
|---|---|---:|
| v18 | channel/component-invariant MoE 5% | **0.736341** |
| v19 | 유사 seed 3개, 10% | 0.734214 |
| v32 | File attention 20% + phone route 25% | 0.723413 |
| v33 | v32 + Music residual 50% | 0.714841 |
| inv4-w30 | invariant 4-head 30% | 0.723976 |

결론은 세 가지다.

1. 큰 expert weight와 hard router는 로컬 개선을 실제 hidden으로 전달하지 못했다.
2. 전화 여부를 먼저 결정해 경로를 갈라 버리기보다, 같은 원본에서 channel-invariant
   evidence를 작은 residual로 결합하는 편이 안전했다.
3. 현재 추정 병목은 Music OOD와 동시 혼합의 File 판정이다. 특히 fake voice +
   real music에서 fake voice 증거가 반주에 묻힌다.

초기 `twin_v1`과 한 열만 0.5로 만든 공식 probe를 역산하면 당시 EER은 대략
File `0.27414`, Voice `0.21556`, Music `0.37143`이었다. 상수 점수의 대회식
EER가 0.5이므로 이 값은 점수 차이에서 정확히 복원된다. v18은 이후 세 항목을
동시에 조금 개선했지만, probe가 보여 주는 우선순위는 명확하다. Music이 가장
나쁘고 가중치도 0.3이며, File은 가중치가 0.5라 두 번째 핵심 병목이다. Voice만
더 좋게 만드는 것으로는 1위 격차를 메울 수 없다.

## v47과 WPT Voice 후보

v47의 **새 branch**는 분리 없이 원본 mixture에서 EAT patch와 SPEAR temporal
bin을 component-query MHFA로 읽고, Voice/Music 확률의 noisy-OR를 File에
반영한다. 다만 전체 v47은 v18 anchor를 유지하므로 기존 Demucs stem branch도
함께 실행한다. 즉 v47 전체를 separation-free라고 부르면 안 된다. 새 evidence가
분리 artifact에 의존하지 않고 기존 stem score를 낮은 비중으로 보완하는 구조다.
잠금 진단 ADS는 factorial/phone/YuE에서 `0.775818 / 0.826357 / 0.865216`이지만,
이 세트는 이미 반복 사용했으므로 공식 예상 점수로 부르지 않는다.

WPT/Spectra의 직접 residual은 factorial 개발셋만 사용해 다시 선택했다.

```text
Voice = 0.95 * logit(v47) + 0.05 * logit(WPT)
File  = v47 그대로
Music = v47 그대로
CPS   = v47 그대로
```

- 개발 ADS `+0.003429`
- 23개 통제 contrast 중 2개 개선, 회귀 0
- 선택 후 관찰한 기존 잠금 세트 ADS `+0.001143 / +0.001500 / +0.003750`
- File weight는 모든 non-zero 설정에서 개발 contrast 1개가 악화돼 0으로 고정

이는 안전한 보조 후보이지 1위 격차를 메우는 새 주력 모델은 아니다. 과거 잠금
결과를 보고 골랐던 Voice `.075` + File `.01` 설정은 폐기한다.

재현 빌드 `component_query_wpt_voice_v48_final.zip`은 압축
`8,279,337,977` bytes, 해제 `9,181,027,998` bytes다. entry 127개,
중복/unsafe path/CRC 오류는 모두 0이며 SHA-256은
`bdc5e0278a3f1b8ca9338022bde35f9c1f695c51865662a3fe7a48ea6eceb3b6`다.
3개 Suno 전체 CUDA entrypoint smoke에서 File/Music fake가 모두 0.5를 넘었고,
중간 cache 네 개가 모두 삭제되는 것까지 확인했다.

## 기각한 최신 가설

### post-hoc File compositor

개발에서는 File EER가 `0.2382 -> 0.1982`로 좋아졌지만 잠금 세 세트에서 모두
반대로 악화됐다. 최종 확률 세 개만 다시 조합하는 방법에는 새 정보가 없으며,
hidden에서 일반화되지 않았다.

### component-query seed05 hybrid

cell-balanced seed05는 7개 개발 bank의 선택 점수를 `0.816465 -> 0.820914`로
높였지만 seed별 hard-cell 변동이 컸다. 잠금 결과를 본 뒤 고른 hybrid weight는
절차상 오염됐으므로 패키징하지 않았다. 다음에는 seed ensemble보다 새로운
prospective bank를 먼저 확보한다.

### WPT + query Music consensus

7개 authorized dev의 leave-one-domain-out에서는 Music EER가 평균
`0.114158 -> 0.104966`으로 좋아졌지만 exact v47 chain에 넣으면 factorial-dev
Music EER가 `0.205714 -> 0.240000`, ADS가 `0.762052 -> 0.751766`으로
악화됐다. 좋은 독립 모델이 반드시 좋은 nested residual은 아니다.

### naive X-Codec와 real-only density

현재 데이터로 학습한 X-Codec 계열은 source-disjoint Music EER가 약
`0.50`, YuE `0.548`로 generator OOD에 실패했다. real-only MusicDET 재현도
factorial/phone Music EER가 `0.44~0.535`였다. 토큰이나 likelihood라는 이름만
추가하는 것은 해결책이 아니며, 학습 split과 representation 목적까지 바뀌어야 한다.

## 기존 평가셋의 한계와 새 prospective v3

`phone_factorial_1200_v1`은 300개 parent 모두 기존 개발/source bank에서 왔고
이미 32개 실험 스크립트에서 참조됐다. full factorial manifest도 dev/holdout과
locked 역할이 겹치며, 일부 forensic/temporal manifest는 내부 dev까지 train으로
등록돼 있었다. 이 데이터는 앞으로 channel mechanics용 retrospective diagnostic으로만
사용한다.

아직 사용하지 않은 MixFake official eval 40,000개에서 현재 모든 역할의 exact
identity와 SONICS canonical song group을 제외해도 35,578개가 남았다. 첫 예약은
SONICS의 AI 보컬이 `fake music`에 섞여 RF의 Voice truth를 깨는 것을 detector
실행 전에 발견해 무효 처리했다. 두 번째 예약은 MusicCaps caption/aspect의 보수적
instrumental 규칙으로 FakeMusicCaps 후보를 제한했고, PANNs vocal score와 Demucs
vocal-energy는 진위 detector가 아니라 의미 감사에만 사용했다. 최초 후보 120개 중
3개를 동결 임계값에서 교체했으며 교체 3개도 같은 임계값에서 모두 통과했다.

최종 1,200개 bank는 detector를 열기 전에 다음처럼 예약한다.

- 진위 4셀 `RR/RF/FR/FF`
- layout 3종 `concurrent/partial_overlap/sequential`
- 각 셀 20개 base: 총 240개
- 각 base를 `clean/G.711/G.722/Opus-NB/G.711->Opus` 5개 paired condition으로
  렌더링: 총 1,200개
- voice/music identity는 한 base에만 사용하고 5개 channel pair 안에서만 공유
- concurrent/partial SNR `-10/-5/0/+5/+10 dB`, overlap 25/50/75%, 순서 균형
- 최종 mixture 뒤에 통신 변환을 적용하고 source separation은 하지 않음
- truth를 locked role로 등록한 뒤 candidate와 weight를 먼저 동결하고 딱 한 번 채점

이 bank는 두 component가 모두 존재하는 어려운 혼합 ADS 전용 진단이다. 따라서
Voice/Music fake EER와 File EER는 계산할 수 있지만 presence label이 모두 1이라
CPS/Total은 정의되지 않는다. scorer는 이를 `NaN`과 `CPS_DEFINED=false`로 명시하며
임의 대체값을 만들지 않는다. CPS 개선은 component-absent parent를 포함한 별도의
prospective presence bank로 검증한다.

이 bank는 source-disjoint지만 현재 로컬 metadata만으로는 완전한
generator-disjoint라고 부를 수 없다. fake music 7개 계열이 모두 기존 실험에서
등장했고, ASVspoof attack/speaker provenance도 공식 protocol join이 필요하다.
새로운 music generator와 실제 통화 음성 RTCFake를 확보해야 최종 blind bank가 된다.

### 첫 동결 평가 결과

의미 감사를 통과한 instrumental v2를 동결한 v47/v48에 한 번만 채점했다. v47은
File/Voice/Music EER `0.28333/0.25500/0.24667`, ADS `0.73333`이었고,
이는 실제 v18 ADS `0.736341`과 `0.00301` 차이다. WPT Voice 5%인 v48은
Voice EER를 `0.24500`으로 낮춰 ADS `0.73533`을 기록했다.

가장 중요한 결과는 channel별 분해다. clean/G.722 ADS는
`0.85917/0.84417`로 이미 목표를 넘지만 G.711/Opus-NB/G.711→Opus는
`0.75000/0.63917/0.59361`이다. 특히 concurrent Opus-NB와 재압축에서
File/Voice/Music EER가 각각 `0.500/0.425/0.500`과
`0.550/0.475/0.475`까지 무너졌다. 따라서 다음 주력은 더 큰 post-hoc MoE가
아니라 clean evidence를 보존하는 paired-codec representation 학습과 ordered local
XLS-R evidence다. 이 bank는 채점 직후 retrospective 역할로 퇴역했으며 다음
checkpoint/weight 선택에는 쓰지 않는다.

### 누수 없는 codec development v4

퇴역한 v2의 점수나 예측을 전혀 읽지 않고 seed `20260905`로 새 source-disjoint
개발 bank를 만들었다. 첫 예약은 파일 ID가 달라도 같은 MusicCaps 곡의 다른
generator rendition 12개가 퇴역 v2와 겹치는 canonicalization 버그를 독립 검증에서
찾아 폐기했다. 수정 후 예약에서는 exact source뿐 아니라 speaker, archive member,
MusicCaps song group 등 8개 identity view를 모두 차단했다.

- RR/RF/FR/FF 4셀 x `concurrent/partial_overlap/sequential` 3 layout
- 셀당 10 base, base당 `clean/G.711/G.722/Opus-NB/G.711->Opus` 5 paired render
- 120 base / 600 audio / 240 unique sources
- FakeMusicCaps 5 generator에서 12개씩 균형 표집
- PANNs 0.20과 Demucs -1.5 dB 의미 검사 60/60 통과
- 등록된 47개 manifest와 124개 identity 비교 및 퇴역 v2 8개 view overlap 0
- archive 67개, source 240개, render 600개의 해시와 구조를 포함한 30/30 검증 통과

최종 truth SHA-256은
`6a87bc333eb88306d3f31002ac519bb484f903dd1c419824bbe1c28a5b41881b`다.
이 검증이 끝난 뒤에만 `development`로 등록했다. 따라서 반복적인 checkpoint/loss
선택에는 사용할 수 있지만 blind 성능 주장에는 사용하지 않는다. 첫 실패 빌드와
음향검사 실패 source tree는 `rejected_artifacts/`에 남겨 성공 데이터와 혼동하지
않는다.

## 큰 폭의 개선을 위한 모델 우선순위

### 1. Music source-restricted sequence detector

가장 큰 기대값은 Music EER 감소다. 공개 연구도 random split의 포화 점수와 달리
unseen-generator 제한에서 성능이 크게 무너짐을 보이며, generator별로 강한 표현이
다르다고 보고한다. CoMoE에서는 X-Codec과 MERT token이 학습 generator에 따라
서로 다른 강점을 보였다: <https://arxiv.org/abs/2606.08663>.

다음 모델은 이미 제출에서 계산하는 MERT/EAT representation을 재사용하되, 평균
embedding이 아니라 시간 순서를 보존한 token sequence를 본다. generator/source를
group으로 둔 leave-one-generator/source-out 및 Group-DRO로 checkpoint를 선택하고,
real source와 codec hard negative를 충분히 넣는다. MERT와 wav2vec2 dual stream의
상보성을 쓰는 CLAM 방향도 참고한다:
<https://openreview.net/pdf?id=65a6f4477fff1b4b63279d295d4f625c1b6113eb>.

### 2. mixture-native component MIL

분리 모델은 쓰지 않는다. 원본의 짧은 local token에 Voice/Music query를 각각 두고
시간별 존재와 진위를 동시에 예측한다. File은 clip-level 새 classifier 하나에만
맡기지 않고, local Voice/Music fake evidence의 differentiable noisy-OR와
clip evidence를 함께 학습한다. RR/RF/FR/FF, concurrent/partial/sequential을
동일 질량으로 sampling하며 `VF+MR`과 `VR+MF`의 양방향 rank loss를 직접 건다.

### 3. paired channel consistency

clean과 네 telephone rendering이 같은 base임을 학습에 명시한다. probability
일치만 강제하지 않고, component-query latent와 clean/phone 순위를 함께 보존한다.
hard phone router는 쓰지 않으며 soft channel expert는 새 prospective clean-phone
pair 모두에서 회귀가 없을 때만 최대 5%로 허용한다.

구현한 첫 실험 head는 기존 v47을 정확히 재현하는 0-초기화 residual로 시작한다.
EAT patch, SPEAR temporal bin, 원본 mixture의 ordered XLS-R window를 세 stream으로
읽고, 공식 task 비중 `Voice 0.2 / Music 0.3 / File 0.5`의 BCE와 RR/RF/FR/FF
rank loss를 함께 최적화한다. 같은 mixture의 clean logit/latent는 stop-gradient
teacher로 두고 codec render만 그쪽으로 당긴다. dataset x layout x channel의
batch-local soft worst-group loss를 추가하되 residual은 task별 최대 1 logit으로
제한한다.

checkpoint 선택은 큰 corpus가 작은 speech-only/music-only OOD corpus를 지우지
않도록 다음 하나의 고정식만 쓴다.

```text
0.50 * pooled official ADS
+ 0.25 * mean normalized-available domain ADS
+ 0.25 * worst normalized-available domain ADS
```

component 하나가 없는 domain은 정의된 EER의 공식 가중치만 재정규화한다. 전체
development를 합친 첫 항은 언제나 대회식 ADS 그대로 계산한다.

### 4. long-context composition evidence

Suno처럼 실제 악기 timbre를 사용하고 구성만 AI인 음악은 짧은 vocoder artifact만으로
탐지하기 어렵다. 30~60초 구간의 반복, 전개, transition을 보는 long-context branch가
필요하다. SONICS의 long-context spectro-temporal 접근을 참고하되
<https://openreview.net/pdf?id=PY7KSh29Z8>, 현재 대회 시간 안에서 같은 backbone의
temporal token을 저해상도로 재사용한다. 이 branch는 새로운 generator holdout에서
독립 개선을 보일 때만 낮은 비중으로 결합한다.

## 채택 기준과 제출 순서

새 후보는 다음 조건을 모두 만족해야 한다.

1. authorized development의 모든 domain에서 평균과 최악 ADS가 v47 이상
2. RR/RF/FR/FF × layout 및 clean/phone paired contrast에 회귀 0
3. fusion weight를 동결한 뒤 prospective v3를 한 번 열어 v47보다 개선
4. 1,200파일 L4 추론 60분 미만, ZIP 10GB/해제 32GB 미만, 오프라인 smoke/CRC 통과

공식 제출은 `v44 -> v47 -> 개발 전용 WPT Voice 5%`처럼 한 단계씩 진행해 hidden
기여도를 분리한다. 이 세 후보의 공식 상승을 확인하기 전에는 로컬 `.82`를 SoTA로
표현하지 않는다. 1위를 실제로 넘으려면 안전 residual 다음에 Music OOD와
mixture-native 학습에서 최소 약 `0.085` ADS를 추가로 확보해야 한다.

## 현재 정확한 SoTA 기준과 3-stream 실험

현재 리더보드 1위는 Total/ADS/CPS
`0.837980 / 0.821160 / 0.989320`이다. 우리 CPS를 `0.9893079`로 유지한다고
가정하면 동률 Total에 필요한 ADS는 `0.8211658`이다. v18 ADS
`0.7363413`에서 weighted EER를 `0.2636587 -> 0.1788342` 아래로 줄여야 하며,
이는 약 32.2% 상대 감소다. 따라서 기존 모델 확률을 5~30% 바꾸는 블렌드만으로는
목표에 도달할 수 없다.

첫 3-stream matrix는 점수를 보기 전에
`configs/three_stream_experiment_v1.yaml`로 고정했다. exact-v47을 0 residual
identity로 두고 EAT patch graph, SPEAR component bins, 원본 mixture의 ordered
XLS-R window를 task-query로 읽는다. RR/RF/FR/FF rank, 동일 mixture의
clean-to-codec logit/latent consistency, dataset×layout×channel soft group-DRO를
사용한다. residual bound 1/2, channel penalty 강도, train-core/all, XLS-R 제거
ablation을 따로 둬 어느 요소가 기여했는지 분리한다.

두 번째 matrix도 첫 결과를 읽기 전에
`configs/three_stream_experiment_v2_disentangle.yaml`로 고정했다. 여기에는 혼합
상태의 직접 병목을 겨냥한 두 목적함수가 추가된다.

- component invariance: 같은 Voice를 공유하는 RR↔RF와 FR↔FF의 Voice
  logit/latent, 같은 Music을 공유하는 RR↔FR와 RF↔FF의 Music logit/latent를
  일치시킨다. File은 상대 성분의 fake 여부에 따라 label이 실제로 변하므로 이
  불변성에서 제외한다.
- batch AUC ranking: 각 batch의 양성/음성 모든 쌍 순서를 task별로 직접
  최적화한다. component가 없는 행은 대회의 Voice/Music EER와 똑같이 mask하고
  공식 task 비중 `0.2/0.3/0.5`를 유지한다.

v4에서 전체 ADS뿐 아니라 RF의 Music EER, FR의 Voice EER, clean/G.722 보존을
동시에 통과해야 한다. 점수가 좋은 한 domain만 보고 채택하거나, 하락한 v19/v32/
v33처럼 hard router와 큰 residual을 다시 제출하지 않는다.

### exact-v47 캐시 실행기 결함과 교정

26,681행 4-shard 첫 실행에서 base v47 예측과 원본 XLS-R 추가 pass는 완료됐지만,
후속 EAT fusion이 전체 corpus를 열어 shard CSV 외 ID를 extra로 보고 중단됐다.
이는 base pipeline만 `--num-shards`를 알고 후속 fusion들은 모른다는 실행기
결함이다. schema v2에서는 shard마다 round-robin audio와 sample CSV를 물리적으로
격리하고 내부 실행은 항상 shard `0/1`로 고정한다.

실패 실행의 base 산출물은 예외 위치를 무시하고 사용하지 않았다. 네 shard 모두
동일 package/data provenance, 정확히 하나의 알려진 ID-mismatch traceback,
`CACHE_EXPORT_TIMING`에 기록된 prediction SHA-256과 실제 CSV hash 일치,
EAT/SPEAR 미생성이라는 조건을 모두 다시 검증한 뒤에만 schema-v2 작업공간에
채택했다. 후반 fusion은 base/XLS-R 재계산 없이 격리된 파일 집합에서 재개한다.
전체 회귀 테스트는 이 변경과 새 loss를 포함해 228개가 통과했다.

## v50 frozen one-shot: blind v6

blind v6의 어떤 진위 예측이나 점수도 만들기 전에 v50을 고정했다. v50은 exact
v47에서 두 축만 바꾼다.

- Voice: 기존 XLS-R pass를 재사용하는 Spectra 10% + sparse call-consistency 60%
- File: 원본 오디오 WPT seed06, 균등 5-view, LME T=2, logit 40%
- Music 및 Voice/Music Presence: v47과 bit-exact

48개의 source-disjoint base를 다섯 channel로 렌더링한 240개 blind v6에 v47과
v50을 먼저 모두 추론한 다음 정답을 한 번만 열었다.

| 모델 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| exact v47 | 0.300000 | 0.158333 | 0.258333 | 0.740833 |
| frozen v50 | **0.233333** | **0.083333** | 0.258333 | **0.789167** |

ADS `+0.048333`이고 두 신규 축 모두 독립 bank로 일반화했다. layout별 ADS도
concurrent `0.7267 -> 0.7817`, partial overlap `0.6492 -> 0.7642`, sequential
`0.8550 -> 0.9200`으로 모두 상승했다. Music과 두 Presence 열은 실제 CSV에서도
최대 절대차 `0`이었다.

48개 base를 단위로 channel pair를 함께 재표집한 1,000회 paired bootstrap에서
ADS 차이의 95% 구간은 `[+0.0211, +0.1084]`였고 상승 비율은 `100%`였다. 작은
bank의 EER 격자 때문에 구간은 넓지만, 개선 방향이 우연한 한두 channel row에만
의존하지 않는다는 근거다.

다만 현재 CPS를 유지한 채 1위에 필요한 ADS `0.8211658`에는 아직 `0.031999`가
부족하다. v50의 File/Voice EER를 그대로 둘 경우 필요한 Music EER은
`0.15167` 이하이다. blind v6에서 Music EER은 clean `0.1667`, G.711 `0.2083`,
G.722 `0.2083`, Opus-NB `0.3750`, 이중 transcode `0.3333`이므로 다음 병목은
channel-invariant Music이다. 특히 concurrent Music EER `0.325`와 partial
overlap `0.250`을 줄여야 한다.

채점 직후 blind v6는 `retrospective_diagnostic`으로 퇴역했다. 이후 후보의
가중치, checkpoint, router, stopping point 선택에는 사용할 수 없다. Music/File
후보는 development와 이미 퇴역한 blind v5에서만 선정하고, 최종안은 새로 예약한
blind v7에서 다시 한 번만 검증한다.

## v52 frozen one-shot: blind v7

v4와 퇴역 v5에서 모두 개선된 Music-only 3-head residual 3.75%를 v50에 추가하고,
최종 Voice/Music 중 큰 확률과 File을 logit 20%로 결합하는 v52를 동결했다. v7은
72 base × 5 paired channel, 총 360개이며 기존 83개 truth의 source/speaker/song/hash와
교집합 0인 상태로 예약·의미 검사·렌더·잠금했다.

| 모델 | File EER | Voice EER | Music EER | ADS |
|---|---:|---:|---:|---:|
| frozen v50 | **0.211111** | 0.200000 | 0.305556 | **0.762778** |
| frozen v52 | 0.212963 | 0.200000 | 0.305556 | 0.761852 |

v52는 Music EER를 움직이지 못했고 ADS가 `-0.000926` 하락했다. clean에서는 ADS가
`0.83611 -> 0.87222`로 좋아졌지만 Opus-NB는 `0.67130 -> 0.65833`으로 악화했고,
이중 transcode만 `0.63796 -> 0.65463`으로 일부 회복했다. 즉 작은 original-mixture
residual은 clean/source-known proxy에는 유효해도 codec과 generator가 함께 바뀌는
조건에서 안정적이지 않다.

v7은 채점 직후 퇴역했으며 이후 선택에 쓰지 않는다. 다음 후보는 얕은 residual
weight 탐색이 아니라 generator-balanced leave-one-generator-out 학습과 paired
clean↔codec consistency를 결합한 원본-audio Voice/Music expert로 전환하고, 별도
blind v8에서 다시 한 번만 확인한다.
