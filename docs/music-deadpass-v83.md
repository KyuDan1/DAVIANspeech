# v83: Music-only 후보의 불필요한 XLS-R 재실행 제거

v82는 최고 anchor의 모든 계산이 끝난 후 Music 확률만 새 EAT로 대체한다.
기존 v57 Music 보정도 따라서 실행 결과가 버려진다. 이 보정만을 위해 생성하는
XLS-R original-window embedding cache 역시 다른 최종 출력에 쓰이지 않는다.

검증한 변경은 정확히 두 문장이다.

1. `args.xlsr_window_embeddings_output = None`: pipeline 마지막 추가 XLS-R pass 생략.
2. `task_mode="music"`인 `apply_three_stream_anchor_residual` 호출 삭제.

EAT/SPEAR 통계와 다른 보정 계산은 File 등에 필요하므로 유지했다.
모델 가중치, 오디오 입력, 새 Music 분류기, File/Voice/CPS 계산은 변경하지 않았다.

## 실제 실행 결과

동일한 3개 개발 혼합 파일에 reference v82와 fast 버전을 별도 프로세스로 실행했다.
5개 확률 및 ID가 모두 문자열까지 동일했고 최대 확률 차이는 0.0이었다.
reference 83.37초, fast 74.25초였다. 모델 로드 포함, 순차 1회씩 B200 측정이므로
정밀 throughput benchmark나 L4 60분 제한 보증으로 해석하지 않는다.

자동 구조 검사 3개도 통과했다. 변경 marker 불일치 또는 최종 Music 교체보다
뒤의 residual 삭제 요청은 거부한다. 다른 실험의 frozen 소스는 수정하지 않았다.

- 실행/구조 변경: `scripts/verify_music_deadpass_v83.py`
- 구조 테스트: `tests/test_music_deadpass_v83.py`
- 결과: `reports/music_deadpass_v83/report.json`

이 최적화는 점수 향상을 의도한 새 ablation이 아니다. 같은 Music-only 후보의
제출 시간 위험을 줄이기 위한 변경이다.
2026-09-06 UTC에 `DAVIANspeech` 팀으로 공식 API 접수 성공을 확인했다.
채점 완료/점수 개선은 아직 확인되지 않았다.

## ZIP 완료

`best_eat_music_v83.zip` 생성 및 전체 CRC 검증 완료.
9,012,561,340 bytes, 압축 해제 10,425,245,571 bytes, 130개 member.
SHA-256: `92d5f2f3508f7b3d7748f2214a7319c5a9029d826f85779bb8ccb64f18ecf48f`.
v82 대비 script.py만 변경했으며, 다른 member의 CRC/원본 크기/압축 크기는 동일하다.
포함한 script.py는 실제 fast 실행 검증에 사용한 파일과 byte-exact다.
세부 결과: `best_eat_music_v83.report.json`.

제출 시 Music 후보는 v82 대신 이 v83을 사용한다. v82와 v83을 둘 다 제출해
두 슬롯을 쓰려는 계획이 아니다. Voice 진단 파일은 `best_voice_probe_v82.zip` 그대로다.
