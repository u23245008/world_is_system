# world_is_system
신호 및 시스템 2조(상하이 조)

## 1차 챌린지: 음성 잡음 제거 균형 후보

**전통적 신호 처리만 사용합니다.** STFT 기반 잡음 추정, Wiener 형태의 이득, 대역 필터와 시간·주파수 평활화를 사용합니다. 실제 잡음 제거에는 Noisy WAV만 입력합니다. Clean은 공개 Validation 평가에서만 사용합니다.

### 설치 및 실행

Python 3.13에서 확인했습니다.

```powershell
python -m pip install -r requirements.txt
python denoise.py --input input --output output
```

`input` 폴더의 모든 WAV마다 `output` 폴더에 같은 이름의 WAV를 저장합니다. 샘플링 주파수, 채널 수, 샘플 수, WAV subtype을 유지합니다. 개인 PC의 절대 경로에 의존하지 않습니다.

### 공개 Validation 결과

164개 파일 평균, 16 kHz PESQ-WB 기준입니다. 파일별 값은 [validation_results.csv](validation_results.csv)에 있습니다.

| 신호 | PESQ-WB | SNR (dB) |
|---|---:|---:|
| Noisy 입력 | 1.3947 | 9.6341 |
| 균형 후보 출력 | **1.9022** | **11.0618** |

`evaluate.py`를 다시 실행하려면 저장소의 상위 폴더에 공개 `validation/samples.csv`, `validation/clean/`, `validation/noisy/`가 있어야 합니다. 평가 코드는 히든 테스트 실행에 필요하지 않습니다.

### 다운로드 및 청취

- [제출 자료 초안 ZIP](sample_archives/균형후보_제출자료_초안.zip)
- [원본·처리 결과 WAV 3쌍 ZIP](sample_archives/청취예시_원본과처리결과.zip)
- [WAV 개별 파일](listening_examples/)

### 제출 전에 팀이 확인할 사항

- 출력 WAV를 직접 듣고 음성 왜곡·잡음 잔류를 확인하고, 스펙트로그램도 비교
- [AI 활용 내역서 HWPX 초안](AI_활용내역서_작성초안.hwpx)의 소속·조원·서명과 청취·스펙트로그램 관찰 작성 ([Markdown 내용 사본](AI_활용내역서_작성초안.md))
- 제출 회차 명시: **[팀 작성]**
- 런어스 비밀글에 이 저장소 링크와 제출 커밋 해시 또는 태그, 완성한 AI 활용 내역서 제출

최종 순위는 히든 테스트셋의 PESQ-WB로 결정됩니다. 공개 Validation 점수와 다를 수 있습니다.
