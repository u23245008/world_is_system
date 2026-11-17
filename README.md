# world_is_system

신호 및 시스템 2조(상하이 조) — 전통적 신호 처리 기반 음성 잡음 제거

이 브랜치의 기본 알고리즘은 **raw3**입니다. 기존 gentle을 기반으로 원본 Noisy의 잡음 신뢰도에 따른 대역 이득 보정을 추가했습니다. 딥러닝·머신러닝 모델과 학습 가중치를 사용하지 않습니다. 실제 처리에는 각 Noisy WAV만 사용하며 Clean은 평가에만 사용합니다.

## 설치 및 실행

Python 3.13에서 검증한 환경입니다. 저장소 폴더에서 실행합니다.

```powershell
python -m pip install -r requirements.txt
python denoise.py --input input --output output
```

위 명령과 `--method raw3` 지정은 같습니다. input 바로 아래 모든 WAV를 처리해 output에 동일한 파일명으로 저장합니다. 샘플링 주파수·채널 수·샘플 수·WAV subtype을 유지합니다. 개인 PC 절대 경로에 의존하지 않습니다. 2048 샘플 미만 또는 무음은 그대로 출력합니다.

기존 gentle 비교 실행:

```powershell
python denoise.py --method gentle --input input --output output_gentle
```

## Validation 결과

164개 전체의 저장 WAV를 다시 읽어 계산한 파일별 산술평균, 16 kHz PESQ-WB입니다. [파일별 결과](validation_results.csv)에서 확인할 수 있습니다.

| 신호 | 평균 PESQ-WB | 평균 SNR(dB) | STOI |
|---|---:|---:|---:|
| Noisy 입력 | 1.39467 | 9.63415 | 0.86107 |
| 기존 gentle | 2.37781 | 11.72034 | 0.90033 |
| **기본 raw3** | **2.43540** | **10.10892** | **0.89912** |

PESQ를 우선한 선택으로 SNR은 gentle보다 내려갔습니다. 추가 Train96에서는 raw3 PESQ 2.29043 / SNR 8.87179 / STOI 0.86299였습니다. 이 집합은 개발 과정에서 비교에 사용한 공개 Train 표본이며 히든 테스트가 아닙니다. 평균 SNR이며 개별 파일의 하한을 보장하지 않습니다. 히든 테스트 결과는 아직 확인하지 않았습니다.

SNR은 원본 Clean과 실제 저장 결과의 오차로 계산합니다. Clean에 맞춰 출력 이득을 조절하거나 평가 전에 출력 크기를 맞추지 않습니다. 후처리의 RMS 기준은 같은 Noisy에서 얻은 gentle 출력입니다.

## 알고리즘

STFT 기반 SPP 잡음 추정 및 OM-LSA 형태 이득 → 음성 신뢰도에 따른 저역 복원 → 하모닉 필터 → 웨이블릿 축소 → 정방향/역방향 처리와 분석 위치 4개 평균 → Noisy 잡음 PSD와 국소 전력에 따른 최대 3 dB 대역 보정 순서입니다. 후처리는 전체 에너지를 gentle 기준으로 맞추고 피크 여유를 둡니다. 기본 대역 설정은 gentle과 같습니다. 후처리에 따라 음색과 대역별 이득은 변합니다.

입력별 파라미터는 Noisy만으로 계산합니다. 파일명·정답·잡음 분류 라벨을 추론에 사용하지 않습니다. 새 신뢰도 마스크는 자체 실험이며 특정 논문의 완전한 재현이라고 주장하지 않습니다. 원리·확인한 자료·한계는 [실험 기록](results/experiment_notes.md)과 [gentle 기반 문헌 기록](results/gentle_baseline_notes.md)에 있습니다. 두 문서는 로컬 개발 당시의 기록이며 현재 기본 실행 방법은 이 README를 따릅니다.

## 평가 재현

공개 데이터가 저장소 상위 폴더의 validation/에 있을 때:

```powershell
python evaluate.py
```

다른 위치라면 train/·validation/을 포함하는 데이터 루트를 지정합니다.

```powershell
python evaluate.py --data-root DATA_ROOT
```

추가 Train96 목록의 재평가:

```powershell
python evaluate.py --split train --manifest results/additional_train96_manifest.csv
```

기본 전체 평가 결과는 validation_results.csv에 저장됩니다. 평가에는 Clean과 Noisy가 모두 필요하지만 실제 잡음 제거 실행에는 Clean이 필요하지 않습니다. 개발 재현 환경은 OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1입니다.

## 검증 및 청취

```powershell
python verify_io.py
```

기본 raw3 CLI로 형식 검사와 포함된 실제 음성 3개 결과의 일치를 확인합니다.

- 모노·스테레오·44.1 kHz·짧은 입력·무음 형식 검사: [기록](results/io_checks.json)
- 실제 음성 3구간×5방법의 저장 파형 일치 검사: [기록](results/waveform_equivalence.json)
- [같은 음성 3구간의 Noisy·gentle·raw3 WAV](listening_examples/raw3/)
- [추가 Train96 파일별 결과](results/additional_train96.csv)

첫 비교 구간에서 raw3이 선호되었습니다. 전체 파일의 주관적 청취 검증을 모두 마쳤다는 뜻은 아닙니다. 지표 상승만으로 모든 음질 문제가 해결됐다고 단정하지 않습니다.

## 제출 버전과 팀 확인 사항

브랜치: feature/raw3-pesq-2.4354. 제출 버전은 GitHub에 표시되는 해당 브랜치의 커밋 해시로 지정합니다. main 병합은 별도입니다.

저장소 루트의 기존 AI 활용 내역서 초안, sample_archives/의 ZIP, listening_examples/의 상위 WAV는 **이전 균형 후보 자료**입니다. 현재 raw3 제출 자료와 혼동하지 마세요. 이번 회차의 AI 활용 내역서·제출 회차·팀 서명 및 최종 청취/스펙트로그램 고찰은 제출 전에 팀이 갱신해야 합니다. 현재 AI 초안이 raw3에 맞춰 작성됐다고 주장하지 않습니다.
