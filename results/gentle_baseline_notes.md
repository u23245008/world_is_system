> 이전 gentle 개발 기록입니다. 현재 브랜치의 기본값은 raw3이며 실행 방법은 저장소 루트 README를 따릅니다.

# PESQ와 SNR 동시 개선 — 로컬 검토 후보

GitHub와 기존 선호 2.3464 버전은 수정하지 않았습니다. 기본 실행 후보는 gentle입니다. 새 후보의 직접 청취 확인은 보류 상태입니다. 이 문서는 정식 AI 활용 내역서가 아닙니다.

## 결과

동일 Validation 164개 전체, 저장한 WAV를 다시 읽어 파일별 점수를 산술평균했습니다. 16 kHz PESQ-WB입니다.

| 버전 | PESQ-WB | SNR(dB) | STOI |
|---|---:|---:|---:|
| 기존 선호 cycle4 | 2.346394 | 11.427270 | 0.899511 |
| gentle (기본) | 2.377806 | 11.720343 | 0.900334 |
| snr (SNR 우선) | 2.371214 | 11.831837 | 0.900376 |

추가 Train96 비교:

| 버전 | PESQ-WB | SNR(dB) | STOI |
|---|---:|---:|---:|
| 기존 선호 | 2.202841 | 10.162997 | 0.864209 |
| gentle | 2.223545 | 10.292039 | 0.864237 |
| snr | 2.216921 | 10.395050 | 0.864319 |

이 96개는 초기 선정 시 이전 Train 점수 CSV에 기록된 170개 ID를 제외하고 골랐습니다. 이번 중간 후보 비교에도 사용했으므로 히든 테스트나 최종 독립 검증이라고 부르면 안 됩니다. 고정 목록은 research_results/joint_growth_reserved_train96.csv입니다. 평균 SNR이며 개별 파일이 모두 10 이상이라는 뜻은 아닙니다. PESQ 2.4는 아직 달성하지 못했습니다.

joint 옵션은 Validation 2.385204 / 11.547423이지만 추가 Train96의 SNR이 기존 10.162997에서 10.145797로 소폭 낮아져 기본값에서 제외했습니다. PESQ 2.396710 / SNR 11.042482 후보도 기존 SNR보다 낮아 제외했습니다.

## 알고리즘 변경

기존 양방향 SPP/OM-LSA 형태 이득, 저역 복원, 하모닉 필터, 웨이블릿 축소 및 분석 위치 4개 평균을 유지합니다.

1. Noisy의 긴 FFT에서 좁고 지속적인 스펙트럼 선 후보를 찾습니다. 12개 구간의 복소 진폭 일관성 0.65 이상, 진폭 지속성 0.4 이상인 성분만 고정 정현파 기저로 최소제곱 투영해 뺍니다. 대비 16 dB, 최대 12개, 80 Hz~min(6000 Hz, 0.45*sr) 규칙입니다.
2. 처리 중 감쇠된 말소리 크기를 Noisy만으로 계산한 기존 OM-LSA 기준 출력의 에너지와 혼합해 복원합니다(level_mix .5→.75). 정답 음성으로 출력 이득을 맞추지 않습니다. SNR 개선에는 이 크기 복원 효과가 포함되어 있어 전부 잡음 자체 감소라고 해석하면 안 됩니다.
3. 웨이블릿 인접 대역 결합 비중을 1→0.5로 조정합니다.
4. gentle은 마지막에 500 Hz 중심, 폭 1옥타브의 부드러운 0.25 dB 보정을 합니다. snr 옵션은 이 보정을 생략합니다. 기존 메인 대역 설정 2250 Hz 저역통과/고역 혼합 .15는 유지합니다. 기존부터 고역 감쇠가 있으며 완전한 전 대역 보존을 뜻하지 않습니다.

학습 모델이나 학습 가중치, Clean을 이용한 추론, 파일명별 규칙을 사용하지 않습니다. 정현파의 주파수·위상·진폭은 각 Noisy 파일에서 추정하는 전통적 신호 처리 파라미터입니다. FFT/최소제곱 투영은 외부 코드 복사 없이 직접 구현했습니다.

## 검증과 제한

- 실제 저장 WAV와 Clean의 차이로 SNR을 계산합니다. Clean 기반 재스케일링 없이 SNR=10log10(sum(clean²)/sum((clean-output)²))입니다. SI-SDR 보조 계산의 투영 계수는 평가 전용이며 출력/SNR에 적용하지 않습니다.
- Validation164의 Clean만 간섭 제거기에 넣은 대조 실험에서 채택한 0.65/16 dB 규칙은 원음을 변경하지 않았습니다. 이것이 모든 새로운 음성에서 오검출이 없음을 보장하지는 않습니다. 지나치게 느슨한 8 dB 규칙은 원음 1개에서 성분이 제거되어 제외했습니다.
- 추가 Train96에서 계측한 기존/SNR 우선 처리의 크기 복원 및 웨이블릿 복원 단계에서 full-scale 초과 샘플은 0개, 해당 단계 최대 피크는 0.922404였습니다. gentle 추가 평가에서도 동일했습니다. 최종 gentle Validation WAV의 한계값 샘플 수는 0입니다.
- 모노·스테레오·44.1 kHz·짧은 입력·무음 형식 검사와 실제 제공 음성 3세트 출력을 확인했습니다. io_checks.json 및 smoke_clipping_audit.json을 참고하세요. 형식 검사는 해당 샘플링 주파수의 음질 보장을 의미하지 않습니다.
- 청취로 자연스러움과 금속성/물소리 잔여 잡음을 확인하는 단계는 남아 있습니다. 클리핑 없이 지표가 올랐다는 사실만으로 모든 음질 문제가 해결됐다고 주장하지 않습니다.

## 실행

Python 3.13, requirements.txt에 기록한 설치 환경에서 실행 검증했습니다. 이 폴더에서:

    python -m pip install -r requirements.txt
    python denoise.py --input input --output output

기본값은 gentle입니다. SNR 우선은 --method snr, 기존 비교 버전은 --method previous입니다. input 바로 아래 모든 WAV를 처리해 같은 이름으로 output에 저장합니다. 샘플링 주파수·채널·샘플 수를 유지하며 짧은 입력/무음은 그대로 출력합니다.

상위 폴더에 원본 validation/ 및 train/이 있을 때 재평가:

    python evaluate.py
    python evaluate.py --method snr

실험 환경: OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. validation_gentle.csv가 고정 후보의 재평가 결과입니다. 연구용 실행 파일은 research_results/REPRODUCTION.txt를 참고하세요. 추가 96개 재현에는 고정 manifest를 사용하고 표본 예약 스크립트를 다시 실행해 목록을 바꾸지 마세요.

## 실제 청취 자료

listening_examples/에 제공 데이터의 같은 3구간을 담았습니다. *_noisy.wav=원본 잡음 음성, *_clean.wav=정답, *_previous_23464.wav=기존 선호, *_gentle.wav=새 기본 후보, *_snr.wav=SNR 우선, *_joint.wav=더 강한 보정 실험입니다. 생성한 예시 음성이 아닙니다.

## 실제 확인한 문헌

- B. Widrow 등 (1975), Adaptive Noise Cancelling: Principles and Applications, DOI 10.1109/PROC.1975.10036. https://isl.stanford.edu/~widrow/papers/j1975adaptivenoise.pdf — 특히 외부 기준 입력 없이 주기 간섭을 제거하는 VIII.F절의 개념을 확인했습니다. 우리 방법은 긴 FFT/지속성 판정/고정 정현파 투영의 자체 변형이며 논문의 LMS 구현을 그대로 복제한 것은 아닙니다.
- D. L. Donoho, I. M. Johnstone, Adapting to Unknown Smoothness via Wavelet Shrinkage (1994 기술보고서, 1995 논문), https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/ausws.pdf — 2.3절 식(6),(7)의 SURE 임계값 선택 원리를 확인하고 시험했습니다. 잔여 SWT 계수에 대한 자체 혼합 변형은 PESQ가 낮아져 최종 후보에 사용하지 않았습니다. 원문의 iid Gaussian 위험 보장을 이 파이프라인에 그대로 주장하지 않습니다.
- SciPy find_peaks 공식 문서: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html — 스펙트럼 피크의 높이·prominence·거리 조건 사용법 확인.

기존 SPP/OM-LSA/YIN/SWT 방법의 문헌과 출처는 이전 제출 초안 및 원본 코드의 참고 기록도 함께 유지해야 합니다. 실험 평균은 experiment_means.csv, 파일별 결과와 연구 코드는 research_results/에 있습니다. 중단된 시험은 aborted로 표시했고 전체 평균으로 쓰지 않습니다. 원본 데이터셋 전체와 외부 바이너리는 포함하지 않았습니다.
