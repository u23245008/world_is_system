# 신호및시스템 AI 활용 내역서 — 균형 후보 초안

> 원본 HWPX 양식의 항목에 맞춘 작성 초안입니다. **[팀 작성]** 표시는 팀에서 사실을 확인한 뒤 채워야 합니다. 이 문서 자체를 서명된 최종 내역서로 제출하지 마세요.

## 기본 정보

- 소속: 연세대학교 미래캠퍼스 **[학과/학부/전공 — 팀 작성]**
- 조 및 조원: **[조 번호·팀명·조원 — 팀 작성]**
- 제출 회차: **[팀 작성]**
- 프로젝트명: 내 목소리를 구해줘! (음성 잡음 제거 챌린지)

## Part 1. AI 활용 개요

- 활용 AI: OpenAI Codex (GPT-6)
- 활용 목적: 제출 코드 초안 작성, STFT 및 Wiener 방식 탐색, 실행 오류 수정, Validation 평가 및 파라미터 비교, AI 활용 내역서 초안 작성
- 사용하지 않은 것: 딥러닝 및 머신러닝 모델

## Part 2. 프롬프트 및 검증 로그

### 사례 1 — 제출용 코드와 평가 코드 작성

- 질문 요약: “너가 직접할 수 있어?” 제공된 Train/Validation 데이터와 기본 노트북을 바탕으로 제출 형식에 맞는 음성 잡음 제거 코드와 평가 코드를 만들어 달라고 요청함.
- AI 답변 요약: STFT에서 Noisy 신호의 잡음 전력을 추정하고 Wiener 형태의 이득을 곱하는 `denoise.py`, WAV 입출력 코드, `evaluate.py`, `requirements.txt`, README 초안을 작성함.
- 검증 및 수정: Validation 164개에 실행하여 실제 저장된 WAV로 PESQ-WB와 SNR을 계산함. 출력의 샘플링 주파수·채널 수·샘플 수가 입력과 같은지도 확인함. Clean 정답은 평가 코드에서만 사용하고 잡음 제거 함수에는 전달하지 않음. 초기 제출 후보는 PESQ-WB 1.5363, SNR 11.8121 dB였음.

### 사례 2 — 전통적 방법 비교와 균형 후보 선정

- 질문 요약: 딥러닝·머신러닝을 제외하고 PESQ와 SNR을 함께 높이는 설정을 시험해 달라고 요청함. 이후 PESQ-WB 1.9 이상, SNR 11 dB 이상을 목표로 지정함.
- AI 답변 요약: 주파수별 하위 전력 추정, 조용한 프레임 선택, 스펙트럼 차감, Wiener 형태의 이득, MMSE 계열 추정기, 대역 필터와 이득 평활화 등 여러 전통적 신호 처리 후보를 비교함. 최종 균형 후보는 122 Hz 고역 통과 결과를 63% 혼합하고, 1536샘플 STFT·조용한 프레임 10%·잡음 전력 1.23배·프레임 간 평활화 0.91·최소 이득 0.10·시간 및 주파수 3칸 평활화를 적용함.
- 검증 및 수정: 최종 제출 코드로 Validation 164개 전체를 다시 실행함. 평균 PESQ-WB는 Noisy 1.3947 → 출력 1.9022, 평균 SNR은 9.6341 → 11.0618 dB. 초기 후보보다 PESQ는 높지만 SNR은 낮으므로 음성 보존 여부를 청취로 확인해야 함. 히든 테스트 성능은 아직 알 수 없음.

## 참고 문헌 및 코드 출처

1. Steven F. Boll, “Suppression of Acoustic Noise in Speech Using Spectral Subtraction,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(2), 113–120, 1979. DOI: [10.1109/TASSP.1979.1163209](https://doi.org/10.1109/TASSP.1979.1163209). 스펙트럼 차감 비교 방법의 원리를 확인함. 최종 후보는 스펙트럼 차감을 사용하지 않음.
2. Yariv Ephraim and David Malah, “Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 32(6), 1109–1121, 1984. DOI: [10.1109/TASSP.1984.1164453](https://doi.org/10.1109/TASSP.1984.1164453). 단시간 스펙트럼의 음성 향상과 Wiener 계열 비교를 확인함. 최종 후보는 논문의 MMSE-STSA 추정기 자체를 구현한 것은 아님.
3. SciPy Developers, [`scipy.signal.stft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html) 및 [`scipy.signal.istft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html), SciPy 문서. STFT·역 STFT 및 중첩 복원 조건을 확인함.
4. 수업에서 제공된 `noise_reduction_baselines.ipynb`의 잡음 전력 추정 및 Wiener 필터 예시를 참고함. 제출 코드는 별도로 구성했으며 외부 코드를 그대로 복사하지 않았음.

## Part 3. 종합 고찰

Validation에서 PESQ-WB 1.9와 SNR 11 dB를 동시에 넘는 설정을 찾았으나, 여러 설정을 같은 Validation에서 비교했으므로 히든 테스트에서는 결과가 달라질 수 있다. 고역 통과 필터가 음성의 낮은 주파수 성분도 줄일 수 있으므로 점수만으로 음성 품질을 판단하지 않는다.

- 청취에서 확인한 잡음 감소·음성 왜곡: **[팀 작성 — 아직 수행하지 않음]**
- 스펙트로그램에서 확인한 변화: **[팀 작성 — 아직 수행하지 않음]**
- 남은 한계와 다음 회차 개선 방향: **[팀 작성]**
- 조 번호·조원 서명: **[팀 작성]**
