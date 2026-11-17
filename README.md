# world_is_system

신호 및 시스템 2조 (상하이 조)

## 현재 제출 버전

raw3 기반의 전통적 음성 잡음 제거입니다. STFT 잡음 추정, 파형 복원, 웨이블릿 감쇠와 출력 EQ를 사용합니다. `denoise.py`의 기본 설정은 [검증한 프리셋](raw3_dsp/traditional_pesq_narrow_config.json)을 사용합니다.

## 설치 및 실행

Python 3.12.10에서 확인했습니다.

```sh
python -m pip install -r requirements.txt
python denoise.py --input input --output output
```

`input/`에 잡음이 섞인 WAV를 넣으면 `output/`에 같은 이름으로 처리한 WAV가 저장됩니다. 샘플링 주파수, 채널 수, 샘플 수 및 WAV subtype을 유지합니다.

## 공개 Validation 결과

| 파일 수 | 평균 PESQ-WB | 평균 SNR |
|---:|---:|---:|
| 164 | **2.8259** | **3.0062 dB** |

16 kHz 모노 WAV의 파일별 점수를 산술평균했습니다. [파일별 점수](validation_results.csv)에서 확인할 수 있습니다.

공개 데이터 폴더의 `validation/samples.csv`에 기록된 clean·noisy 쌍으로 평가를 재현할 수 있습니다.

```sh
python -m pip install -r requirements-eval.txt
python evaluate.py --data-root "/path/to/noise_reduction_challenge" --results replay_results.csv
```

이미 처리한 Validation WAV를 채점하려면 평가 명령에 `--output-dir`로 출력 폴더를 추가합니다.
