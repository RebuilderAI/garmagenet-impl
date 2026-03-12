# GarmageNet 설치 가이드 (Detailed)

> README.md의 간단한 설치 안내로는 환경에 따라 빌드 실패가 발생할 수 있습니다.
> 이 문서는 실제 설치 과정에서 만난 이슈와 해결 방법을 포함한 **재현 가능한 설치 가이드**입니다.

## 테스트된 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| CUDA | 12.4 |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| GPU | NVIDIA (CUDA 지원) |

---

## 1단계: 시스템 패키지 설치 (root 권한 필요)

`chamferdist` 같은 CUDA C++ 익스텐션 패키지를 빌드하려면 **Python 개발 헤더**가 필요합니다.
이것이 없으면 `fatal error: Python.h: No such file or directory` 에러가 발생합니다.

```bash
# Python 개발 헤더 설치 (Python 버전에 맞게)
sudo apt-get update
sudo apt-get install -y python3.10-dev

# (선택) ninja 빌드 시스템 - CUDA 익스텐션 컴파일 속도 향상
# pip으로도 설치 가능 (아래 참고)
```

> **⚠️ 이 단계를 건너뛰면 chamferdist 빌드가 반드시 실패합니다.**

---

## 2단계: 프로젝트 클론 및 가상환경 생성

```bash
git clone https://github.com/Style3D/garmagenet-impl.git garmagenet
cd garmagenet

python3.10 -m venv .venv/garmagenet
source .venv/garmagenet/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 3단계: 기본 의존성 설치

```bash
pip install -r requirements.txt
```

> `requirements.txt`는 `diffusers`와 `wandb` 버전을 최신 호환 버전으로 업데이트해 두었습니다. 
> 다만, `chamferdist`와 `nvdiffrast`는 빌드 환경이 필요하여 여전히 주석 처리되어 있습니다.

---

## 4단계: huggingface_hub / diffusers 호환성 수정

`requirements.txt`에 명시된 `diffusers==0.27`은 `huggingface_hub`의 제거된 API(`cached_download`)를
사용하기 때문에, `transformers>=4.50`과 함께 사용하면 import 에러가 발생합니다.

**해결:** diffusers를 0.30.x로 업그레이드합니다.

```bash
pip install "diffusers[torch]>=0.30,<0.31"
```

> 프로젝트에서 사용하는 diffusers API
> (`ConfigMixin`, `BaseOutput`, `ModelMixin`, `Decoder`, `DDPMScheduler` 등)는
> 0.30.x에서도 정상 동작하는 것을 확인했습니다.

---

## 5단계: chamferdist 설치

CUDA C++ 익스텐션을 포함하는 패키지입니다. `ninja`를 먼저 설치하면 빌드 속도가 빨라집니다.

```bash
# ninja 설치 (빌드 속도 향상)
pip install ninja

# chamferdist 빌드 & 설치
pip install chamferdist --no-build-isolation
```

> **`--no-build-isolation`**: 가상환경에 설치된 PyTorch를 빌드 시 직접 참조하기 위해 필요합니다.

### 빌드 실패 시 체크리스트

| 에러 메시지 | 원인 | 해결 |
|-------------|------|------|
| `Python.h: No such file or directory` | python3.10-dev 미설치 | `sudo apt install python3.10-dev` |
| `ninja: not found` (경고) | ninja 미설치 | `pip install ninja` (느린 빌드로 진행은 됨) |
| `nvcc not found` | CUDA toolkit 미설치 | CUDA toolkit 설치 필요 |

---

## 6단계: nvdiffrast 설치 (데이터 전처리용, 선택)

`nvdiffrast`는 **데이터 전처리**(`data_process/process_garmage.py`)에서만 사용됩니다.
이미 전처리된 데이터로 학습만 할 경우 설치하지 않아도 됩니다.

```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

> OpenGL 개발 라이브러리가 필요할 수 있습니다:
> ```bash
> sudo apt-get install -y libgl1-mesa-dev libgles2-mesa-dev libegl1-mesa-dev
> ```

---

## 전체 설치 스크립트 (한 번에 실행)

```bash
#!/bin/bash
set -e

# ===== 시스템 패키지 (root 권한) =====
sudo apt-get update
sudo apt-get install -y python3.10-dev

# ===== 프로젝트 클론 =====
git clone https://github.com/Style3D/garmagenet-impl.git garmagenet
cd garmagenet

# ===== 가상환경 생성 & 활성화 =====
python3.10 -m venv .venv/garmagenet
source .venv/garmagenet/bin/activate
pip install --upgrade pip setuptools wheel

# ===== 의존성 설치 =====
pip install -r requirements.txt

# diffusers 호환성 수정 (huggingface_hub 충돌 해결)
pip install "diffusers[torch]>=0.30,<0.31"

# ninja + chamferdist (CUDA 빌드)
pip install ninja
pip install chamferdist --no-build-isolation

# (선택) nvdiffrast - 데이터 전처리시만 필요
# pip install git+https://github.com/NVlabs/nvdiffrast.git

echo "✅ GarmageNet 설치 완료!"
```

---

## 설치 검증

```bash
source .venv/garmagenet/bin/activate
python -c "
import torch
import diffusers
import transformers
import chamferdist
import wandb
import einops

print(f'torch:        {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'diffusers:    {diffusers.__version__}')
print(f'transformers: {transformers.__version__}')
print('chamferdist:  OK')
print('wandb:        OK')
print('einops:       OK')
print()
print('✅ 모든 핵심 패키지 정상!')
"
```

---

## 요약: README 대비 추가로 필요한 작업

| # | README에 없는 내용 | 왜 필요한가 |
|---|---------------------|-------------|
| 1 | `sudo apt install python3.10-dev` | chamferdist CUDA 빌드에 Python.h 헤더 필요 |
| 2 | `pip install ninja` | CUDA 익스텐션 빌드 속도 향상 (없으면 경고 발생) |
| 3 | `pip install chamferdist --no-build-isolation` | `--no-build-isolation` 플래그 필요 |
| 4 | `pip install "diffusers[torch]>=0.30,<0.31"` | diffusers 0.27과 huggingface_hub 버전 충돌 해결 |
