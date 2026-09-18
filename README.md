# FULX_Test

FLUX.2 [klein] 4B 로컬 구동 테스트.

**타깃 환경:** Windows 11 / RTX 4070 12GB / CUDA 13.1 driver / Python 3.12

## 모델 구성

`Flux2KleinPipeline` — 전부 bf16, 다운로드 약 16 GB.

| 컴포넌트 | 클래스 | 크기 |
|---|---|---|
| transformer | `Flux2Transformer2DModel` (~3.9B) | 7.75 GB |
| text_encoder | `Qwen3ForCausalLM` (Qwen3-4B) | 8.05 GB |
| vae | `AutoencoderKLFlux2` | 0.17 GB |

step-distilled 모델이라 **4 steps**, guidance는 파이프라인이 무시합니다.
Apache 2.0, gated 아님 (HF 토큰 불필요).

## 12GB에서의 제약

가중치 합계가 15.97 GB라 bf16 전체 상주는 불가능합니다. BFL 공식 수치는
`enable_model_cpu_offload()` 기준 **~13 GB peak**인데, Windows가 데스크톱용으로
VRAM 일부를 선점하므로 12GB 카드의 실사용량은 약 10.5~11.4 GB입니다.
→ **양자화가 사실상 필수.**

## 셋업

```powershell
uv sync
```

`torch`/`torchvision`는 PyPI가 아니라 cu130 휠 인덱스에서 받습니다
(PyPI 휠은 Windows에서 CPU-only). `pyproject.toml`의 `[tool.uv.sources]` 참고.

가중치 미리 받아두기:

```powershell
uv run main.py --download-only
```

### 모델 저장 위치

기본값은 사용자 프로필이 아니라 **프로젝트 안의 `./models/`** 입니다.
`main.py`가 HF 라이브러리를 import하기 전에 `HF_HOME`을 거기로 세팅합니다
(`huggingface_hub`는 캐시 경로를 import 시점에 module-level로 확정하므로
그 전에 세팅해야 합니다 — 그래서 스크립트의 HF import가 전부 함수 안에 있습니다).

```
models/
  hub/      <- 모델 가중치 (~16 GB)
  xet/      <- Xet chunk 캐시 (추가 용량 차지)
  assets/
```

`HF_HUB_CACHE`가 아니라 `HF_HOME`을 쓰는 이유는, 전자는 `hub/`만 옮기고
Xet chunk 캐시는 사용자 프로필에 그대로 남기 때문입니다.

다른 경로(예: 큰 드라이브)로 빼려면:

```powershell
uv run main.py --cache-dir D:\models
```

`models/`는 `.gitignore`에 들어가 있습니다.

## 실행

```powershell
uv run main.py                     # te4bit (기본) — 768x768
uv run main.py --mode both4bit     # 여유 필요할 때
uv run main.py --size 1024 --prompt "..."
```

### 모드

| 모드 | 구성 | 예상 peak | 비고 |
|---|---|---|---|
| `te4bit` | text_encoder NF4 + transformer bf16 | ~10.5 GB | **기본.** 전부 GPU 상주, offload 없어서 빠름 |
| `both4bit` | 둘 다 NF4 | ~6.5 GB | 여유 최대, transformer 화질 저하 체감됨 |
| `bf16offload` | 양자화 없음 + model offload | ~11.5 GB | BFL 레퍼런스 구성. 12GB에선 OOM 예상. 시스템 RAM 32GB 필요 |
| `sequential` | 레이어 단위 스트리밍 | ~3.5 GB | 어디서든 돌지만 매우 느림 |

peak 수치는 768x768 기준 추정치입니다. 스크립트가 `max_memory_allocated` /
`max_memory_reserved` / `nvidia-smi` 실측을 같이 출력하니 그걸로 판단하세요.

## Windows 함정

1. **Sysmem Fallback** — VRAM 초과 시 드라이버가 OOM 대신 시스템 RAM으로 조용히
   넘겨서 10~50배 느려집니다. NVIDIA 제어판 → 3D 설정 관리 → `python.exe`에
   "Prefer No Sysmem Fallback"을 걸어두면 OOM이 제대로 터져 원인 파악이 됩니다.
2. **flash-attn 설치하지 말 것** — Windows 휠이 없습니다. PyTorch SDPA 폴백으로 충분.
3. **디스크** — `./models/`에 16 GB + Xet 캐시분이 쌓입니다. 프리플라이트가
   여유 공간을 확인해서 부족하면 경고합니다.
