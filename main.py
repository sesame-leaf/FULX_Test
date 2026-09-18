"""FLUX.2 [klein] 4B local smoke test.

Target box: Windows 11, RTX 4070 (12 GB), CUDA 13.1 driver. The point of this
script is to find out which memory mode actually fits before you build anything
on top of it, so every run reports peak VRAM and wall time per phase.

    uv run main.py                    # NF4 text encoder + bf16 transformer
    uv run main.py --mode both4bit    # everything NF4, most headroom
    uv run main.py --mode bf16offload # BFL reference config, expected to OOM here
    uv run main.py --download-only    # prefetch ~16 GB of weights first
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = "black-forest-labs/FLUX.2-klein-4B"

# Weights land next to the code instead of C:\Users\<user>\.cache\huggingface.
DEFAULT_CACHE = Path(__file__).resolve().parent / "models"

# Download size of the pipeline components, for the free-disk check.
DOWNLOAD_GB = 16.0


@dataclass(frozen=True)
class Mode:
    key: str
    quantize: tuple[str, ...]
    offload: str | None
    est_peak_gb: float
    note: str


# est_peak_gb is a rough figure at the default 768x768: component weights plus
# typical activations. Treat it as a go/no-go hint, not a measurement.
MODES: dict[str, Mode] = {
    m.key: m
    for m in (
        Mode(
            "te4bit", ("text_encoder",), None, 10.5,
            "NF4 text encoder + bf16 transformer, both resident. Recommended for 12 GB.",
        ),
        Mode(
            "both4bit", ("text_encoder", "transformer"), None, 6.5,
            "Everything NF4. Most headroom; quality loss is visible on the transformer.",
        ),
        Mode(
            "bf16offload", (), "model", 11.5,
            "No quantization, model CPU offload. BFL's reference config. Needs 32 GB system RAM.",
        ),
        Mode(
            "sequential", (), "sequential", 3.5,
            "Per-layer streaming over PCIe. Fits almost anywhere, very slow.",
        ),
    )
}


def setup_cache(cache_dir: Path) -> Path:
    """Point the HF cache at `cache_dir`.

    huggingface_hub resolves HF_HOME at module import time, so this has to run
    before diffusers/transformers pull it in. Everything in this file imports
    those lazily inside functions, which is what makes that possible.
    """
    if "huggingface_hub" in sys.modules:
        print("WARNING: huggingface_hub was already imported; HF_HOME will not take effect.")

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # HF_HOME covers hub/, xet/ and assets/ in one go -- HF_HUB_CACHE would only
    # move hub/ and leave the Xet chunk cache in the user profile.
    os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir


def gpu_info() -> dict[str, str] | None:
    """Query nvidia-smi directly. torch reports total VRAM but not what Windows already took."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    fields = "name,memory.total,memory.used,memory.free,driver_version"
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, IndexError, OSError):
        return None
    return dict(zip(fields.split(","), [v.strip() for v in out.split(",")]))


def preflight(mode: Mode, cache_dir: Path) -> None:
    import diffusers
    import torch
    import transformers

    print("=" * 70)
    print(f"FLUX.2 [klein] 4B  |  mode={mode.key}")
    print(f"  {mode.note}")
    print("=" * 70)
    print(f"  python       {sys.version.split()[0]}")
    print(f"  torch        {torch.__version__}  (built for cuda {torch.version.cuda})")
    print(f"  diffusers    {diffusers.__version__}")
    print(f"  transformers {transformers.__version__}")

    try:
        import bitsandbytes
        print(f"  bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        tail = "  <- REQUIRED for this mode" if mode.quantize else ""
        print(f"  bitsandbytes MISSING{tail}")

    # Read this back from huggingface_hub rather than trusting the env var, so a
    # cache that did not actually move is visible here instead of at download time.
    from huggingface_hub import constants as hf_constants

    hub_cache = Path(hf_constants.HF_HUB_CACHE)
    print(f"\n  hf cache     {hub_cache}")
    if cache_dir not in hub_cache.parents and hub_cache != cache_dir:
        print(f"  WARNING: expected the cache under {cache_dir}")

    free_gb = shutil.disk_usage(cache_dir).free / 2**30
    have = sum(1 for _ in hub_cache.glob(f"models--{REPO.replace('/', '--')}/**/*.safetensors"))
    print(f"  disk         {free_gb:.1f} GiB free"
          + (f"  ({have} safetensors already cached)" if have else f"  (need ~{DOWNLOAD_GB:.0f} GB)"))
    if not have and free_gb < DOWNLOAD_GB:
        print(f"  WARNING: under {DOWNLOAD_GB:.0f} GB free; the download will fail partway.")

    if not torch.cuda.is_available():
        sys.exit(
            "\nERROR: torch cannot see a CUDA device. The usual cause is the CPU-only "
            "wheel from PyPI instead of the cu130 index -- check [tool.uv.sources] in "
            "pyproject.toml, then `uv sync --reinstall-package torch`."
        )

    major, minor = torch.cuda.get_device_capability()
    print(f"\n  gpu          {torch.cuda.get_device_name(0)}  (sm_{major}{minor})")
    if major < 8:
        print("  WARNING: bf16 needs Ampere (sm_80) or newer. This GPU will be slow or fail.")

    info = gpu_info()
    if info:
        total = int(info["memory.total"]) / 1024
        used = int(info["memory.used"]) / 1024
        free = int(info["memory.free"]) / 1024
        print(f"  driver       {info['driver_version']}")
        print(f"  vram         {total:.2f} GiB total / {used:.2f} used / {free:.2f} free")
        print(f"\n  rough peak estimate for this mode: ~{mode.est_peak_gb:.1f} GB")
        if mode.est_peak_gb > free:
            print(f"  >>> Over the {free:.2f} GiB currently free. Expect an OOM.")
    print()


def build_quant_config(mode: Mode):
    if not mode.quantize:
        return None

    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from diffusers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBnb

    nf4 = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # text_encoder is a transformers model and transformer is a diffusers model,
    # so each needs its own library's config class.
    mapping = {}
    if "text_encoder" in mode.quantize:
        mapping["text_encoder"] = TransformersBnb(**nf4)
    if "transformer" in mode.quantize:
        mapping["transformer"] = DiffusersBnb(**nf4)
    return PipelineQuantizationConfig(quant_mapping=mapping)


def load_pipeline(mode: Mode, args) -> tuple[object, float]:
    import torch
    from diffusers import Flux2KleinPipeline

    kwargs: dict = {"torch_dtype": torch.bfloat16, "local_files_only": args.offline}
    quant = build_quant_config(mode)
    if quant is not None:
        kwargs["quantization_config"] = quant

    print(f"[load] {REPO}")
    started = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(REPO, **kwargs)

    if mode.offload == "model":
        pipe.enable_model_cpu_offload()
    elif mode.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    # Tiled VAE decode stops the decode step from spiking peak VRAM.
    if args.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    elapsed = time.perf_counter() - started
    print(f"[load] {elapsed:.1f}s  |  peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return pipe, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=list(MODES), default="te4bit")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--size", type=int, default=768, help="square side in px (default 768)")
    # klein is step-distilled: 4 steps, and the pipeline ignores guidance_scale.
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out.png"))
    parser.add_argument("--offline", action="store_true", help="use only the local HF cache")
    parser.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false")
    parser.add_argument("--download-only", action="store_true", help="fetch weights (~16 GB) and exit")
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE,
        help=f"where model weights live (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args()

    # Must happen before anything imports huggingface_hub.
    cache_dir = setup_cache(args.cache_dir)

    if args.download_only:
        from diffusers import Flux2KleinPipeline
        print(f"[download] {REPO} -- about {DOWNLOAD_GB:.0f} GB into {cache_dir}")
        print(f"[download] cached at {Flux2KleinPipeline.download(REPO)}")
        return 0

    mode = MODES[args.mode]
    preflight(mode, cache_dir)

    import torch

    torch.cuda.reset_peak_memory_stats()
    try:
        pipe, load_s = load_pipeline(mode, args)

        print(f"[generate] {args.size}x{args.size}, {args.steps} steps, seed {args.seed}")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = pipe(
            prompt=args.prompt,
            height=args.size,
            width=args.size,
            guidance_scale=1.0,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
        gen_s = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError as exc:
        print(f"\n!!! CUDA OOM in mode '{mode.key}'\n{exc}\n")
        print("Next things to try, cheapest first:")
        if args.size > 512:
            print(f"  1. --size 512                 (down from {args.size})")
        if mode.key != "both4bit":
            print("  2. --mode both4bit            (~6.5 GB, quality drops)")
        if mode.key != "sequential":
            print("  3. --mode sequential          (~3.5 GB, much slower)")
        print("  4. Close anything else on the GPU. Windows reserves VRAM for the desktop.")
        return 1

    image.save(args.out)

    info = gpu_info()
    print("\n" + "-" * 70)
    print(f"  mode              {mode.key}")
    print(f"  load              {load_s:.1f}s")
    print(f"  generate          {gen_s:.1f}s  ({gen_s / args.steps:.2f}s/step)")
    print(f"  peak allocated    {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"  peak reserved     {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB  <- compare to your card")
    if info:
        print(f"  nvidia-smi used   {int(info['memory.used']) / 1024:.2f} GiB (incl. CUDA context + desktop)")
    print(f"  saved             {args.out.resolve()}")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
