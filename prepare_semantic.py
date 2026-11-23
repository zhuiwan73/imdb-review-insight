"""Prepare the project's local MiniLM files and Python dependencies."""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlopen

from embedding import DEFAULT_MODEL_DIR, MODEL_ID, MODEL_REVISION, RUNTIME_DIR


MODEL_SHA256 = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer.json": "7fa9272f7ef1ebd1666bb3bfd9d4707660ff0076ca9d1671cd9a9c6e18e03331",
    "tokenizer_config.json": "857c5db35e9664bd0ea1db3a5ee62c0ce86d79e3ec85861ca1680bd7aaff12f8",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "model.safetensors": "7bec4fd9eba43073d5c5dcf1b79b0a3397608fa063e6f626d6f8fd70a81f2d8c",
}
DEPENDENCIES = (
    "transformers==4.51.3",
    "huggingface-hub==0.30.2",
    "tokenizers==0.21.4",
    "safetensors==0.5.3",
)
SUPPORT_PACKAGES = {
    "filelock": "filelock",
    "fsspec": "fsspec",
    "packaging": "packaging",
    "yaml": "PyYAML",
    "regex": "regex",
    "requests": "requests",
    "certifi": "certifi",
    "charset_normalizer": "charset-normalizer",
    "idna": "idna",
    "urllib3": "urllib3",
    "tqdm": "tqdm",
    "colorama": "colorama",
    "typing_extensions": "typing-extensions",
}


def has_hash(path, expected):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def download(url, destination):
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if curl:
        subprocess.run(
            [curl, "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "12",
             "--max-time", "300", "--retry", "1", "--retry-max-time", "20",
             "--output", str(destination), url],
            check=True,
        )
        return
    with urlopen(url, timeout=25) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def prepare_model():
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_SHA256.items():
        destination = DEFAULT_MODEL_DIR / name
        if has_hash(destination, expected):
            continue
        print(f"正在下载 {name}…", flush=True)
        encoded_name = quote(name)
        urls = (
            f"https://modelscope.cn/models/{MODEL_ID}/resolve/master/{encoded_name}",
            f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{encoded_name}",
        )
        failures = []
        with TemporaryDirectory(prefix="download-", dir=DEFAULT_MODEL_DIR) as directory:
            temporary = Path(directory) / name
            for url in urls:
                try:
                    download(url, temporary)
                    if not has_hash(temporary, expected):
                        raise ValueError("文件校验失败")
                    temporary.replace(destination)
                    break
                except (OSError, subprocess.CalledProcessError, ValueError) as error:
                    failures.append(f"{url}: {error}")
            else:
                raise RuntimeError(f"无法取得 {name}。请检查网络后重新运行。\n" + "\n".join(failures))
    print("MiniLM 模型文件已校验。", flush=True)


def prepare_runtime():
    runtime = str(RUNTIME_DIR)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = runtime + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["USE_TF"] = "0"
    probe = [
        sys.executable, "-B", "-c",
        "import torch, numpy; from transformers import BertModel, AutoTokenizer",
    ]
    test = subprocess.run(probe, env=environment, capture_output=True, text=True)
    if test.returncode == 0:
        return
    print("正在准备项目目录里的语义检索依赖…", flush=True)
    sys.path.insert(0, runtime)
    try:
        missing = [package for module, package in SUPPORT_PACKAGES.items()
                   if importlib.util.find_spec(module) is None]
    finally:
        sys.path.remove(runtime)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-B", "-m", "pip", "--disable-pip-version-check", "install", "--target", runtime,
        "--no-deps", "--no-cache-dir", "--upgrade", "--timeout", "15", "--retries", "1",
        *DEPENDENCIES, *missing,
    ]
    if subprocess.run(command).returncode:
        subprocess.run([*command, "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    test = subprocess.run(probe, env=environment, capture_output=True, text=True)
    if test.returncode != 0:
        raise RuntimeError(f"依赖已安装，但当前 Python 无法加载语义模型。\n{test.stderr.strip()}")


def main():
    if any(importlib.util.find_spec(name) is None for name in ("torch", "numpy")):
        print("这个 Python 缺少 PyTorch 或 NumPy。请使用已安装这两项的 Anaconda Python。", file=sys.stderr)
        return 1
    try:
        prepare_runtime()
        prepare_model()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"语义检索准备失败。\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
