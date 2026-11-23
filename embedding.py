"""Local CPU encoder for multi-qa-MiniLM-L6-cos-v1."""

import os
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"
DEFAULT_MODEL_DIR = PROJECT / "models" / MODEL_ID.split("/")[-1]
RUNTIME_DIR = PROJECT / "runtime"


class MiniLMEncoder:
    def __init__(self, model_dir=None):
        path = Path(model_dir or DEFAULT_MODEL_DIR)
        missing = [name for name in ("config.json", "model.safetensors", "tokenizer.json")
                   if not (path / name).is_file()]
        if missing:
            raise ValueError(f"语义模型文件不全（{', '.join(missing)}）。请运行 Start-Semantic.cmd 完成准备。")
        if RUNTIME_DIR.is_dir() and str(RUNTIME_DIR) not in sys.path:
            sys.path.insert(0, str(RUNTIME_DIR))
        os.environ.setdefault("USE_TF", "0")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ValueError("缺少语义检索的 Python 依赖。请运行 Start-Semantic.cmd。") from error

        self._torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, use_safetensors=True).eval()

    def encode(self, texts):
        import numpy as np

        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        torch = self._torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                tokens = self.tokenizer(
                    texts[start:start + 32], padding=True, truncation=True, max_length=512, return_tensors="pt",
                )
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens.attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                batches.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.concatenate(batches, axis=0)
