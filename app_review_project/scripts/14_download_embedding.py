"""下载 BERTopic 所需的中文句向量模型（BAAI/bge-small-zh-v1.5）。

为什么走 ModelScope 而不是 Hugging Face：本机实测 hf.co 与 hf-mirror.com 均连接失败
（curl http=000），ModelScope 实测 2.6 MB/s，因此用它做镜像。两份权重是同一模型。

模型体积：safetensors 95.8 MB + tokenizer 约 0.5 MB。下载后放在 models/bge-small-zh-v1.5/，
可直接作为 SentenceTransformer 的本地路径使用：

    SentenceTransformer(r"d:\\健康管理\\app_review_project\\models\\bge-small-zh-v1.5")

注意：本脚本只下模型，不装 torch / sentence-transformers / bertopic。运行前需要这些依赖，
且 Python 3.14 上是否有对应的 torch wheel 需先确认（见 README 的依赖说明）。

用法：
    python scripts/14_download_embedding.py
"""
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_ID = "BAAI/bge-small-zh-v1.5"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "bge-small-zh-v1.5"
BASE = "https://modelscope.cn/api/v1/models/{mid}/repo?Revision=master&FilePath={path}"

FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "configuration.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
]


def fetch(path):
    target = DEST / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"跳过（已存在）{path}  {target.stat().st_size / 1e6:.2f} MB")
        return
    url = BASE.format(mid=MODEL_ID, path=urllib.parse.quote(path))
    print(f"下载 {path} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(target, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"完成 {path}  {target.stat().st_size / 1e6:.2f} MB")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        fetch(p)
    total = sum((DEST / p).stat().st_size for p in FILES)
    print(f"\n全部完成：{len(FILES)} 个文件，共 {total / 1e6:.1f} MB → {DEST}")


if __name__ == "__main__":
    main()
