
import os
from pathlib import Path

# 路径配置
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data")).resolve()
AUTHOR_DIR = ROOT / "author"
PICTURE_DIR = ROOT / "picture"
INDEX_DIR = ROOT / "index_cache"

# 确保目录存在
AUTHOR_DIR.mkdir(exist_ok=True)
PICTURE_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

AUTHOR_JSON_PATH = AUTHOR_DIR / "author.json"
EMB_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"
PASSAGES_PATH = INDEX_DIR / "passages.json"
META_PATH = INDEX_DIR / "index_meta.json"

# 模型配置
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
TEXT2IMG_MODEL = os.getenv("TEXT2IMG_MODEL", "runwayml/stable-diffusion-v1-5")

# 参数配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CONTENT_QUERY_TOP_K = 3
WORK_LIST_MAX_CHUNKS = 100

