
import os
import shutil
import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import DATA_DIR, PICTURE_DIR
from .utils import build_corpus, compute_data_signature
from .rag_engine import RAGEngine
from .image_gen import build_image_prompts, generate_images

# 初始化App
app = FastAPI(title="Multimodal RAG API")

# 挂载静态文件（前端）
# 假设前端文件放在 static 目录下
STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir()

# 全局变量存储RAG引擎实例
rag_engine: Optional[RAGEngine] = None

class ChatRequest(BaseModel):
    query: str
    generate_image: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    images: List[str] = []

def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        # 初始化RAG系统
        print("Initializing RAG Engine...")
        data_sig = compute_data_signature(DATA_DIR)
        corpus = build_corpus(DATA_DIR)
        rag_engine = RAGEngine(corpus, data_signature=data_sig)
    return rag_engine

@app.on_event("startup")
async def startup_event():
    get_rag_engine()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到DATA_DIR"""
    try:
        # 允许的扩展名
        allowed_exts = {".pdf", ".txt", ".md"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="只支持 .pdf, .txt, .md 文件")
        
        # 保存文件 - 创建与文件同名的文件夹
        file_path_obj = Path(file.filename)
        file_stem = file_path_obj.stem # 获取不带扩展名的文件名
        
        # 创建对应的目录: data/文件名/
        save_path = DATA_DIR / file_stem
        save_path.mkdir(exist_ok=True)
        
        file_location = save_path / file.filename
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 上传后可能需要重新构建索引，或者简单的重启服务
        # 这是一个简化实现，实际上可能需要动态更新索引
        # 为了简单起见，这里暂不立即更新索引，建议重启服务生效，或者提供一个刷新接口
        
        return {"info": f"file '{file.filename}' saved at '{file_location}'", "message": "上传成功，请刷新服务以建立索引"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")

@app.post("/refresh")
async def refresh_index():
    """强制刷新索引"""
    global rag_engine
    rag_engine = None
    get_rag_engine()
    return {"message": "Index refreshed"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    engine = get_rag_engine()
    
    # 检索
    contexts, query_type = engine.retrieve(request.query)
    
    # 生成回答
    answer = engine.generate_answer(request.query, contexts, query_type)
    
    # 图片生成
    image_urls = []
    if request.generate_image:
        prompts = build_image_prompts(request.query, contexts)
        timestamp_dir = PICTURE_DIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_paths = generate_images(prompts, timestamp_dir)
        # 转换为相对URL
        # 假设我们挂载了 /images 路径到 PICTURE_DIR
        for path in image_paths:
            rel_path = path.relative_to(PICTURE_DIR)
            image_urls.append(f"/images/{rel_path}")

    # 格式化Sources
    sources_data = []
    for ctx in contexts[:3]:
        sources_data.append({
            "source": ctx["meta"]["source"],
            "score": ctx["score"],
            "text_snippet": ctx["text"][:100] + "..."
        })

    return ChatResponse(
        answer=answer,
        sources=sources_data,
        images=image_urls
    )

# 挂载图片目录
app.mount("/images", StaticFiles(directory=str(PICTURE_DIR)), name="images")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

