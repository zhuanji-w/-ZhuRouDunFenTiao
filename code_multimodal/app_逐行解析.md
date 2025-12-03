# app.py 逐行解析（28-138行）

## 第 28 行：全局变量声明
```python
rag_engine: Optional[RAGEngine] = None
```
**解释**：
- 声明一个全局变量 `rag_engine`，类型是 `Optional[RAGEngine]`（可以是 `RAGEngine` 对象或 `None`）
- 初始值为 `None`，表示 RAG 引擎还未初始化
- 使用全局变量的原因是：RAG 引擎初始化成本高（加载模型），需要在多个请求间复用
- `Optional` 表示这个变量可以是 `RAGEngine` 类型或 `None`

---

## 第 30-32 行：请求数据模型
```python
class ChatRequest(BaseModel):
    query: str
    generate_image: bool = False
```
**解释**：
- 定义一个 Pydantic 数据模型类 `ChatRequest`，用于接收聊天请求
- `query: str`：必需字段，用户的问题/查询文本
- `generate_image: bool = False`：可选字段，是否生成配图，默认 `False`
- `BaseModel` 是 Pydantic 的基类，自动进行数据验证和类型转换
- 当客户端发送 JSON 请求时，FastAPI 会自动将 JSON 转换为这个对象

**示例请求**：
```json
{
    "query": "你是谁",
    "generate_image": true
}
```

---

## 第 34-37 行：响应数据模型
```python
class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    images: List[str] = []
```
**解释**：
- 定义响应数据模型 `ChatResponse`，用于返回聊天结果
- `answer: str`：AI 生成的回答文本
- `sources: List[dict]`：检索到的参考来源列表（包含文档路径、得分等信息）
- `images: List[str] = []`：生成的图片 URL 列表，默认为空列表
- FastAPI 会自动将这个对象转换为 JSON 返回给客户端

**示例响应**：
```json
{
    "answer": "我是多模态RAG智能助手...",
    "sources": [
        {"source": "001 - 刘慈欣/三体.txt", "score": 0.85}
    ],
    "images": ["/images/20231201_120000/picture_1.png"]
}
```

---

## 第 39-47 行：获取 RAG 引擎函数（单例模式）
```python
def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        # 初始化RAG系统
        print("Initializing RAG Engine...")
        data_sig = compute_data_signature(DATA_DIR)
        corpus = build_corpus(DATA_DIR)
        rag_engine = RAGEngine(corpus, data_signature=data_sig)
    return rag_engine
```
**逐行解释**：
- **第 39 行**：定义函数 `get_rag_engine()`，用于获取或创建 RAG 引擎实例
- **第 40 行**：`global rag_engine` 声明使用全局变量，否则函数内赋值会被当作局部变量
- **第 41 行**：`if rag_engine is None:` 检查引擎是否已初始化
- **第 43 行**：打印初始化信息，方便调试
- **第 44 行**：`data_sig = compute_data_signature(DATA_DIR)` 计算数据目录的签名（文件列表、大小、修改时间等），用于判断数据是否变化
- **第 45 行**：`corpus = build_corpus(DATA_DIR)` 从 `data/` 目录构建文档语料库（读取所有 PDF/TXT/MD 文件并分块）
- **第 46 行**：创建 `RAGEngine` 实例，传入语料库和数据签名
- **第 47 行**：返回 RAG 引擎实例

**设计模式**：这是**单例模式（Singleton）**，确保整个应用只有一个 RAG 引擎实例，避免重复加载模型浪费资源

---

## 第 49-51 行：应用启动事件
```python
@app.on_event("startup")
async def startup_event():
    get_rag_engine()
```
**解释**：
- `@app.on_event("startup")`：FastAPI 的装饰器，表示在应用启动时执行
- `async def`：异步函数，FastAPI 支持异步操作
- `get_rag_engine()`：应用启动时预加载 RAG 引擎，这样第一个请求就能立即使用，不需要等待初始化

**作用**：预热（Warm-up），提前加载模型，提升首请求响应速度

---

## 第 53-83 行：文件上传接口
```python
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
```
**第 53 行**：`@app.post("/upload")` 定义 POST 接口，路径为 `/upload`
**第 54 行**：`async def upload_file(...)` 异步上传处理函数
- `file: UploadFile = File(...)`：接收上传的文件
- `UploadFile` 是 FastAPI 的文件类型
- `File(...)` 表示文件是必需的

```python
    """上传文件到DATA_DIR"""
    try:
```
**第 55-56 行**：函数文档字符串和异常处理开始

```python
        # 允许的扩展名
        allowed_exts = {".pdf", ".txt", ".md"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="只支持 .pdf, .txt, .md 文件")
```
**第 57-61 行**：文件类型验证
- 定义允许的文件扩展名集合
- 获取上传文件的扩展名并转为小写
- 如果不在允许列表中，返回 400 错误

```python
        # 保存文件 - 创建与文件同名的文件夹
        file_path_obj = Path(file.filename)
        file_stem = file_path_obj.stem # 获取不带扩展名的文件名
        
        # 创建对应的目录: data/文件名/
        save_path = DATA_DIR / file_stem
        save_path.mkdir(exist_ok=True)
        
        file_location = save_path / file.filename
```
**第 63-71 行**：文件保存路径构建
- 将文件名转为 `Path` 对象
- `file_stem` 获取文件名（不含扩展名），例如 `"三体.txt"` → `"三体"`
- 在 `data/` 目录下创建同名文件夹
- `mkdir(exist_ok=True)` 如果文件夹已存在不报错
- 构建完整保存路径：`data/文件名/文件名.扩展名`

```python
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
```
**第 73-74 行**：保存文件
- `"wb+"` 以二进制写入模式打开文件
- `shutil.copyfileobj()` 将上传的文件流复制到目标文件

```python
        return {"info": f"file '{file.filename}' saved at '{file_location}'", "message": "上传成功，请刷新服务以建立索引"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")
```
**第 80 行**：返回成功信息
**第 82-83 行**：捕获所有异常，返回 500 错误

---

## 第 85-91 行：刷新索引接口
```python
@app.post("/refresh")
async def refresh_index():
    """强制刷新索引"""
    global rag_engine
    rag_engine = None
    get_rag_engine()
    return {"message": "Index refreshed"}
```
**解释**：
- **第 85 行**：定义 POST 接口 `/refresh`
- **第 87 行**：文档字符串说明功能
- **第 88 行**：使用全局变量
- **第 89 行**：将引擎设为 `None`，强制重新初始化
- **第 90 行**：重新初始化引擎（会重新扫描 `data/` 目录并重建索引）
- **第 91 行**：返回成功消息

**使用场景**：上传新文件后，调用此接口刷新索引，新文件才会被检索到

---

## 第 93-128 行：聊天接口（核心接口）
```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
```
**第 93 行**：定义 POST 接口 `/chat`
- `response_model=ChatResponse` 指定响应格式为 `ChatResponse`，FastAPI 会自动验证

**第 94 行**：接收 `ChatRequest` 类型的请求

```python
    engine = get_rag_engine()
    
    # 检索
    contexts, query_type = engine.retrieve(request.query)
    
    # 生成回答
    answer = engine.generate_answer(request.query, contexts, query_type)
```
**第 95-101 行**：核心处理流程
- 获取 RAG 引擎实例
- `retrieve()` 检索相关文档片段，返回上下文和查询类型
- `generate_answer()` 基于上下文生成回答

```python
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
```
**第 103-113 行**：可选图片生成
- 如果请求中 `generate_image=True`
- 构建图片生成提示词
- 创建时间戳目录保存图片
- 生成图片并保存
- 将文件路径转换为 Web 访问 URL（相对于 `PICTURE_DIR`）

```python
    # 格式化Sources
    sources_data = []
    for ctx in contexts[:3]:
        sources_data.append({
            "source": ctx["meta"]["source"],
            "score": ctx["score"],
            "text_snippet": ctx["text"][:100] + "..."
        })
```
**第 115-122 行**：格式化检索来源信息
- 只取前 3 个最相关的结果
- 提取文档路径、相似度得分、文本片段（前100字符）

```python
    return ChatResponse(
        answer=answer,
        sources=sources_data,
        images=image_urls
    )
```
**第 124-128 行**：返回响应对象，FastAPI 自动转为 JSON

---

## 第 130-132 行：静态文件挂载
```python
# 挂载图片目录
app.mount("/images", StaticFiles(directory=str(PICTURE_DIR)), name="images")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
```
**解释**：
- **第 131 行**：将 `/images` 路径映射到 `picture/` 目录
  - 访问 `http://localhost:8000/images/xxx.png` 会返回 `picture/xxx.png` 文件
- **第 132 行**：将根路径 `/` 映射到 `static/` 目录
  - `html=True` 表示默认返回 `index.html`
  - 访问 `http://localhost:8000/` 会返回前端页面

**注意**：挂载顺序很重要，`/` 必须放在最后，否则会拦截所有请求

---

## 第 134-136 行：主程序入口
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
**解释**：
- **第 134 行**：`if __name__ == "__main__":` 判断是否直接运行此脚本
- **第 135 行**：导入 uvicorn 服务器
- **第 136 行**：启动服务器
  - `app`：FastAPI 应用实例
  - `host="0.0.0.0"`：监听所有网络接口
  - `port=8000`：端口号

**作用**：可以直接运行 `python app.py` 启动服务（但通常用 `uvicorn` 命令启动）

---

## 📊 代码流程图

```
客户端请求
    │
    ├─→ POST /upload  → 上传文件 → 保存到 data/文件名/
    │
    ├─→ POST /refresh → 重置引擎 → 重新加载数据
    │
    └─→ POST /chat    → 检索 + 生成回答 → 返回结果
                          │
                          ├─→ 可选：生成图片
                          └─→ 返回文本 + 来源 + 图片URL
```

## 🔑 关键设计点

1. **单例模式**：RAG 引擎全局唯一，避免重复加载模型
2. **懒加载**：首次使用时才初始化引擎
3. **预加载**：启动时预热，提升响应速度
4. **异常处理**：所有接口都有错误处理
5. **类型安全**：使用 Pydantic 模型进行数据验证

