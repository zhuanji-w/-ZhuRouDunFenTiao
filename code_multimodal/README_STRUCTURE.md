# 多模态 RAG 项目 - 文件结构说明

## 📁 项目结构

### 🚀 主启动文件
- **`app.py`** - **这是 Web 服务的入口文件**
  - 定义了 FastAPI 应用实例 `app`
  - 提供 API 接口：`/chat`, `/upload`, `/refresh`
  - 挂载静态文件（前端页面）
  - **启动命令指向这个文件**

### 🎨 前端文件
- **`static/`** - 前端资源目录
  - `index.html` - 主页面
  - `style.css` - 样式文件
  - `script.js` - 前端交互逻辑

### ⚙️ 核心业务逻辑
- **`rag_engine.py`** - RAG 引擎核心
  - 处理向量检索
  - 生成回答
  - 查询分类
  
- **`utils.py`** - 工具函数
  - 文档加载（PDF/TXT/MD）
  - 文本分块
  - 语料库构建

- **`author_manager.py`** - 作者管理
  - 作者信息提取
  - 作者识别

- **`image_gen.py`** - 图片生成
  - Stable Diffusion 图片生成

### 🔧 配置文件
- **`config.py`** - 全局配置
  - 路径配置
  - 模型配置
  - 参数配置

### 📝 其他文件
- **`multimodal.py`** - **旧版本主文件**（已被重构，现在不使用）
- `download_model.py` - 模型下载脚本

## 🔄 启动流程

1. **启动命令**：`python3 -m uvicorn code_multimodal.app:app ...`
2. **加载**：`app.py` 被加载
3. **初始化**：在 `@app.on_event("startup")` 中初始化 RAG 引擎
4. **提供服务**：启动 FastAPI Web 服务，监听 8000 端口

## 📊 代码调用关系

```
app.py (启动)
  ├── config.py (配置)
  ├── utils.py (工具函数)
  ├── rag_engine.py (核心引擎)
  │   ├── author_manager.py (作者管理)
  │   └── config.py (配置)
  └── image_gen.py (图片生成)
```

## 💡 快速定位

- **想修改 API 接口** → 编辑 `app.py`
- **想修改前端界面** → 编辑 `static/` 下的文件
- **想修改 RAG 逻辑** → 编辑 `rag_engine.py`
- **想修改配置** → 编辑 `config.py`

