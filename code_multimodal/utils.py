
import re
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf_text(pdf_path: Path) -> List[str]:
    """加载PDF文本内容"""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
        doc.close()
        return pages
    except Exception as e:
        print(f"加载PDF失败 {pdf_path}: {e}")
        return []

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """文本分块处理"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        words = para.split()
        para_length = len(words)
        
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # 保留重叠部分
            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words
            current_length = len(overlap_words)
        
        current_chunk.extend(words)
        current_length += para_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def iter_documents(data_dir: Path) -> List[Path]:
    """遍历文档目录"""
    targets = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            targets.append(path)
    return targets

def load_text_file(path: Path) -> List[str]:
    """加载文本文件"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [text]
    except Exception as e:
        print(f"加载文本文件失败 {path}: {e}")
        return []

def compute_data_signature(data_dir: Path) -> dict:
    """计算数据目录签名"""
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            st = path.stat()
            files.append({
                "rel": str(path.relative_to(data_dir)),
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            })
    return {"files": files}

def extract_book_name_from_filename(filename: str) -> str:
    """从文件名提取作品名 - 只提取文件名中的作品，不提取内容"""
    # 优先匹配《书名》格式
    book_pattern = re.compile(r"《([^》]+)》")
    match = book_pattern.search(filename)
    if match:
        return match.group(1)
    
    # 如果没有书名号，清理文件名获取作品名
    # 移除年份和序号 [1997-2] 等
    filename_clean = re.sub(r"\[\d{4}[-\.]?\d*\]", "", filename)
    filename_clean = re.sub(r"\(\d{4}\)", "", filename_clean)
    
    # 去掉文件扩展名和路径
    filename_clean = Path(filename_clean).stem
    
    # 移除序号和分隔符
    filename_clean = re.sub(r"^\d+[-\.\s]*", "", filename_clean)
    filename_clean = re.sub(r"[-_\s]+", " ", filename_clean).strip()
    
    return filename_clean

def build_corpus(data_dir: Path) -> List[Tuple[str, dict]]:
    """构建文档语料库"""
    corpus = []
    documents = iter_documents(data_dir)
    
    if not documents:
        # 如果没有文件，返回空列表而不是报错，以便系统可以启动
        return []
    
    for doc in documents:
        if doc.suffix.lower() == ".pdf":
            pages = load_pdf_text(doc)
        else:
            pages = load_text_file(doc)
        
        filename = doc.name
        # 只从文件名提取作品名，不涉及文件内容
        file_book_name = extract_book_name_from_filename(filename)
        
        for page_idx, page_text in enumerate(pages):
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                meta = {
                    "source": str(doc.relative_to(data_dir)),
                    "full_path": str(doc),
                    "filename": filename,
                    "file_book_name": file_book_name,  # 只使用文件名提取的作品名
                    "page": page_idx + 1,
                    "chunk": chunk_idx + 1,
                }
                corpus.append((chunk, meta))
    
    return corpus

