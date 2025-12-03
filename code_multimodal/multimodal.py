from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

import faiss
import fitz  # PyMuPDF
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data")).resolve()
AUTHOR_DIR = ROOT / "author"
AUTHOR_DIR.mkdir(exist_ok=True)
AUTHOR_JSON_PATH = AUTHOR_DIR / "author.json"
PICTURE_DIR = ROOT / "picture"
INDEX_DIR = ROOT / "index_cache"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"
PASSAGES_PATH = INDEX_DIR / "passages.json"
META_PATH = INDEX_DIR / "index_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
TEXT2IMG_MODEL = os.getenv("TEXT2IMG_MODEL", "runwayml/stable-diffusion-v1-5")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CONTENT_QUERY_TOP_K = 3
WORK_LIST_MAX_CHUNKS = 100


class AuthorManager:
    """作者信息管理器，从data目录自动提取作者信息"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.author_json_path = AUTHOR_JSON_PATH
        self.authors = self._load_or_create_author_data()
    
    def _extract_authors_from_data_dir(self) -> Dict[str, Dict]:
        """从data目录提取作者信息"""
        authors = {}
        
        if not self.data_dir.exists():
            print(f"警告: 数据目录不存在: {self.data_dir}")
            return authors
        
        # 遍历data目录下的所有子目录
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # 解析目录名，提取作者信息
                author_info = self._parse_author_from_dirname(item.name)
                if author_info:
                    author_name = author_info["primary_name"]
                    authors[author_name] = author_info
        
        return authors
    
    def _parse_author_from_dirname(self, dirname: str) -> Optional[Dict]:
        """从目录名解析作者信息"""
        # 匹配模式: "001 - 刘慈欣(Cixin Liu)" 或 "002 - Frank Herbert"
        pattern = r"^\d+\s*-\s*([^(]+)(?:\(([^)]+)\))?$"
        match = re.match(pattern, dirname.strip())
        
        if match:
            primary_name = match.group(1).strip()
            alternate_name = match.group(2).strip() if match.group(2) else ""
            
            author_info = {
                "primary_name": primary_name,
                "dirname": dirname,
                "is_chinese": self._is_chinese_name(primary_name)
            }
            
            if alternate_name:
                author_info["alternate_names"] = [alternate_name]
            else:
                author_info["alternate_names"] = []
                
            return author_info
        
        return None
    
    def _is_chinese_name(self, name: str) -> bool:
        """判断是否为中文名"""
        return any('\u4e00' <= char <= '\u9fff' for char in name)
    
    def _load_or_create_author_data(self) -> Dict[str, Dict]:
        """加载或创建作者数据"""
        # 检查是否需要更新author.json
        needs_update = self._needs_update()
        
        if not needs_update and self.author_json_path.exists():
            try:
                with open(self.author_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"从 {self.author_json_path} 加载了 {len(data.get('authors', {}))} 位作者信息")
                    return data.get("authors", {})
            except Exception as e:
                print(f"加载author.json失败: {e}")
        
        # 重新提取作者信息
        authors = self._extract_authors_from_data_dir()
        self._save_author_data(authors)
        print(f"从data目录提取了 {len(authors)} 位作者信息")
        return authors
    
    def _needs_update(self) -> bool:
        """检查是否需要更新author.json"""
        if not self.author_json_path.exists():
            return True
        
        try:
            # 检查data目录的最后修改时间
            data_mtime = self.data_dir.stat().st_mtime
            author_mtime = self.author_json_path.stat().st_mtime
            
            # 如果data目录有更新，需要重新生成
            if data_mtime > author_mtime:
                return True
            
            # 检查目录结构是否有变化
            current_dirs = {item.name for item in self.data_dir.iterdir() if item.is_dir()}
            
            with open(self.author_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_dirs = {info["dirname"] for info in existing_data.get("authors", {}).values()}
            
            return current_dirs != existing_dirs
                
        except Exception as e:
            print(f"检查更新状态失败: {e}")
            return True
    
    def _save_author_data(self, authors: Dict[str, Dict]):
        """保存作者数据到JSON文件"""
        data = {
            "generated_time": datetime.datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "authors": authors
        }
        
        try:
            with open(self.author_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"作者信息已保存到: {self.author_json_path}")
        except Exception as e:
            print(f"保存author.json失败: {e}")
    
    def get_all_author_names(self) -> Set[str]:
        """获取所有作者名称（主名+别名）"""
        all_names = set()
        for author_info in self.authors.values():
            all_names.add(author_info["primary_name"])
            all_names.update(author_info.get("alternate_names", []))
        return all_names
    
    def get_primary_author_names(self) -> Set[str]:
        """获取所有作者的主名称"""
        return {author_info["primary_name"] for author_info in self.authors.values()}
    
    def find_author_by_name(self, name: str) -> Optional[Dict]:
        """根据名称查找作者信息（支持主名和别名）"""
        for author_info in self.authors.values():
            if (name == author_info["primary_name"] or 
                name in author_info.get("alternate_names", [])):
                return author_info
        return None


class AuthorRecognizer:
    """智能作者识别器 - 使用data目录中的实际作者信息"""
    
    def __init__(self, author_manager: AuthorManager):
        self.author_manager = author_manager
        # 从author_manager获取作者名称
        self.common_authors = author_manager.get_all_author_names()
        
        self.question_words = {
            "作品", "著作", "书目", "哪些", "什么", "有哪些", "都有什么", 
            "包含什么", "是什么", "介绍", "列举", "查询", "查找", "搜索"
        }
        self.work_keywords = {"作品", "著作", "书目", "小说", "文章", "书籍", "文献"}
        
    def recognize_author(self, query: str) -> Optional[str]:
        """多策略智能识别作者"""
        strategies = [
            self._strategy_common_author,
            self._strategy_pattern_match,
            self._strategy_keyword_context,
            self._strategy_semantic_extraction,
        ]
        
        for strategy in strategies:
            author = strategy(query)
            if author and self._validate_author(author):
                return author
        return None
    
    def _strategy_common_author(self, query: str) -> Optional[str]:
        """从data目录的作者库中识别"""
        for author in self.common_authors:
            if author in query:
                return author
        return None
    
    def _strategy_pattern_match(self, query: str) -> Optional[str]:
        """改进的模式匹配"""
        patterns = [
            r"([\u4e00-\u9fa5]{2,})的作品",
            r"([\u4e00-\u9fa5]{2,})作品",
            r"([\u4e00-\u9fa5]{2,})写过什么",
            r"([\u4e00-\u9fa5]{2,})有什么作品",
            r"介绍([\u4e00-\u9fa5]{2,})的作品",
            r"查询([\u4e00-\u9fa5]{2,})的作品",
            r"查找([\u4e00-\u9fa5]{2,})的作品",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                author = match.group(1)
                if author not in self.question_words and author in self.common_authors:
                    return author
        return None
    
    def _strategy_keyword_context(self, query: str) -> Optional[str]:
        """基于关键词上下文的作者识别"""
        words = re.findall(r"[\u4e00-\u9fa5a-zA-Z]{2,}", query)
        
        # 在作品关键词前后寻找作者名
        for i, word in enumerate(words):
            if word in self.work_keywords:
                # 检查关键词前的词
                if i > 0:
                    candidate = words[i-1]
                    if (len(candidate) >= 2 and candidate not in self.question_words and 
                        candidate in self.common_authors):
                        return candidate
                # 检查关键词后的词
                if i < len(words) - 1:
                    candidate = words[i+1]
                    if (len(candidate) >= 2 and candidate not in self.question_words and 
                        candidate in self.common_authors):
                        return candidate
        return None
    
    def _strategy_semantic_extraction(self, query: str) -> Optional[str]:
        """语义提取作者名"""
        # 移除疑问词
        clean_query = query
        for word in self.question_words | self.work_keywords:
            clean_query = clean_query.replace(word, "")
        
        # 提取可能的作者名
        candidates = re.findall(r"[\u4e00-\u9fa5a-zA-Z]{2,}", clean_query)
        for candidate in candidates:
            if candidate not in self.question_words and candidate in self.common_authors:
                return candidate
        return None
    
    def _validate_author(self, author: str) -> bool:
        """验证作者名的合理性"""
        return (2 <= len(author) and 
                author not in self.question_words and
                author in self.common_authors)


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
        raise ValueError(f"未在 {data_dir} 下找到可用文档（支持 PDF/TXT/MD）。")
    
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


class RAGEngine:
    """多模态RAG引擎"""
    
    def __init__(self, passages: List[Tuple[str, dict]], data_signature: dict | None = None):
        self.passages = passages
        self.data_signature = data_signature or {}
        self.encoder = SentenceTransformer(EMBED_MODEL_NAME)
        
        # 初始化作者管理器
        self.author_manager = AuthorManager(DATA_DIR)
        self.author_recognizer = AuthorRecognizer(self.author_manager)

        # 初始化摘要模型
        self.summarizer = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            device=0 if torch.cuda.is_available() else -1,
        )

        # 加载或构建索引
        if self._load_index_from_disk():
            print("索引命中缓存，跳过重新编码")
        else:
            print("索引未命中，重新构建索引")
            self._build_index()
            self._save_index_to_disk()

    def _build_index(self):
        """构建向量索引"""
        texts = [p[0] for p in self.passages]
        self.embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def _save_index_to_disk(self):
        """保存索引到磁盘"""
        try:
            np.save(EMB_PATH, self.embeddings)
            faiss.write_index(self.index, str(FAISS_PATH))
            
            with PASSAGES_PATH.open("w", encoding="utf-8") as f:
                for text, meta in self.passages:
                    f.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False) + "\n")
            
            meta = {
                "data_signature": self.data_signature,
                "embed_model": EMBED_MODEL_NAME,
                "dim": int(self.embeddings.shape[1]),
                "created_time": datetime.datetime.now().isoformat()
            }
            
            with META_PATH.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存索引失败: {e}")

    def _load_index_from_disk(self) -> bool:
        """从磁盘加载索引"""
        try:
            if not all([EMB_PATH.exists(), FAISS_PATH.exists(), 
                       PASSAGES_PATH.exists(), META_PATH.exists()]):
                return False

            with META_PATH.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            if (meta.get("embed_model") != EMBED_MODEL_NAME or 
                meta.get("data_signature") != self.data_signature):
                return False

            self.embeddings = np.load(EMB_PATH)
            self.index = faiss.read_index(str(FAISS_PATH))

            passages = []
            with PASSAGES_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    passages.append((obj["text"], obj["meta"]))
                    
            self.passages = passages
            return True
            
        except Exception as e:
            print(f"索引缓存读取失败: {e}")
            return False

    def classify_query_type(self, query: str) -> Tuple[str, Optional[str]]:
        """智能查询分类"""
        query_lower = query.lower()
        
        # 查询类型关键词
        work_list_keywords = {"作品", "著作", "书目", "有哪些", "都有什么", "包含什么", 
                            "写过什么", "有什么作品", "介绍作品", "列举作品"}
        content_query_keywords = {"故事", "内容", "讲了什么", "情节", "主要内容", "梗概", 
                                "简介", "介绍", "讲述", "描写", "总结"}

        # 1. 作品罗列类识别
        if any(keyword in query_lower for keyword in work_list_keywords):
            author = self.author_recognizer.recognize_author(query)
            return "work_list", author

        # 2. 内容查询类识别
        elif any(keyword in query_lower for keyword in content_query_keywords):
            # 提取书名号内的作品名
            book_pattern = re.compile(r"《([^》]+)》")
            books = book_pattern.findall(query)
            if books:
                return "content_query", books[0]
            
            # 提取可能的作品名
            book_candidates = re.findall(r"([\u4e00-\u9fa5a-zA-Z]{2,10})", query)
            for candidate in book_candidates:
                if len(candidate) >= 2 and candidate not in work_list_keywords | content_query_keywords:
                    return "content_query", candidate
                    
            return "content_query", None

        # 3. 其他类型
        else:
            return "other", None

    def retrieve(self, query: str) -> Tuple[List[dict], str]:
        """智能检索"""
        query_type, target = self.classify_query_type(query)
        contexts = []
        
        print(f"查询类型: {query_type}, 目标: {target}")

        if query_type == "work_list" and target:
            print(f"检索作者'{target}'的作品...")
            contexts = self._retrieve_works_by_author(target)
            
        elif query_type == "content_query":
            print(f"检索作品'{target if target else '相关内容'}...")
            contexts = self._retrieve_content_by_query(query, target)
            
        else:
            print("执行通用检索...")
            contexts = self._retrieve_general(query)
            
        return contexts, query_type

    def _retrieve_works_by_author(self, author: str) -> List[dict]:
        """检索指定作者的作品 - 只基于文件路径匹配"""
        contexts = []
        
        for text, meta in self.passages:
            # 只匹配文件路径中的作者名，不匹配文本内容
            source_path = meta.get("source", "").lower()
            full_path = meta.get("full_path", "").lower()
            
            # 路径匹配：检查作者名是否出现在文件路径中
            path_match = (
                author in source_path or
                author in full_path or
                any(author in part for part in source_path.split('/'))
            )
            
            if path_match:
                contexts.append({
                    "text": text,
                    "meta": meta,
                    "score": 1.0
                })

        # 去重处理：同一文档只保留一个片段
        unique_sources = set()
        unique_contexts = []
        for ctx in contexts:
            source = ctx["meta"]["source"]
            if source not in unique_sources:
                unique_sources.add(source)
                unique_contexts.append(ctx)
                
        return unique_contexts[:WORK_LIST_MAX_CHUNKS]

    def _retrieve_content_by_query(self, query: str, target: Optional[str]) -> List[dict]:
        """基于查询内容检索"""
        contexts = []
        
        if target:
            # 先筛选相关文档（基于文件名中的作品名）
            candidate_indices = []
            for i, (text, meta) in enumerate(self.passages):
                file_book_name = meta.get("file_book_name", "").lower()
                if target in file_book_name or target in meta.get("source", "").lower():
                    candidate_indices.append(i)
            
            if candidate_indices:
                # 对候选文档计算相似度
                candidate_texts = [self.passages[i][0] for i in candidate_indices]
                candidate_embeddings = self.encoder.encode(candidate_texts, convert_to_numpy=True)
                query_emb = self.encoder.encode([query], convert_to_numpy=True)
                
                faiss.normalize_L2(query_emb)
                faiss.normalize_L2(candidate_embeddings)
                
                # 创建临时索引进行搜索
                temp_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
                temp_index.add(candidate_embeddings)
                
                scores, idx = temp_index.search(query_emb, min(len(candidate_indices), CONTENT_QUERY_TOP_K * 3))
                
                for score, i in zip(scores[0], idx[0]):
                    if score < 0.3:
                        continue
                    original_idx = candidate_indices[i]
                    text, meta = self.passages[original_idx]
                    contexts.append({
                        "text": text,
                        "meta": meta,
                        "score": float(score)
                    })
                
                contexts.sort(key=lambda x: x["score"], reverse=True)
        else:
            # 全局检索
            query_emb = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_emb)
            scores, idx = self.index.search(query_emb, CONTENT_QUERY_TOP_K)
            
            for score, i in zip(scores[0], idx[0]):
                if i == -1 or score < 0.3:
                    continue
                text, meta = self.passages[i]
                contexts.append({
                    "text": text,
                    "meta": meta,
                    "score": float(score)
                })
        
        return contexts[:CONTENT_QUERY_TOP_K]

    def _retrieve_general(self, query: str) -> List[dict]:
        """通用检索"""
        query_emb = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        scores, idx = self.index.search(query_emb, CONTENT_QUERY_TOP_K)
        
        contexts = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1 or score < 0.3:
                continue
            text, meta = self.passages[i]
            contexts.append({
                "text": text,
                "meta": meta,
                "score": float(score)
            })
            
        return contexts

    def generate_answer(self, query: str, contexts: List[dict], query_type: str) -> str:
        """生成回答"""
        if query_type == "work_list":
            return self._generate_work_list_answer(contexts)
        elif query_type == "content_query":
            return self._generate_content_answer(query, contexts)
        else:
            return self._generate_general_answer(query, contexts)

    def _generate_work_list_answer(self, contexts: List[dict]) -> str:
        """生成作品列表回答 - 只基于文件名，不提取内容中的作品"""
        # 使用集合来去重，基于文件名中的作品名
        works = set()
        work_source_map = {}
        
        for ctx in contexts:
            meta = ctx["meta"]
            # 只使用文件名中提取的作品名
            book_name = meta.get("file_book_name", "")
            source = meta["source"]
            
            # 确保作品名有效
            if book_name and len(book_name.strip()) >= 2:
                # 清理作品名
                clean_book_name = self._clean_book_name(book_name)
                if clean_book_name:
                    works.add(clean_book_name)
                    if clean_book_name not in work_source_map:
                        work_source_map[clean_book_name] = set()
                    work_source_map[clean_book_name].add(source)
        
        if not works:
            return "未找到相关作品。"

        # 格式化输出
        answer_lines = [f"作品汇总（共{len(works)}部）："]
        for idx, work in enumerate(sorted(works), 1):
            sources = work_source_map.get(work, set())
            source_str = "、".join(sorted(sources)[:2])  # 最多显示2个来源
            answer_lines.append(f"{idx}. 《{work}》（来源：{source_str}）")
            
        return "\n".join(answer_lines)

    def _clean_book_name(self, book_name: str) -> str:
        """清理作品名"""
        # 移除可能的数字和符号
        cleaned = re.sub(r'^\d+[-\.\s]*', '', book_name)
        cleaned = re.sub(r'[\[\]\(\)\{\}]+', '', cleaned)
        cleaned = cleaned.strip()
        return cleaned

    def _generate_content_answer(self, query: str, contexts: List[dict]) -> str:
        """生成内容回答"""
        if not contexts:
            return "未找到相关内容。"

        joined_contexts = "\n\n".join([
            f"[来源 {ctx['meta']['source']} - 第{ctx['meta']['page']}页]\n{ctx['text']}" 
            for ctx in contexts
        ])
        
        prompt = f"""请用中文详细回答以下问题，必须严格基于提供的参考材料：

问题：{query}

参考材料：
{joined_contexts}

要求：
1. 中文回答，条理清晰；
2. 准确概括材料内容；
3. 无相关信息时回复"未找到足够信息"。"""

        # 截断过长的Prompt
        max_input_chars = 1500
        if len(prompt) > max_input_chars:
            prompt = prompt[:max_input_chars] + "\n...(材料过长，已截断)"

        try:
            summary = self._generate_with_model(prompt, max_new_tokens=300)
            return summary
        except Exception as e:
            return f"生成回答时出错: {e}"

    def _generate_general_answer(self, query: str, contexts: List[dict]) -> str:
        """生成通用回答"""
        if not contexts:
            return "未找到相关内容。"

        joined_contexts = "\n\n".join([
            f"[来源 {ctx['meta']['source']}]\n{ctx['text']}" 
            for ctx in contexts
        ])
        
        prompt = f"""问题：{query}

参考材料：
{joined_contexts}

请基于以上材料用中文回答："""

        max_input_chars = 1200
        if len(prompt) > max_input_chars:
            prompt = prompt[:max_input_chars] + "\n...(截断)"

        try:
            summary = self._generate_with_model(prompt, max_new_tokens=200)
            return summary
        except Exception as e:
            return f"生成回答时出错: {e}"

    def _generate_with_model(self, prompt: str, max_new_tokens: int = 200) -> str:
        """使用模型生成文本"""
        tokenizer = self.summarizer.tokenizer
        model = self.summarizer.model
        device = model.device

        model_max_len = getattr(tokenizer, "model_max_length", 1024)
        enc = tokenizer(
            prompt,
            max_length=model_max_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def build_image_prompts(query: str, contexts: List[dict]) -> List[str]:
    """构建图片生成提示"""
    prompts = []
    base_instruction = "digital art, highly detailed, cinematic lighting"
    
    for ctx in contexts[:2]:  # 最多生成2张图片
        meta = ctx["meta"]
        prompt = (
            f"{base_instruction}. Scene from {meta['source']}: "
            f"{ctx['text'][:100]}... Focus on key elements from the text."
        )
        prompts.append(prompt)
    
    return prompts[:2]


def generate_images(prompts: List[str], output_dir: Path) -> List[Path]:
    """生成图片"""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_paths = []
    
    if not prompts:
        return image_paths
        
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            TEXT2IMG_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
            
    except Exception as e:
        print(f"加载文生图模型失败: {e}")
        return image_paths

    for idx, prompt in enumerate(prompts, start=1):
        try:
            image = pipe(prompt, num_inference_steps=20).images[0]
            img_path = output_dir / f"picture_{idx}.png"
            image.save(img_path)
            image_paths.append(img_path)
        except Exception as e:
            print(f"生成图片失败: {e}")
            
    return image_paths


def main():
    """主函数"""
    # 构建RAG系统
    try:
        data_sig = compute_data_signature(DATA_DIR)
        corpus = build_corpus(DATA_DIR)
        rag = RAGEngine(corpus, data_signature=data_sig)
        
    except Exception as e:
        print(f"初始化RAG系统失败: {e}")
        return

    # 交互式查询
    while True:
        try:
            query = input("\n请输入问题（输入'退出'结束）: ").strip()
            if query.lower() in ['退出', 'exit', 'quit']:
                break
            if not query:
                continue
                
            # 检索
            contexts, query_type = rag.retrieve(query)
            
            if not contexts:
                print("未检索到相关内容。")
                continue
                
            # 显示检索结果
            print(f"\n=== 检索到 {len(contexts)} 个相关结果 ===")
            for i, ctx in enumerate(contexts[:3], 1):
                print(f"[{i}] 分数: {ctx['score']:.4f}")
                print(f"    来源: {ctx['meta']['source']}")
                print(f"    文本: {ctx['text'][:100]}...")
            
            # 生成回答
            answer = rag.generate_answer(query, contexts, query_type)
            print("\n=== RAG 回答 ===")
            print(answer)
            
            # 生成图片（可选）
            generate_img = input("\n是否生成图片？(y/n): ").strip().lower()
            if generate_img == 'y':
                prompts = build_image_prompts(query, contexts)
                timestamp_dir = PICTURE_DIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_paths = generate_images(prompts, timestamp_dir)
                
                if image_paths:
                    print("\n=== 图片生成完成 ===")
                    for path in image_paths:
                        print(f"- {path}")
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理过程中出错: {e}")


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs(PICTURE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(AUTHOR_DIR, exist_ok=True)
    
    print("多模态RAG系统启动")
    print(f"数据目录: {DATA_DIR}")
    print(f"作者信息目录: {AUTHOR_DIR}")
    print(f"图片目录: {PICTURE_DIR}")
    
    main()