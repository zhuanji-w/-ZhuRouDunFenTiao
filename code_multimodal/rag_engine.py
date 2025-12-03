
import os
import json
import re
import datetime
import numpy as np
import faiss
import torch
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from .config import (
    DATA_DIR, EMBED_MODEL_NAME, SUMMARIZER_MODEL,
    EMB_PATH, FAISS_PATH, PASSAGES_PATH, META_PATH,
    CONTENT_QUERY_TOP_K, WORK_LIST_MAX_CHUNKS
)
from .author_manager import AuthorManager, AuthorRecognizer

class RAGEngine:
    """多模态RAG引擎"""
    
    def __init__(self, passages: List[Tuple[str, dict]], data_signature: dict | None = None):
        self.passages = passages
        self.data_signature = data_signature or {}
        
        # 初始化模型
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.encoder = SentenceTransformer(EMBED_MODEL_NAME)
        
        # 初始化作者管理器
        self.author_manager = AuthorManager(DATA_DIR)
        self.author_recognizer = AuthorRecognizer(self.author_manager)

        # 初始化摘要模型
        print(f"Loading summarizer model: {SUMMARIZER_MODEL}")
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
        if not self.passages:
            print("No passages to index.")
            self.embeddings = np.zeros((0, 384)) # 假设MiniLM维度
            self.index = faiss.IndexFlatIP(384)
            return

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
        if not hasattr(self, 'embeddings') or self.embeddings is None:
             return
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
        if not self.passages:
            return [], "other"
            
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
        # 某些模型model_max_length可能是非常大的整数，导致int转换错误或内存问题
        if model_max_len > 2048:
            model_max_len = 2048
            
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

