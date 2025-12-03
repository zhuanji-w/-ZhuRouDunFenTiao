
import json
import datetime
import re
from pathlib import Path
from typing import Dict, Optional, Set
from .config import AUTHOR_JSON_PATH

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

