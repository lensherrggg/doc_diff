#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import difflib
import argparse
import sys
from typing import List, Tuple, Dict, Callable
from enum import Enum
import os
import numpy as np

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# 可选依赖
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChangeType(Enum):
    ADD = "新增"
    DELETE = "删除"
    MODIFY = "修改"
    UNCHANGED = "未变更"

class Paragraph:
    def __init__(self, para_id: str, content: str, order: int):
        self.para_id = para_id
        self.content = content
        self.order = order
        self.keywords = None
    
    def get_keywords(self) -> set:
        if self.keywords is not None:
            return self.keywords
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', self.content)
        stopwords = {'的', '了', '和', '与', '或', '且', '并', '对', '向', '为', '在', '于', '也', '都', '还', '要', '会', '可以', '应当', '必须', '不得'}
        keywords = {w for w in words if w not in stopwords and len(w) >= 2}
        if len(keywords) > 20:
            keywords = set(sorted(keywords)[:20])
        self.keywords = keywords
        return self.keywords

class ParagraphMatcher:
    """段落匹配器，支持多种相似度计算策略"""
    
    def __init__(self, similarity_threshold: float = 0.3, method: str = "tfidf"):
        self.threshold = similarity_threshold
        self.method = method.lower()
        
        # 条款编号正则（用于预处理时移除）
        self.clause_patterns = [
            re.compile(r'第[一二三四五六七八九十百千万\d]+条'),
            re.compile(r'第[一二三四五六七八九十百千万\d]+章'),
            re.compile(r'^\d+[\.、]'),
            re.compile(r'^[（(]\d+[）)]'),
            re.compile(r'^\d+、'),
            re.compile(r'^\d+\s+'),
        ]
        
        # 策略注册表
        self.strategies: Dict[str, Callable] = {
            "tfidf": self._compute_tfidf_similarity,
            "bm25": self._compute_bm25_similarity,
            "jaccard": self._compute_jaccard_similarity,
            "levenshtein": self._compute_levenshtein_similarity,
        }
        
        if self.method not in self.strategies:
            raise ValueError(f"不支持的方法: {self.method}，可选: {list(self.strategies.keys())}")
        
        # BM25 依赖检查
        if self.method == "bm25" and not BM25_AVAILABLE:
            raise ImportError("BM25 需要安装 rank_bm25，请运行: pip install rank-bm25")
    
    def preprocess(self, text: str, return_tokens: bool = False):
        """
        预处理文本：移除条款编号、标点，返回清洗后的字符串或词列表
        """
        # 移除条款编号
        for pattern in self.clause_patterns:
            text = pattern.sub('', text)
        # 只保留中英文和空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 合并空格
        text = re.sub(r'\s+', ' ', text).strip()
        if return_tokens:
            return text.split()
        return text
    
    def _compute_tfidf_similarity(self, old_texts: List[str], new_texts: List[str]) -> np.ndarray:
        """TF-IDF + 余弦相似度"""
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2))
        # 对旧文本拟合
        old_matrix = vectorizer.fit_transform(old_texts)
        new_matrix = vectorizer.transform(new_texts)
        return cosine_similarity(new_matrix, old_matrix)
    
    def _compute_bm25_similarity(self, old_texts: List[str], new_texts: List[str]) -> np.ndarray:
        """BM25 相似度（归一化到 [0,1]）"""
        # 将文本转为词列表（预处理时已做分词）
        old_tokenized = [self.preprocess(t, return_tokens=True) for t in old_texts]
        bm25 = BM25Okapi(old_tokenized)
        sim_matrix = np.zeros((len(new_texts), len(old_texts)))
        for i, new_text in enumerate(new_texts):
            new_tokens = self.preprocess(new_text, return_tokens=True)
            scores = bm25.get_scores(new_tokens)  # 原始 BM25 分数
            # 归一化：除以最大分数（避免阈值失效），若最大为0则保持0
            max_score = np.max(scores) if scores.size > 0 else 1.0
            if max_score > 0:
                scores = scores / max_score
            sim_matrix[i, :] = scores
        return sim_matrix
    
    def _compute_jaccard_similarity(self, old_texts: List[str], new_texts: List[str]) -> np.ndarray:
        """Jaccard 相似度（基于词集）"""
        old_sets = [set(self.preprocess(t, return_tokens=True)) for t in old_texts]
        new_sets = [set(self.preprocess(t, return_tokens=True)) for t in new_texts]
        sim_matrix = np.zeros((len(new_texts), len(old_texts)))
        for i, new_set in enumerate(new_sets):
            for j, old_set in enumerate(old_sets):
                if not new_set and not old_set:
                    sim = 1.0
                elif not new_set or not old_set:
                    sim = 0.0
                else:
                    inter = len(new_set & old_set)
                    union = len(new_set | old_set)
                    sim = inter / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
        return sim_matrix
    
    def _compute_levenshtein_similarity(self, old_texts: List[str], new_texts: List[str]) -> np.ndarray:
        """Levenshtein 编辑距离相似度：1 - 编辑距离 / max(len1, len2)"""
        try:
            from Levenshtein import distance as lev_distance
        except ImportError:
            # 若未安装 python-Levenshtein，使用纯 Python 实现（较慢）
            def lev_distance(a, b):
                # 简单的动态规划，仅作 fallback
                if len(a) < len(b):
                    a, b = b, a
                distances = range(len(b) + 1)
                for i, ca in enumerate(a):
                    new_distances = [i + 1]
                    for j, cb in enumerate(b):
                        cost = 0 if ca == cb else 1
                        new_distances.append(min(
                            distances[j + 1] + 1,      # 删除
                            new_distances[j] + 1,      # 插入
                            distances[j] + cost        # 替换
                        ))
                    distances = new_distances
                return distances[-1]
        
        sim_matrix = np.zeros((len(new_texts), len(old_texts)))
        for i, new_text in enumerate(new_texts):
            for j, old_text in enumerate(old_texts):
                max_len = max(len(new_text), len(old_text))
                if max_len == 0:
                    sim = 1.0
                else:
                    dist = lev_distance(new_text, old_text)
                    sim = 1.0 - dist / max_len
                sim_matrix[i, j] = sim
        return sim_matrix
    
    def find_best_matches(self, old_paras: List[Paragraph], new_paras: List[Paragraph]) -> Dict[int, Tuple[int, float]]:
        if not old_paras or not new_paras:
            return {}
        
        # 预处理所有段落文本（字符串形式，用于TF-IDF等）
        old_texts = [self.preprocess(p.content) for p in old_paras]
        new_texts = [self.preprocess(p.content) for p in new_paras]
        
        # 调用选定的相似度计算方法
        sim_matrix = self.strategies[self.method](old_texts, new_texts)
        
        # 贪心匹配
        matches = {}
        used_old = set()
        pairs = []
        for i in range(len(new_paras)):
            for j in range(len(old_paras)):
                pairs.append((sim_matrix[i, j], i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        for sim, i, j in pairs:
            if sim >= self.threshold and i not in matches and j not in used_old:
                matches[i] = (j, sim)
                used_old.add(j)
        return matches

class DocumentReader:
    @staticmethod
    def read_txt(file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs
    
    @staticmethod
    def read_docx(file_path: str) -> List[str]:
        if not DOCX_AVAILABLE:
            raise ImportError("请安装 python-docx: pip install python-docx")
        doc = Document(file_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return paragraphs
    
    @staticmethod
    def read(file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            return DocumentReader.read_txt(file_path)
        elif ext == '.docx':
            return DocumentReader.read_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 .txt 和 .docx")

class DocumentComparator:
    COLORS = {
        'HEADER': '\033[95m',
        'ADD': '\033[92m',
        'DELETE': '\033[91m',
        'MODIFY': '\033[93m',
        'UNCHANGED': '\033[0m',
        'BOLD': '\033[1m',
        'INFO': '\033[96m',
        'RESET': '\033[0m'
    }
    
    def __init__(self, similarity_threshold: float = 0.3, method: str = "tfidf"):
        self.threshold = similarity_threshold
        self.method = method
        self.matcher = ParagraphMatcher(similarity_threshold, method)
    
    def parse_paragraphs(self, paragraphs: List[str]) -> List[Paragraph]:
        paras = []
        for idx, content in enumerate(paragraphs):
            para_id = f"段落{idx+1}"
            paras.append(Paragraph(para_id, content, idx))
        return paras
    
    def classify_changes(self, old_paras: List[Paragraph], new_paras: List[Paragraph]) -> List[dict]:
        matches = self.matcher.find_best_matches(old_paras, new_paras)
        changes = []
        used_old = set()
        used_new = set()
        
        for new_idx, (old_idx, sim) in matches.items():
            old_p = old_paras[old_idx]
            new_p = new_paras[new_idx]
            used_old.add(old_idx)
            used_new.add(new_idx)
            
            if sim >= 0.85:
                if old_p.content == new_p.content:
                    change_type = ChangeType.UNCHANGED
                else:
                    change_type = ChangeType.MODIFY
            elif sim >= 0.5:
                change_type = ChangeType.MODIFY
            else:
                old_kw = old_p.get_keywords()
                new_kw = new_p.get_keywords()
                if len(old_kw & new_kw) > 0:
                    change_type = ChangeType.MODIFY
                else:
                    changes.append({'type': ChangeType.DELETE, 'old': old_p, 'new': None, 'similarity': sim})
                    changes.append({'type': ChangeType.ADD, 'old': None, 'new': new_p, 'similarity': sim})
                    continue
            changes.append({'type': change_type, 'old': old_p, 'new': new_p, 'similarity': sim})
        
        for idx, p in enumerate(old_paras):
            if idx not in used_old:
                changes.append({'type': ChangeType.DELETE, 'old': p, 'new': None, 'similarity': 0.0})
        
        for idx, p in enumerate(new_paras):
            if idx not in used_new:
                changes.append({'type': ChangeType.ADD, 'old': None, 'new': p, 'similarity': 0.0})
        
        return changes
    
    def highlight_text_diff(self, old_text: str, new_text: str) -> str:
        if not old_text or not new_text:
            return new_text or old_text
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result.append(new_text[j1:j2])
            elif tag == 'replace':
                result.append(f"{self.COLORS['DELETE']}[删除:{old_text[i1:i2]}]{self.COLORS['RESET']}")
                result.append(f"{self.COLORS['ADD']}[新增:{new_text[j1:j2]}]{self.COLORS['RESET']}")
            elif tag == 'delete':
                result.append(f"{self.COLORS['DELETE']}[删除:{old_text[i1:i2]}]{self.COLORS['RESET']}")
            elif tag == 'insert':
                result.append(f"{self.COLORS['ADD']}[新增:{new_text[j1:j2]}]{self.COLORS['RESET']}")
        return ''.join(result)
    
    def display_changes(self, changes: List[dict], show_details: bool = True):
        for change in changes:
            ct = change['type']
            if ct == ChangeType.DELETE:
                old_p = change['old']
                change['_sort_key'] = (old_p.order, 0)
            elif ct == ChangeType.ADD:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 1)
            elif ct == ChangeType.MODIFY:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 2)
            else:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 3)
        
        changes_sorted = sorted(changes, key=lambda x: x['_sort_key'])
        
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['HEADER']}{'='*80}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['HEADER']}文档版本对比报告（相似度方法: {self.method.upper()}）{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['HEADER']}{'='*80}{self.COLORS['RESET']}\n")
        
        stats = {t: 0 for t in ChangeType}
        
        for change in changes_sorted:
            ct = change['type']
            stats[ct] += 1
            
            if ct == ChangeType.ADD:
                p = change['new']
                print(f"{self.COLORS['ADD']}{self.COLORS['BOLD']}[新增] {p.para_id}{self.COLORS['RESET']}")
                if show_details:
                    preview = p.content[:200] + ('...' if len(p.content) > 200 else '')
                    print(f"{self.COLORS['ADD']}  {preview}{self.COLORS['RESET']}")
            elif ct == ChangeType.DELETE:
                p = change['old']
                print(f"{self.COLORS['DELETE']}{self.COLORS['BOLD']}[删除] {p.para_id}{self.COLORS['RESET']}")
                if show_details:
                    preview = p.content[:200] + ('...' if len(p.content) > 200 else '')
                    print(f"{self.COLORS['DELETE']}  {preview}{self.COLORS['RESET']}")
            elif ct == ChangeType.MODIFY:
                old_p = change['old']
                new_p = change['new']
                sim = change['similarity']
                print(f"{self.COLORS['MODIFY']}{self.COLORS['BOLD']}[修改] {old_p.para_id} -> {new_p.para_id} (相似度 {sim:.2%}){self.COLORS['RESET']}")
                if show_details:
                    diff_text = self.highlight_text_diff(old_p.content, new_p.content)
                    if len(diff_text) > 500:
                        diff_text = diff_text[:500] + '...'
                    print(f"  {diff_text}")
            else:
                p = change['old'] or change['new']
                print(f"{self.COLORS['UNCHANGED']}[未变更] {p.para_id}{self.COLORS['RESET']}")
                if show_details:
                    preview = p.content[:200] + ('...' if len(p.content) > 200 else '')
                    print(f"{self.COLORS['UNCHANGED']}  {preview}{self.COLORS['RESET']}")
            print()
        
        print(f"\n{self.COLORS['INFO']}{self.COLORS['BOLD']}{'='*80}{self.COLORS['RESET']}")
        print(f"{self.COLORS['INFO']}{self.COLORS['BOLD']}统计汇总:{self.COLORS['RESET']}")
        print(f"{self.COLORS['ADD']}  新增: {stats[ChangeType.ADD]} 段{self.COLORS['RESET']}")
        print(f"{self.COLORS['DELETE']}  删除: {stats[ChangeType.DELETE]} 段{self.COLORS['RESET']}")
        print(f"{self.COLORS['MODIFY']}  修改: {stats[ChangeType.MODIFY]} 段{self.COLORS['RESET']}")
        print(f"{self.COLORS['UNCHANGED']}  未变更: {stats[ChangeType.UNCHANGED]} 段{self.COLORS['RESET']}")
        print(f"{self.COLORS['INFO']}{self.COLORS['BOLD']}{'='*80}{self.COLORS['RESET']}\n")
        
    def generate_html_report(self, changes: List[dict], output_file: str = "comparison_report.html", show_details: bool = True):
        """生成 HTML 格式的对比报告"""
        # 先为每个变更添加排序键（与 display_changes 相同）
        for change in changes:
            ct = change['type']
            if ct == ChangeType.DELETE:
                old_p = change['old']
                change['_sort_key'] = (old_p.order, 0)
            elif ct == ChangeType.ADD:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 1)
            elif ct == ChangeType.MODIFY:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 2)
            else:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 3)
        
        type_map = {
            ChangeType.ADD: 'add',
            ChangeType.DELETE: 'delete',
            ChangeType.MODIFY: 'modify',
            ChangeType.UNCHANGED: 'unchanged'
        }
        
        changes_sorted = sorted(changes, key=lambda x: x['_sort_key'])
        
        # 统计
        stats = {t: 0 for t in ChangeType}
        for c in changes_sorted:
            stats[c['type']] += 1
        
        # 构建 HTML
        html = f"""<!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>文档版本对比报告</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            .stats {{
                display: flex;
                gap: 20px;
                margin: 20px 0;
                padding: 15px;
                background: #f0f0f0;
                border-radius: 5px;
            }}
            .stat {{
                flex: 1;
                text-align: center;
            }}
            .stat-number {{
                font-size: 28px;
                font-weight: bold;
                display: block;
            }}
            .stat-add .stat-number {{ color: #4CAF50; }}
            .stat-delete .stat-number {{ color: #f44336; }}
            .stat-modify .stat-number {{ color: #ff9800; }}
            .stat-unchanged .stat-number {{ color: #9e9e9e; }}
            .change-item {{
                margin: 15px 0;
                padding: 15px;
                border-left: 4px solid;
                border-radius: 4px;
                background-color: #fafafa;
            }}
            .change-add {{ border-left-color: #4CAF50; background-color: #e8f5e9; }}
            .change-delete {{ border-left-color: #f44336; background-color: #ffebee; }}
            .change-modify {{ border-left-color: #ff9800; background-color: #fff3e0; }}
            .change-unchanged {{ border-left-color: #9e9e9e; background-color: #f5f5f5; }}
            .badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                color: white;
                font-size: 12px;
                font-weight: bold;
                margin-right: 10px;
            }}
            .badge-add {{ background-color: #4CAF50; }}
            .badge-delete {{ background-color: #f44336; }}
            .badge-modify {{ background-color: #ff9800; }}
            .badge-unchanged {{ background-color: #9e9e9e; }}
            .content {{
                margin-top: 10px;
                padding: 10px;
                background: white;
                border-radius: 4px;
                white-space: pre-wrap;
                font-family: monospace;
                font-size: 14px;
            }}
            .diff-add {{ background-color: #ccffcc; }}
            .diff-delete {{ background-color: #ffcccc; text-decoration: line-through; }}
            hr {{
                margin: 20px 0;
                border: none;
                border-top: 1px solid #ddd;
            }}
            footer {{
                margin-top: 30px;
                text-align: center;
                color: #777;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>📋 文档版本对比报告</h1>
        <div class="stats">
            <div class="stat stat-add"><span class="stat-number">+{stats[ChangeType.ADD]}</span> 新增</div>
            <div class="stat stat-delete"><span class="stat-number">-{stats[ChangeType.DELETE]}</span> 删除</div>
            <div class="stat stat-modify"><span class="stat-number">~{stats[ChangeType.MODIFY]}</span> 修改</div>
            <div class="stat stat-unchanged"><span class="stat-number">={stats[ChangeType.UNCHANGED]}</span> 未变更</div>
        </div>
    """
        
        for change in changes_sorted:
            ct = change['type']
            class_name = f"change-{ct.value}"
            # badge = f"badge-{ct.value}"
            badge = f"badge-{type_map[ct]}"
            if ct == ChangeType.ADD:
                p = change['new']
                html += f"""
        <div class="change-item {class_name}">
            <div><span class="badge {badge}">新增</span> <strong>{p.para_id}</strong></div>
            <div class="content">{p.content.replace('<', '&lt;').replace('>', '&gt;')}</div>
        </div>
    """
            elif ct == ChangeType.DELETE:
                p = change['old']
                html += f"""
        <div class="change-item {class_name}">
            <div><span class="badge {badge}">删除</span> <strong>{p.para_id}</strong></div>
            <div class="content">{p.content.replace('<', '&lt;').replace('>', '&gt;')}</div>
        </div>
    """
            elif ct == ChangeType.MODIFY:
                old_p = change['old']
                new_p = change['new']
                sim = change['similarity']
                # 生成差异对比的简单HTML版本（可复用highlight_text_diff但转换为HTML）
                diff_text = self.highlight_text_diff(old_p.content, new_p.content)
                # 将ANSI颜色转换为HTML span
                diff_html = diff_text.replace(self.COLORS['DELETE'], '<span style="background-color:#ffcccc;text-decoration:line-through;">')
                diff_html = diff_html.replace(self.COLORS['ADD'], '<span style="background-color:#ccffcc;">')
                diff_html = diff_html.replace(self.COLORS['RESET'], '</span>')
                html += f"""
        <div class="change-item {class_name}">
            <div><span class="badge {badge}">修改</span> <strong>{old_p.para_id} → {new_p.para_id}</strong> (相似度 {sim:.2%})</div>
            <div class="content">{diff_html}</div>
        </div>
    """
            else:  # UNCHANGED
                p = change['old'] or change['new']
                if show_details:
                    html += f"""
        <div class="change-item {class_name}">
            <div><span class="badge {badge}">未变更</span> <strong>{p.para_id}</strong></div>
            <div class="content">{p.content.replace('<', '&lt;').replace('>', '&gt;')}</div>
        </div>
    """
        
        html += f"""
        <hr>
        <footer>生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
    </div>
    </body>
    </html>
    """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML 报告已生成: {output_file}")

    def generate_markdown_report(self, changes: List[dict], output_file: str = "comparison_report.md", show_details: bool = True):
        """生成 Markdown 格式的对比报告"""
        # 排序（同HTML）
        for change in changes:
            ct = change['type']
            if ct == ChangeType.DELETE:
                old_p = change['old']
                change['_sort_key'] = (old_p.order, 0)
            elif ct == ChangeType.ADD:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 1)
            elif ct == ChangeType.MODIFY:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 2)
            else:
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 3)
        
        changes_sorted = sorted(changes, key=lambda x: x['_sort_key'])
        
        stats = {t: 0 for t in ChangeType}
        for c in changes_sorted:
            stats[c['type']] += 1
        
        md = f"# 文档版本对比报告\n\n"
        md += f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += "## 统计汇总\n\n"
        md += f"- **新增**: {stats[ChangeType.ADD]} 段\n"
        md += f"- **删除**: {stats[ChangeType.DELETE]} 段\n"
        md += f"- **修改**: {stats[ChangeType.MODIFY]} 段\n"
        md += f"- **未变更**: {stats[ChangeType.UNCHANGED]} 段\n\n"
        md += "---\n\n"
        
        for change in changes_sorted:
            ct = change['type']
            if ct == ChangeType.ADD:
                p = change['new']
                md += f"### ✨ 新增：{p.para_id}\n\n"
                md += f"```\n{p.content}\n```\n\n"
            elif ct == ChangeType.DELETE:
                p = change['old']
                md += f"### ❌ 删除：{p.para_id}\n\n"
                md += f"```\n{p.content}\n```\n\n"
            elif ct == ChangeType.MODIFY:
                old_p = change['old']
                new_p = change['new']
                sim = change['similarity']
                md += f"### 📝 修改：{old_p.para_id} → {new_p.para_id} (相似度 {sim:.2%})\n\n"
                md += "**修改详情**:\n\n"
                # 生成纯文本差异（去除ANSI颜色）
                diff_text = self.highlight_text_diff(old_p.content, new_p.content)
                diff_text = re.sub(r'\x1b\[[0-9;]*m', '', diff_text)  # 移除ANSI
                md += f"```\n{diff_text}\n```\n\n"
            else:  # UNCHANGED
                p = change['old'] or change['new']
                if show_details:
                    md += f"### ✅ 未变更：{p.para_id}\n\n"
                    md += f"```\n{p.content}\n```\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"Markdown 报告已生成: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='段落级别文档对比工具（可切换相似度算法，支持导出报告）')
    parser.add_argument('old_file', help='旧版本文件路径')
    parser.add_argument('new_file', help='新版本文件路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='相似度阈值 (0-1)')
    parser.add_argument('--method', '-m', type=str, default='tfidf',
                        choices=['tfidf', 'bm25', 'jaccard', 'levenshtein'],
                        help='相似度计算方法')
    parser.add_argument('--no-detail', action='store_true', help='不显示详细差异')
    parser.add_argument('--report', '-r', type=str, choices=['html', 'md', 'both'], default=None,
                        help='生成报告格式：html, md, both (同时生成HTML和Markdown)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='报告输出目录（默认当前目录）')
    args = parser.parse_args()
    
    try:
        if not DOCX_AVAILABLE and (args.old_file.endswith('.docx') or args.new_file.endswith('.docx')):
            print("错误: 需要安装 python-docx: pip install python-docx")
            sys.exit(1)
        
        print(f"正在读取文件段落... (相似度方法: {args.method.upper()})")
        old_paragraphs = DocumentReader.read(args.old_file)
        new_paragraphs = DocumentReader.read(args.new_file)
        
        comparator = DocumentComparator(similarity_threshold=args.threshold, method=args.method)
        print("正在解析段落...")
        old_paras = comparator.parse_paragraphs(old_paragraphs)
        new_paras = comparator.parse_paragraphs(new_paragraphs)
        print(f"旧版本: {len(old_paras)} 段")
        print(f"新版本: {len(new_paras)} 段")
        
        print(f"正在对比（阈值 = {args.threshold}）...")
        changes = comparator.classify_changes(old_paras, new_paras)
        
        # 生成报告
        if args.report in ('html', 'both'):
            html_path = os.path.join(args.output_dir, f"comparison_{os.path.basename(args.old_file)}_{os.path.basename(args.new_file)}.html")
            comparator.generate_html_report(changes, html_path, show_details=not args.no_detail)
        if args.report in ('md', 'both'):
            md_path = os.path.join(args.output_dir, f"comparison_{os.path.basename(args.old_file)}_{os.path.basename(args.new_file)}.md")
            comparator.generate_markdown_report(changes, md_path, show_details=not args.no_detail)
        
        # 仍然显示终端输出
        comparator.display_changes(changes, show_details=not args.no_detail)
        
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()