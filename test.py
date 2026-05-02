#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import difflib
import argparse
import sys
from typing import List, Tuple, Dict
from enum import Enum
import os

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

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
    def __init__(self, similarity_threshold: float = 0.3):
        self.threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=(1, 2),
            min_df=1,
        )
        # 条款编号正则模式（用于移除）
        self.clause_patterns = [
            re.compile(r'第[一二三四五六七八九十百千万\d]+条'),   # 第X条
            re.compile(r'第[一二三四五六七八九十百千万\d]+章'),   # 第X章
            re.compile(r'^\d+[\.、]'),                           # 行首的 1. 或 1、
            re.compile(r'^[（(]\d+[）)]'),                       # (1) 或 （1）
            re.compile(r'^\d+、'),                               # 1、
            re.compile(r'^\d+\s+'),                              # 1 后跟空格
        ]
    
    def preprocess(self, text: str) -> str:
        # 1. 移除条款编号
        for pattern in self.clause_patterns:
            text = pattern.sub('', text)
        # 2. 移除标点符号，只保留中英文和空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 3. 合并多余空格，去除首尾空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def find_best_matches(self, old_paras: List[Paragraph], new_paras: List[Paragraph]) -> Dict[int, Tuple[int, float]]:
        if not old_paras or not new_paras:
            return {}
        old_texts = [self.preprocess(p.content) for p in old_paras]
        new_texts = [self.preprocess(p.content) for p in new_paras]
        
        self.vectorizer.fit(old_texts)
        old_matrix = self.vectorizer.transform(old_texts)
        new_matrix = self.vectorizer.transform(new_texts)
        
        sim_matrix = cosine_similarity(new_matrix, old_matrix)
        
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
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.threshold = similarity_threshold
        self.matcher = ParagraphMatcher(similarity_threshold)
    
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
        # 构造排序键
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
            else:  # UNCHANGED
                new_p = change['new']
                change['_sort_key'] = (new_p.order, 3)
        
        changes_sorted = sorted(changes, key=lambda x: x['_sort_key'])
        
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['HEADER']}{'='*80}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['HEADER']}文档版本对比报告（按段落顺序）{self.COLORS['RESET']}")
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
            else:  # UNCHANGED
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

def main():
    parser = argparse.ArgumentParser(description='段落级别文档对比工具（按原文顺序，无emoji）')
    parser.add_argument('old_file', help='旧版本文件路径')
    parser.add_argument('new_file', help='新版本文件路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='相似度阈值 (0-1)')
    parser.add_argument('--no-detail', action='store_true', help='不显示详细差异')
    args = parser.parse_args()
    
    try:
        if not DOCX_AVAILABLE and (args.old_file.endswith('.docx') or args.new_file.endswith('.docx')):
            print("错误: 需要安装 python-docx: pip install python-docx")
            sys.exit(1)
        
        print("正在读取文件段落...")
        old_paragraphs = DocumentReader.read(args.old_file)
        new_paragraphs = DocumentReader.read(args.new_file)
        
        comparator = DocumentComparator(similarity_threshold=args.threshold)
        print("正在解析段落...")
        old_paras = comparator.parse_paragraphs(old_paragraphs)
        new_paras = comparator.parse_paragraphs(new_paragraphs)
        print(f"旧版本: {len(old_paras)} 段")
        print(f"新版本: {len(new_paras)} 段")
        
        print(f"正在对比（阈值 = {args.threshold}）...")
        changes = comparator.classify_changes(old_paras, new_paras)
        
        comparator.display_changes(changes, show_details=not args.no_detail)
        
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()