"""
This module provides a dynamic Markdown heading-based splitter.
It splits Markdown documents along heading boundaries (h1-h6) and recursively splits
sections that exceed the character limit (default: 2000 characters), falls back to subheadings,
and prepends hierarchical heading paths as context.
"""

import re
import logging
from typing import List, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MarkdownSection:
    def __init__(self, level: int, title: str, heading_line: str = ""):
        self.level = level  # 0 for root/preamble, 1 for #, 2 for ##, etc.
        self.title = title
        self.heading_line = heading_line
        self.content_lines = []
        self.children = []

def parse_markdown_to_tree(text: str) -> MarkdownSection:
    root = MarkdownSection(level=0, title="Root")
    stack = [root]
    
    for line in text.split("\n"):
        stripped = line.strip()
        header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            
            new_section = MarkdownSection(level=level, title=title, heading_line=line)
            
            # Pop stack until we find a section with level < level
            while stack and stack[-1].level >= level:
                stack.pop()
                
            if stack:
                stack[-1].children.append(new_section)
                stack.append(new_section)
            else:
                root.children.append(new_section)
                stack = [root, new_section]
        else:
            stack[-1].content_lines.append(line)
            
    return root

def get_section_text(section: MarkdownSection) -> str:
    parts = []
    if section.heading_line:
        parts.append(section.heading_line)
    if section.content_lines:
        parts.extend(section.content_lines)
    for child in section.children:
        parts.append(get_section_text(child))
    return "\n".join(parts)

def chunk_section(
    section: MarkdownSection,
    max_size: int = 2000,
    overlap: int = 200,
    headings_path: List[str] = None
) -> List[dict]:
    if headings_path is None:
        headings_path = []
        
    current_path = list(headings_path)
    if section.level > 0:
        current_path.append(section.title)
        
    section_text = get_section_text(section)
    if len(section_text) <= max_size:
        return [{"text": section_text, "headings": current_path}]
        
    # Section text exceeds max_size, so we split it.
    # First extract the preamble text of this section (its own text before any subheadings).
    preamble_parts = []
    if section.heading_line:
        preamble_parts.append(section.heading_line)
    if section.content_lines:
        preamble_parts.extend(section.content_lines)
    preamble_text = "\n".join(preamble_parts)
    
    chunks = []
    if len(preamble_text) > max_size:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=overlap)
        for t in splitter.split_text(preamble_text):
            chunks.append({"text": t, "headings": current_path})
    elif preamble_text.strip():
        chunks.append({"text": preamble_text, "headings": current_path})
        
    # Process child sub-sections recursively
    for child in section.children:
        child_chunks = chunk_section(child, max_size, overlap, current_path)
        for cc in child_chunks:
            if not cc["text"].strip():
                continue
            # Merge with the last chunk if they share the exact same heading path and fit in max_size
            if chunks and chunks[-1]["headings"] == cc["headings"] and len(chunks[-1]["text"]) + len(cc["text"]) + 1 <= max_size:
                chunks[-1]["text"] = chunks[-1]["text"] + "\n" + cc["text"]
            else:
                chunks.append(cc)
                
    return chunks

def chunk_markdown_by_headings(
    text: str,
    filename: str,
    source: str,
    max_size: int = 2000,
    overlap: int = 200
) -> List[Document]:
    """
    Chunks a Markdown document into section/heading-based pieces.
    Prepends section hierarchical heading paths as context.
    """
    root = parse_markdown_to_tree(text)
    raw_chunks = chunk_section(root, max_size, overlap)
    
    docs = []
    for rc in raw_chunks:
        chunk_text = rc["text"].strip()
        if not chunk_text:
            continue
            
        headings = rc["headings"]
        prefix = ""
        if headings:
            prefix = f"Context: {' > '.join(headings)}\n"
            
        full_content = f"{prefix}{chunk_text}".strip()
        docs.append(Document(
            page_content=full_content,
            metadata={"filename": filename, "source": source}
        ))
        
    logger.info(f"Created {len(docs)} heading-based markdown chunks with a max limit of {max_size}.")
    return docs
