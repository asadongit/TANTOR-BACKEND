"""
This module provides a Markdown header-aware and table-context-preserving chunker.
It tracks the active Markdown section headers and table headers, injecting them into
chunk contexts to preserve hierarchical data relationships during extraction.
"""

import re
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def chunk_markdown_with_context(
    text: str,
    filename: str,
    source: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Chunks a Markdown document into smaller pieces.
    - Tracks active # level headers and prepends them as context to every chunk.
    - Tracks active markdown table headers. If a chunk contains table rows but is split
      from its header, it auto-prepends the table's header and separator line.
    """
    lines = text.split("\n")
    chunks = []
    
    current_headers = {1: None, 2: None, 3: None, 4: None}
    current_table_header = None
    current_table_separator = None
    in_table = False
    
    current_chunk_lines = []
    current_chunk_char_count = 0
    
    def get_context_prefix():
        active = []
        for lvl in sorted(current_headers.keys()):
            if current_headers[lvl]:
                active.append(current_headers[lvl])
        prefix = ""
        if active:
            prefix += f"Context: {' > '.join(active)}\n"
        return prefix

    for line in lines:
        stripped = line.strip()
        
        # 1. Track Markdown Headers
        header_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_headers[level] = title
            # Clear deeper subheaders when a higher header changes
            for l in range(level + 1, 5):
                current_headers[l] = None
                
        # 2. Track Markdown Tables
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if is_table_row:
            if not in_table:
                # Potential new table header
                in_table = True
                current_table_header = line
                current_table_separator = None
            elif not current_table_separator and re.match(r"^\|[\s\-\|:]+\|$", stripped):
                current_table_separator = line
        else:
            in_table = False
            
        # Estimate size if we append this line
        line_len = len(line) + 1
        
        # If adding this line exceeds chunk_size, we flush the current chunk
        if current_chunk_char_count + line_len > chunk_size and current_chunk_lines:
            # Construct chunk text
            context = get_context_prefix()
            chunk_body = "\n".join(current_chunk_lines)
            
            # If the chunk body contains table rows but is missing table headers, prepend them
            if in_table and current_table_header and current_table_separator:
                if current_table_header not in chunk_body:
                    chunk_body = f"{current_table_header}\n{current_table_separator}\n{chunk_body}"
            
            full_content = f"{context}{chunk_body}".strip()
            chunks.append(Document(
                page_content=full_content,
                metadata={"filename": filename, "source": source}
            ))
            
            # Keep overlap lines
            overlap_char_count = 0
            overlap_lines = []
            for ol in reversed(current_chunk_lines):
                if overlap_char_count + len(ol) + 1 > chunk_overlap:
                    break
                overlap_lines.insert(0, ol)
                overlap_char_count += len(ol) + 1
                
            current_chunk_lines = overlap_lines
            current_chunk_char_count = overlap_char_count
            
        current_chunk_lines.append(line)
        current_chunk_char_count += line_len
        
    # Flush remaining lines
    if current_chunk_lines:
        context = get_context_prefix()
        chunk_body = "\n".join(current_chunk_lines)
        if in_table and current_table_header and current_table_separator:
            if current_table_header not in chunk_body:
                chunk_body = f"{current_table_header}\n{current_table_separator}\n{chunk_body}"
                
        full_content = f"{context}{chunk_body}".strip()
        chunks.append(Document(
            page_content=full_content,
            metadata={"filename": filename, "source": source}
        ))
        
    logger.info(f"Created {len(chunks)} header-aware and table-context-preserving chunks.")
    return chunks
