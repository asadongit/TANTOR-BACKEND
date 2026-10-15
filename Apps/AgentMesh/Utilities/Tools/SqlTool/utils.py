import logging
import re
import sqlparse
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

def extract_tables_from_ast(token_list, from_seen=False) -> List[str]:
    """Recursively extracts table names from sqlparse tokens in FROM/JOIN clauses."""
    tables = []
    
    for token in token_list:
        if from_seen:
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.DML, sqlparse.tokens.DDL) and token.value.upper() not in ('AS',):
                from_seen = False
            elif isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    real_name = identifier.get_real_name()
                    if real_name:
                        tables.append(real_name)
                from_seen = False
                continue
            elif isinstance(token, sqlparse.sql.Identifier):
                real_name = token.get_real_name()
                if real_name:
                    tables.append(real_name)
                from_seen = False
                continue
            elif token.ttype == sqlparse.tokens.Punctuation and token.value == ';':
                from_seen = False

        if token.ttype is sqlparse.tokens.Keyword:
            if token.value.upper() in ('FROM', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'INTO'):
                from_seen = True
            elif token.value.upper() not in ('AS',):
                from_seen = False
        elif token.ttype is sqlparse.tokens.DML and token.value.upper() == 'UPDATE':
            from_seen = True
        elif token.is_group:
            tables.extend(extract_tables_from_ast(token.tokens, from_seen))
                
    return [t.lower() for t in tables if t]

def is_query_safe(query: str, allowed_table: str) -> Tuple[bool, str]:
    """
    Security Validator: Parser-based check.
    Blocks destructive DDL/DML and enforces whitelisted table SELECT access.
    """
    try:
        stripped = sqlparse.format(query, strip_comments=True).strip()
        parsed = sqlparse.parse(stripped)
       
        if not parsed:
            return False, "Empty query generated."
        if len(parsed) > 1:
            return False, "Multiple statements detected. Only one SELECT allowed."
       
        stmt = parsed[0]
        if stmt.get_type() != 'SELECT':
            return False, "Only SELECT statements are allowed."
       
        forbidden = {'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'INTO', 'OUTFILE'}
        for token in stmt.flatten():
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.DDL, sqlparse.tokens.DML):
                if token.normalized in forbidden:
                    return False, f"Forbidden keyword detected: {token.normalized}"
       
        if allowed_table and allowed_table != "*" and allowed_table.lower() != "any":
            allowed_list = [t.strip().lower() for t in allowed_table.split(",") if t.strip()]
            # Always allow dual dummy table (Oracle)
            allowed_list.append("dual")
            
            # Extract actual table names from the parsed AST
            cleaned_referenced = extract_tables_from_ast(stmt.tokens)
            
            # Ensure every referenced table is in the whitelist
            for ref_table in cleaned_referenced:
                if ref_table not in allowed_list:
                    return False, f"Query references unauthorized table '{ref_table}'. Whitelist: {allowed_table}"
            
            if not cleaned_referenced:
                return False, f"Query must reference at least one table from: {allowed_table}"
           
        return True, "Safe"
    except Exception as e:
        return False, f"Safety check failed with error: {str(e)}"

def is_write_query_safe(query: str, allowed_table: str) -> Tuple[bool, str]:
    """
    Security Validator: Parser-based check for mutating queries.
    Blocks destructive DDL and enforces whitelisted table INSERT/UPDATE/DELETE access.
    """
    try:
        stripped = sqlparse.format(query, strip_comments=True).strip()
        parsed = sqlparse.parse(stripped)
       
        if not parsed:
            return False, "Empty query generated."
        if len(parsed) > 1:
            return False, "Multiple statements detected. Only one statement allowed."
       
        stmt = parsed[0]
        stmt_type = stmt.get_type()
        if stmt_type not in ('SELECT', 'INSERT', 'UPDATE', 'DELETE'):
            return False, f"Only SELECT, INSERT, UPDATE, and DELETE statements are allowed. Got {stmt_type}."
       
        forbidden = {'DROP', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'OUTFILE'}
        for token in stmt.flatten():
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.DDL, sqlparse.tokens.DML):
                if token.normalized in forbidden:
                    return False, f"Forbidden keyword detected: {token.normalized}"
       
        if allowed_table and allowed_table != "*" and allowed_table.lower() != "any":
            allowed_list = [t.strip().lower() for t in allowed_table.split(",") if t.strip()]
            
            # Extract actual table names from the parsed AST
            cleaned_referenced = extract_tables_from_ast(stmt.tokens)
            
            # Ensure every referenced table is in the whitelist
            for ref_table in cleaned_referenced:
                if ref_table not in allowed_list:
                    return False, f"Query references unauthorized table '{ref_table}'. Whitelist: {allowed_table}"
            
            if not cleaned_referenced:
                return False, f"Query must reference at least one table from: {allowed_table}"
           
        return True, "Safe"
    except Exception as e:
        return False, f"Safety check failed with error: {str(e)}"

def attempt_sql_self_heal(sql: str, error_msg: str, schema_name: str, table_name: str, column_names: List[str]) -> Optional[str]:
    """
    Advanced SQL self-healing. Corrects common formatting errors for small models.
    """
    correct_full = f'"{schema_name}"."{table_name}"'
    col_names_lower = {c.lower() for c in column_names}
   
    # Protect literals
    literals = []
    def literal_replacer(match):
        literals.append(match.group(0))
        return f" __LITERAL_{len(literals)-1}__ "
       
    safe_sql = re.sub(r"'(?:''|[^'])*'", literal_replacer, sql)
    safe_sql = safe_sql.replace('`', '"')
   
    # Strip hallucinated JOINs
    safe_sql = re.sub(
        r'\b(INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN\s+.*?\bON\s+.*?(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)',
        '', safe_sql, flags=re.IGNORECASE | re.DOTALL
    )
   
    # Nuclear FROM clause rewrite
    if correct_full not in safe_sql and table_name.lower() not in safe_sql.lower():
        safe_sql = re.sub(
            r'\bFROM\s+["\w.]+(?:\s+(?:AS\s+)?[a-zA-Z_]\w*)?',
            f'FROM {correct_full}',
            safe_sql, count=1, flags=re.IGNORECASE
        )
       
    def strip_alias_prefix(match):
        prefix = match.group(1).strip('"')
        column = match.group(2).strip('"')
        if prefix.lower() == schema_name.lower():
             return f'"{schema_name}"."{column}"'
        if prefix.lower() == table_name.lower():
             return f'"{table_name}"."{column}"'
        return f'"{column}"'
   
    safe_sql = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*|"[a-zA-Z0-9_]+")\.([a-zA-Z_][a-zA-Z0-9_]*|"[a-zA-Z0-9_]+")\b', strip_alias_prefix, safe_sql)
   
    for col in column_names:
        pattern = rf'"?\b{re.escape(col)}\b"?'
        if re.search(pattern, safe_sql, re.IGNORECASE):
            safe_sql = re.sub(pattern, f'"{col}"', safe_sql, flags=re.IGNORECASE)
           
    # Table reference normalization
    loose_pattern = rf'({re.escape(schema_name)}|"{re.escape(schema_name)}")\s*[\.\s"`]+\s*({re.escape(schema_name)}|"{re.escape(schema_name)}")\s*[\.\s"`]+\s*({re.escape(table_name)}|"{re.escape(table_name)}")'
    safe_sql = re.sub(loose_pattern, correct_full, safe_sql, flags=re.IGNORECASE)
 
    if correct_full not in safe_sql:
        escaped_table = re.escape(table_name)
        pattern = rf'(?:(?:public|{re.escape(schema_name)})\b"?\.?)?"?{escaped_table}"?'
        safe_sql = re.sub(pattern, correct_full, safe_sql, flags=re.IGNORECASE)
 
    if '"column_name"' in safe_sql and 'column_name' not in column_names:
        safe_sql = safe_sql.replace('"column_name"', '*')
 
    # Identifier validation vs schema
    reserved_sql = "SELECT|FROM|WHERE|GROUP|BY|ORDER|LIMIT|AS|COUNT|SUM|AVG|MIN|MAX|DISTINCT|CASE|WHEN|THEN|ELSE|END|AND|OR|NOT|IS|NULL|DESC|ASC|IN|ON|BETWEEN|HAVING|EXISTS|LIKE|ILIKE|__LITERAL_\\d+"
    words = re.findall(rf'\b(?!(?:{reserved_sql})\b)([a-zA-Z_]\w*)\b', safe_sql, re.IGNORECASE)
   
    for word in set(words):
        word_lower = word.lower()
        if word_lower in col_names_lower or word_lower in (schema_name.lower(), table_name.lower()):
             continue
             
        best_match = None
        for real_col in column_names:
            if word_lower in real_col.lower() or real_col.lower() in word_lower:
                best_match = real_col
                break
       
        if best_match:
            safe_sql = re.sub(rf'\b{re.escape(word)}\b', f'"{best_match}"', safe_sql)
 
    # Handle redundant DISTINCT in GROUP BY
    if "select distinct" in safe_sql.lower() and "group by" in safe_sql.lower():
        safe_sql = re.sub(r'SELECT DISTINCT', 'SELECT', safe_sql, flags=re.IGNORECASE)
 
    safe_sql = re.sub(r'""+', '"', safe_sql)
    safe_sql = re.sub(r'\s+', ' ', safe_sql).strip()
   
    for i, literal in enumerate(literals):
        safe_sql = safe_sql.replace(f"__LITERAL_{i}__", literal)
 
    if safe_sql.strip() != sql.strip():
        logger.info(f"🔧 SQL Self-Healed: {safe_sql}")
        return safe_sql
   
    return None
