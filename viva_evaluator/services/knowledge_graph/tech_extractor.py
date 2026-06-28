"""
Tech extractor — identifies technologies used in a submission so that
the brief drafter can produce per-technology briefs.

INPUT SOURCES (combined for coverage):
    1. AST imports (from code_indexer): direct evidence of what's USED
    2. Report text mentions: catches things absent from code (e.g.,
       a planned database) plus protocol/standard mentions

OUTPUT:
    sorted list of technology names with metadata about WHERE each was found.
    [
        {'name': 'PostgreSQL', 'sources': ['report'], 'category': 'database'},
        {'name': 'FAISS',      'sources': ['imports', 'report'], 'category': 'library'},
        ...
    ]
"""

import logging
import re
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Curated technology dictionary.
#
# Maps canonical NAME → (regex of mention forms, category).
# Add to this list freely — it directly drives which briefs get drafted.
# =============================================================================

TECH_DICTIONARY = [
    # ---- Databases ----------------------------------------------------------
    ('PostgreSQL',  re.compile(r'\b(?:postgres(?:ql)?|psycopg2?)\b', re.I), 'database'),
    ('MySQL',       re.compile(r'\bmysql\b', re.I), 'database'),
    ('SQLite',      re.compile(r'\bsqlite\b', re.I), 'database'),
    ('MongoDB',     re.compile(r'\bmongo(?:db)?\b|\bpymongo\b', re.I), 'database'),
    ('Redis',       re.compile(r'\bredis\b', re.I), 'database'),
    ('Firebase',    re.compile(r'\bfirebase\b|\bfirestore\b', re.I), 'database'),
    ('DynamoDB',    re.compile(r'\bdynamodb\b', re.I), 'database'),
    ('Cassandra',   re.compile(r'\bcassandra\b', re.I), 'database'),

    # ---- Web frameworks -----------------------------------------------------
    ('Django',      re.compile(r'\bdjango\b', re.I), 'web_framework'),
    ('FastAPI',     re.compile(r'\bfastapi\b', re.I), 'web_framework'),
    ('Flask',       re.compile(r'\bflask\b', re.I), 'web_framework'),
    ('Express',     re.compile(r'\bexpress(?:js)?\b', re.I), 'web_framework'),
    ('Spring',      re.compile(r'\bspring(?:\s?boot)?\b', re.I), 'web_framework'),
    ('Next.js',     re.compile(r'\bnext\.?js\b', re.I), 'web_framework'),
    ('NestJS',      re.compile(r'\bnest\.?js\b', re.I), 'web_framework'),

    # ---- Frontend -----------------------------------------------------------
    ('React',       re.compile(r'\breact(?:js)?\b', re.I), 'frontend'),
    ('Vue',         re.compile(r'\bvue\.?js\b', re.I), 'frontend'),
    ('Angular',     re.compile(r'\bangular\b', re.I), 'frontend'),
    ('Svelte',      re.compile(r'\bsvelte\b', re.I), 'frontend'),

    # ---- ML / AI ------------------------------------------------------------
    ('TensorFlow',  re.compile(r'\btensorflow\b|\btf\.', re.I), 'ml_framework'),
    ('PyTorch',     re.compile(r'\bpytorch\b|\btorch\b', re.I), 'ml_framework'),
    ('scikit-learn', re.compile(r'\bscikit[\s-]?learn\b|\bsklearn\b', re.I), 'ml_framework'),
    ('Hugging Face', re.compile(r'\bhuggingface\b|\btransformers\b', re.I), 'ml_framework'),
    ('OpenAI',      re.compile(r'\bopenai\b|\bgpt-?[34]\b', re.I), 'llm_provider'),
    ('Gemini',      re.compile(r'\bgemini\b|\bgoogle\s?ai\b|\bgenai\b', re.I), 'llm_provider'),
    ('Anthropic',   re.compile(r'\banthropic\b|\bclaude\b', re.I), 'llm_provider'),
    ('FAISS',       re.compile(r'\bfaiss\b', re.I), 'vector_db'),
    ('Pinecone',    re.compile(r'\bpinecone\b', re.I), 'vector_db'),
    ('ChromaDB',    re.compile(r'\bchroma(?:db)?\b', re.I), 'vector_db'),
    ('SBERT',       re.compile(r'\bsbert\b|\bsentence[\s-]?transformers\b', re.I), 'ml_library'),

    # ---- Auth / security ----------------------------------------------------
    ('JWT',         re.compile(r'\bjwt\b|\bjson\s?web\s?token\b', re.I), 'auth'),
    ('OAuth',       re.compile(r'\boauth\b', re.I), 'auth'),
    ('AES',         re.compile(r'\baes(?:[\s-]?(?:128|192|256|gcm|cbc))?\b', re.I), 'crypto'),
    ('RSA',         re.compile(r'\brsa\b', re.I), 'crypto'),
    ('TLS',         re.compile(r'\btls\b|\bssl\b|\bhttps\b', re.I), 'crypto'),

    # ---- Cloud / infra -----------------------------------------------------
    ('Docker',      re.compile(r'\bdocker(?:file|compose)?\b', re.I), 'infra'),
    ('Kubernetes',  re.compile(r'\bkubernetes\b|\bk8s\b', re.I), 'infra'),
    ('AWS',         re.compile(r'\baws\b|\bamazon\s+web\s+services\b', re.I), 'cloud'),
    ('Azure',       re.compile(r'\bazure\b', re.I), 'cloud'),
    ('Cloudinary',  re.compile(r'\bcloudinary\b', re.I), 'cloud'),
    ('GCP',         re.compile(r'\bgcp\b|\bgoogle\s+cloud\b', re.I), 'cloud'),
]


# =============================================================================
# Public API
# =============================================================================

def extract_technologies(
    imports_seen: Optional[Set[str]] = None,
    report_chunks: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Identify which technologies appear in the submission.

    Args:
        imports_seen: from code_indexer's index_code_repo (set of import names)
        report_chunks: from FAISS-indexed report chunks

    Returns:
        Sorted list of tech dicts with provenance:
            {'name': '...', 'sources': ['imports'|'report'], 'category': '...'}
    """
    found: Dict[str, Dict] = {}

    imports_text = ' '.join(imports_seen or [])

    # Combine all report chunk text into one searchable blob (capped)
    report_text_parts: List[str] = []
    for ch in (report_chunks or []):
        if ch.get('source') in ('report', 'figure'):
            report_text_parts.append(ch.get('text', ''))
        if sum(len(p) for p in report_text_parts) > 50000:
            break
    report_text = ' '.join(report_text_parts)

    for name, pattern, category in TECH_DICTIONARY:
        sources: List[str] = []
        if imports_text and pattern.search(imports_text):
            sources.append('imports')
        if report_text and pattern.search(report_text):
            sources.append('report')

        if not sources:
            continue

        # First match wins; if seen again from another source, just add source
        if name in found:
            for s in sources:
                if s not in found[name]['sources']:
                    found[name]['sources'].append(s)
        else:
            found[name] = {
                'name':     name,
                'sources':  sources,
                'category': category,
            }

    result = sorted(found.values(), key=lambda t: (t['category'], t['name']))
    logger.info(
        'tech_extractor: found %d technologies (imports=%d, report_chars=%d)',
        len(result),
        len(imports_seen or []),
        len(report_text),
    )
    return result
