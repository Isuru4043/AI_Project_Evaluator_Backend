"""
Contradiction detector — produces CONTRADICTS_CODE edges by cross-referencing
SonarQube findings with claims in the student's report.

THIS IS THE WEEK 3 NOVELTY CONTRIBUTION.

EXAMPLE:
    Sonar finding: "Hardcoded credentials in auth/login.py line 42"
    Report claim: "Our system implements secure authentication"
        ↓
    CONTRADICTS_CODE edge:
        source: 'hardcoded_credentials'
        target: 'secure authentication'
        severity: high

The Strategist (Week 5) treats CONTRADICTS_CODE edges as highest priority
and forces the question: "Your report claims X — but your code does Y.
How do you reconcile that?"

DESIGN:
    SonarQube findings are mapped to a fixed set of CLAIM PATTERNS.
    Each pattern is a regex that, if matched against any report chunk,
    indicates the student CLAIMED something the code contradicts.

We deliberately keep this rule-based (not LLM-based) for Week 3 because:
    - Deterministic and auditable
    - No extra LLM cost during indexing
    - Any false positive surfaces in the post-viva audit and the examiner
      can dismiss it (Week 7).
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Mapping: SonarQube finding pattern → claim regex
#
# Each entry says: "if SonarQube reports finding F, look in the report for
# claim C. If found, that's a contradiction."
#
# Sonar rule keys come from sonar-server's "rules" — common ones listed below.
# Free-form severity-level patterns also catch findings whose specific rule
# we don't enumerate but whose category we do.
# =============================================================================

CONTRADICTION_RULES = [
    # ---- Authentication / credential issues -----------------------------
    {
        'finding_id':    'hardcoded_credentials',
        'sonar_match':   re.compile(r'(?:hardcoded.*(?:password|secret|api[ _-]?key|credential|token))|HardcodedSecret', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:secure|robust|strong|safe)\s+(?:authentication|auth|login|access\s+control)\b', re.IGNORECASE),
        'severity':      'high',
    },
    {
        'finding_id':    'weak_password_storage',
        'sonar_match':   re.compile(r'(?:plaintext.*password|password.*plain[\s_-]?text|md5.*password|sha1.*password)', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:secure|safe|encrypted)\s+password\b|\bpassword(?:s)?\s+(?:are\s+)?(?:stored\s+)?(?:hashed|encrypted)\b', re.IGNORECASE),
        'severity':      'high',
    },

    # ---- Encryption / cryptography --------------------------------------
    {
        'finding_id':    'weak_cryptography',
        'sonar_match':   re.compile(r'(?:weak\s+(?:cipher|algorithm)|md5|sha1|des(?:[\s_-]?(?:cbc|ecb))?|rc4)', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:strong|modern|industry[ -]standard|aes|sha-?256)\s+(?:encryption|cryptography|cipher)\b', re.IGNORECASE),
        'severity':      'high',
    },
    {
        'finding_id':    'insecure_random',
        'sonar_match':   re.compile(r'(?:insecure.*random|math\.random.*(?:crypto|key|token)|java\.util\.Random.*(?:crypto|key))', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:cryptographically\s+secure|secure\s+random|csprng)\b', re.IGNORECASE),
        'severity':      'high',
    },

    # ---- Input validation / injection ----------------------------------
    {
        'finding_id':    'sql_injection_risk',
        'sonar_match':   re.compile(r'(?:sql\s+injection|tainted\s+sql|S2077|S3649)', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:prevents?|protect(?:s|ed)?\s+against|safe\s+from)\s+(?:sql\s+)?injection\b|\bparameterized\s+quer(?:y|ies)\b', re.IGNORECASE),
        'severity':      'high',
    },
    {
        'finding_id':    'xss_risk',
        'sonar_match':   re.compile(r'(?:xss|cross[\s-]?site\s+scripting|tainted\s+html|S5131)', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:prevents?|protect(?:s|ed)?\s+against|safe\s+from)\s+xss\b|\b(?:input\s+)?sanitiz(?:ed|ation)\b', re.IGNORECASE),
        'severity':      'high',
    },

    # ---- TLS / transport security --------------------------------------
    {
        'finding_id':    'insecure_tls',
        'sonar_match':   re.compile(r'(?:tls\s*1\.0|sslv[23]|disable\s+ssl|verify[_\s]?(?:ssl|cert)\s*=\s*false|trustall|insecure[_\s]?skip[_\s]?verify)', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:tls\s*1\.[23]|secure\s+(?:transport|connection|tls|https)|encrypted\s+(?:in\s+)?transit)\b', re.IGNORECASE),
        'severity':      'high',
    },

    # ---- Coverage / testing claims --------------------------------------
    {
        'finding_id':    'low_test_coverage',
        'sonar_match':   re.compile(r'^coverage_low$', re.IGNORECASE),  # filled by metric check, not rule key
        'claim_match':   re.compile(r'\b(?:thorough|comprehensive|extensive|high)\s+(?:test\s+)?coverage\b|\bunit\s+tests?\s+cover(?:s|ed)?\b', re.IGNORECASE),
        'severity':      'medium',
    },

    # ---- Reliability / error handling ----------------------------------
    {
        'finding_id':    'unhandled_exceptions',
        'sonar_match':   re.compile(r'(?:catch\s+all|broad\s+except|except\s*:|swallow.*exception|empty\s+catch)', re.IGNORECASE),
        'claim_match':   re.compile(r'\b(?:robust|graceful|comprehensive)\s+error\s+handling\b', re.IGNORECASE),
        'severity':      'medium',
    },
]


# =============================================================================
# Public API
# =============================================================================

def detect_contradictions(
    code_submission,
    report_chunks: List[Dict],
) -> List[Dict]:
    """
    Compare SonarQube findings to claims in the report.

    Returns:
        [
            {
                'code_finding':   'hardcoded_credentials',
                'report_claim':   'secure authentication',
                'severity':       'high',
                'finding_detail': 'Sonar rule pythonsecurity:S2068 ...',
                'claim_excerpt':  '...sentence from the report containing the claim...',
            },
            ...
        ]
    """
    if not code_submission or not report_chunks:
        return []

    sonar_findings = _collect_sonar_findings(code_submission)
    if not sonar_findings:
        logger.info('Contradiction detector: no Sonar findings to check.')
        return []

    contradictions: List[Dict] = []
    seen: set = set()

    for rule in CONTRADICTION_RULES:
        # Step 1: Does any Sonar finding match this rule?
        matched_findings = [
            f for f in sonar_findings if rule['sonar_match'].search(f['text'])
        ]
        if not matched_findings:
            continue

        # Step 2: Is the corresponding claim present in any report chunk?
        claim_excerpt = _find_claim_excerpt(report_chunks, rule['claim_match'])
        if not claim_excerpt:
            continue

        # Step 3: Record the contradiction (one per rule, even if many findings)
        finding_detail = matched_findings[0]['text'][:300]
        finding_node = rule['finding_id']
        claim_node = _claim_node_label(claim_excerpt)

        key = (finding_node, claim_node)
        if key in seen:
            continue
        seen.add(key)

        contradictions.append({
            'code_finding':   finding_node,
            'report_claim':   claim_node,
            'severity':       rule['severity'],
            'finding_detail': finding_detail,
            'claim_excerpt':  claim_excerpt[:400],
        })

    logger.info(
        'Contradiction detector: %d CONTRADICTS_CODE edges found.',
        len(contradictions),
    )
    return contradictions


# =============================================================================
# Internals
# =============================================================================

def _collect_sonar_findings(code_submission) -> List[Dict]:
    """
    Gather all sonar findings as searchable text strings.
    Each item is {'rule': '...', 'severity': '...', 'text': '<full searchable text>'}.

    Also synthesizes a 'coverage_low' pseudo-finding when measured coverage
    is below the configured threshold.
    """
    summary = getattr(code_submission, 'sonar_summary', None) or {}
    findings: List[Dict] = []

    # Real issues from Sonar
    for issue in summary.get('issues', []) or []:
        rule = issue.get('rule', '')
        message = issue.get('message', '')
        component = issue.get('component', '')
        severity = issue.get('severity', 'INFO')
        # Combine into one searchable text
        text = f"{rule} {message} {component}"
        findings.append({
            'rule':     rule,
            'severity': severity,
            'text':     text,
        })

    # Pseudo-finding for low coverage (rule-keyed so detector can match)
    measures = summary.get('measures', [])
    coverage = _measure_value(measures, 'coverage')
    if coverage is not None:
        try:
            cov_value = float(coverage)
            if cov_value < 50.0:
                findings.append({
                    'rule':     'coverage_low',
                    'severity': 'INFO',
                    'text':     f'coverage_low (measured {cov_value}%)',
                })
        except (TypeError, ValueError):
            pass

    return findings


def _measure_value(measures: list, metric_key: str):
    for m in measures or []:
        if m.get('metric') == metric_key:
            return m.get('value')
    return None


def _find_claim_excerpt(report_chunks: List[Dict], pattern: re.Pattern) -> str:
    """
    Search every report chunk for the claim regex. Return the first matching
    sentence (with a little surrounding context) or '' if none found.
    """
    for chunk in report_chunks:
        if chunk.get('source') not in ('report', 'figure'):
            continue
        text = chunk.get('text', '')
        match = pattern.search(text)
        if not match:
            continue

        # Pull the containing sentence for examiner context
        start = max(0, text.rfind('.', 0, match.start()) + 1)
        end = text.find('.', match.end())
        if end == -1:
            end = min(len(text), match.end() + 200)
        excerpt = text[start:end + 1].strip()
        return excerpt or text[max(0, match.start() - 50):match.end() + 100]

    return ''


def _claim_node_label(excerpt: str) -> str:
    """
    Turn a long claim excerpt into a short stable node label for the graph.
    Uses the first 6 words, lowercased.
    """
    words = re.sub(r'\s+', ' ', excerpt.strip()).split()
    label = ' '.join(words[:6]).lower()
    return label or 'unspecified_claim'
