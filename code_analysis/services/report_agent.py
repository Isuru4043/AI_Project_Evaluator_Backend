import json
from django.conf import settings

from AI_Evaluator_Backend.llm import get_llm


class CodeAnalysisReportAgent:
    """
    Synthesizes SonarQube static analysis metrics + Gemini's code summary
    into a single finalized report for examiners to review.
    """

    SYSTEM_PROMPT = """
    You are a senior software engineering examiner producing a final code 
    quality report for a university final-year project. You have:
    1. Static analysis metrics from SonarQube (bugs, vulnerabilities, 
       code smells, ratings, coverage, duplication)
    2. A list of specific issues found (with severity and location)
    3. A prior high-level summary of the codebase's architecture

    Produce a structured, honest, examiner-facing report. Be specific 
    about real issues found — do not soften or hide genuine problems, 
    but also acknowledge what was done well. This report will inform 
    grading decisions.

    Return ONLY valid JSON with this exact structure:
    {
      "executive_summary": "2-3 sentence overview of overall code quality",
      "strengths": ["specific strength 1", "specific strength 2"],
      "concerns": [
        {"severity": "high|medium|low", "issue": "description", "recommendation": "what to fix"}
      ],
      "metrics_interpretation": "plain-English explanation of what the SonarQube ratings mean for this project",
      "security_assessment": "summary of any vulnerabilities/security hotspots found, or 'No significant security concerns identified.'",
      "maintainability_verdict": "brief verdict on long-term maintainability",
      "overall_recommendation": "pass|pass_with_concerns|needs_improvement",
      "recommendation_reason": "1-2 sentence justification for the verdict"
    }
    """

    def __init__(self, model=None):
        self.model = model or settings.GEMINI_MODEL
        self.client = get_llm()

    def generate_report(self, sonar_summary: dict, code_summary: str,
                         quality_status: str, quality_reason: str) -> dict:
        measures = {m['metric']: m.get('value') for m in sonar_summary.get('measures', [])}
        issues = sonar_summary.get('issues', [])[:15]  # cap to keep prompt size sane

        issues_text = "\n".join(
            f"- [{i.get('severity')}] {i.get('type')}: {i.get('message')} "
            f"(line {i.get('textRange', {}).get('startLine', '?')})"
            for i in issues
        ) or "No specific issues returned."

        context = f"""
        SONAR METRICS:
        {json.dumps(measures, indent=2)}

        TOTAL ISSUES FOUND: {sonar_summary.get('total_issues', 0)}

        TOP ISSUES:
        {issues_text}

        AUTOMATED QUALITY GATE RESULT: {quality_status}
        REASON: {quality_reason or 'N/A'}

        PRIOR CODE SUMMARY (architecture/approach):
        {code_summary or 'Not available.'}
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.SYSTEM_PROMPT, context],
        )

        raw_text = response.text.strip()
        if raw_text.startswith('```'):
            raw_text = raw_text.strip('`').replace('json\n', '', 1)

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "executive_summary": "Report generation failed to parse correctly.",
                "strengths": [],
                "concerns": [],
                "metrics_interpretation": raw_text[:1000],
                "security_assessment": "N/A",
                "maintainability_verdict": "N/A",
                "overall_recommendation": "unknown",
                "recommendation_reason": "AI response could not be parsed as JSON.",
            }
