# =============================================================================
# tools.py — All custom tools for the Resume Crew pipeline
# =============================================================================
# Organised by the agent that primarily uses each tool.
# All crewai-tools wrappers (FileReadTool, PDFSearchTool, SerperDevTool)
# are also instantiated here and imported by agents.py.
#
# Import pattern in agents.py:
#   from tools import (
#       file_read_tool, pdf_search_tool, section_extractor_tool,   # Parser
#       jd_scraper_tool, serper_tool, company_intel_tool,          # Analyst
#       ats_scorer, integrity_checker,                             # Tailor
#       cl_scorer, tone_analyzer,                                  # CL Writer
#       ats_final_scorer, consistency_checker,                     # QA
#       formatting_linter, gap_detector,                           # QA
#   )
# =============================================================================

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, ClassVar, Dict, List, Optional, Type
from urllib.parse import urlparse

import requests
from crewai.tools import BaseTool
from crewai_tools import FileReadTool, SerperDevTool
from pydantic import BaseModel, Field


# =============================================================================
# SECTION 1 — RESUME PARSER TOOLS
# =============================================================================

# ── Pydantic input schemas ────────────────────────────────────────────────────

class SectionExtractorInput(BaseModel):
    section_name: str = Field(
        description="Resume section to extract, e.g. 'Experience', 'Skills', 'Education'"
    )


# ── Tool definitions ──────────────────────────────────────────────────────────

class ResumeSectionExtractorTool(BaseTool):
    """
    Extracts a specific named section from the resume by header keyword.
    Fallback for unusual layouts that PDFSearchTool misses.
    """
    name: str = "Resume Section Extractor"
    description: str = (
        "Extracts a specific named section from the resume by searching for its header. "
        "Use when PDFSearchTool returns incomplete results for a section. "
        "Input: a section name like 'Work Experience', 'Technical Skills', or 'Education'."
    )
    args_schema: Type[BaseModel] = SectionExtractorInput
    resume_path: str = "resume.pdf"

    def _run(self, section_name: str) -> str:
        try:
            with open(self.resume_path, "rb") as f:
                try:
                    import fitz  # pymupdf
                    doc = fitz.open(self.resume_path)
                    full_text = "\n".join(page.get_text() for page in doc)
                except ImportError:
                    full_text = f.read().decode("utf-8", errors="ignore")

            lines = full_text.split("\n")
            section_lines = []
            in_section = False
            section_headers = [
                "experience", "education", "skills", "projects",
                "certifications", "awards", "publications", "summary",
                "objective", "profile", "work history", "employment",
                "achievements", "languages", "interests", "references",
            ]

            for line in lines:
                line_lower = line.strip().lower()
                if section_name.lower() in line_lower:
                    in_section = True
                    continue
                if in_section:
                    is_new_section = any(
                        h in line_lower
                        for h in section_headers
                        if h not in section_name.lower()
                    )
                    if is_new_section and len(line.strip()) < 40:
                        break
                    section_lines.append(line)

            if not section_lines:
                return f"Section '{section_name}' not found or is empty."
            return "\n".join(section_lines)

        except Exception as e:
            return f"Error reading resume file: {str(e)}"


# ── Instantiated tools (imported by agents.py) ────────────────────────────────

RESUME_PATH = os.getenv("RESUME_PATH", "resume.pdf")

file_read_tool     = FileReadTool(file_path=RESUME_PATH)  # locked to resume — for parser
generic_file_tool  = FileReadTool()                        # accepts any path — for JD analyst

section_extractor_tool = ResumeSectionExtractorTool(resume_path=RESUME_PATH)


# =============================================================================
# SECTION 2 — JOB ANALYST TOOLS
# =============================================================================

class JDScraperInput(BaseModel):
    url: str = Field(description="Full URL of the job posting to scrape")


class CompanyIntelInput(BaseModel):
    company_name: str = Field(description="Company name to research")
    role_title: str   = Field(description="Job title — used to find relevant tech stack info")


class JobDescriptionScraperTool(BaseTool):
    """
    Fetches and cleans job description text from a URL.
    Handles LinkedIn, Greenhouse, Lever, Workday, Indeed, and company career pages.
    """
    name: str = "Job Description Scraper"
    description: str = (
        "Fetches and extracts the clean text of a job description from a URL. "
        "Use when job input is a web link. Returns plain text of the full JD."
    )
    args_schema: Type[BaseModel] = JDScraperInput

    def _run(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html = response.text

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer",
                                  "header", "aside", "form", "iframe", "noscript"]):
                    tag.decompose()

                jd_selectors = [
                    "[class*='job-description']", "[class*='jobDescription']",
                    "[class*='description']", "[id*='job-description']",
                    "[id*='jobDescription']", "main", "article",
                ]
                text = ""
                for selector in jd_selectors:
                    container = soup.select_one(selector)
                    if container:
                        text = container.get_text(separator="\n", strip=True)
                        if len(text) > 200:
                            break
                if not text:
                    text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()

            lines = [l.strip() for l in text.split("\n") if l.strip()]
            clean_text = "\n".join(lines)

            if len(clean_text) < 100:
                return (
                    f"WARNING: Scraped content is very short ({len(clean_text)} chars). "
                    "Page may require JavaScript. Ask user to paste JD text directly.\n\n"
                    + clean_text
                )
            return f"=== JOB DESCRIPTION (scraped from {url}) ===\n\n{clean_text}"

        except requests.exceptions.RequestException as e:
            return (
                f"Failed to fetch URL: {str(e)}. "
                "Workday/SuccessFactors/Taleo postings require JavaScript rendering "
                "and cannot be scraped. Ask the user to paste the JD text."
            )


class CompanyIntelligenceTool(BaseTool):
    """
    Runs five targeted Serper searches for a company and returns a
    consolidated intel report: news, tech stack, culture, Glassdoor, funding.
    """
    name: str = "Company Intelligence Researcher"
    description: str = (
        "Runs targeted web searches to gather company intelligence: recent news, "
        "tech stack, culture signals, funding status, and Glassdoor themes. "
        "Use AFTER reading the JD. Pass the company name and role title."
    )
    args_schema: Type[BaseModel] = CompanyIntelInput

    def _run(self, company_name: str, role_title: str) -> str:
        serper = SerperDevTool()
        queries = {
            "recent_news": f"{company_name} news 2024 2025",
            "tech_stack":  f"{company_name} tech stack engineering blog {role_title}",
            "culture":     f"{company_name} company culture values mission",
            "glassdoor":   f"{company_name} Glassdoor review employee experience",
            "funding":     f"{company_name} funding valuation revenue growth",
        }
        results = {}
        for category, query in queries.items():
            try:
                results[category] = serper._run(query)
            except Exception as e:
                results[category] = f"Search failed: {str(e)}"

        lines = [f"=== COMPANY INTELLIGENCE: {company_name} ===\n"]
        for category, content in results.items():
            lines.append(f"--- {category.upper().replace('_', ' ')} ---")
            lines.append(str(content))
            lines.append("")
        return "\n".join(lines)


# ── Instantiated tools ────────────────────────────────────────────────────────

jd_scraper_tool = JobDescriptionScraperTool()

serper_tool = SerperDevTool(
    n_results=5,
    country="us",
    locale="en",
)

company_intel_tool = CompanyIntelligenceTool()


# =============================================================================
# SECTION 3 — RESUME TAILOR TOOLS
# =============================================================================

class ATSScorerInput(BaseModel):
    resume_text: str       = Field(description="Full text of the tailored resume draft")
    keywords:    List[str] = Field(description="ATS keywords from the job analysis")


class IntegrityCheckerInput(BaseModel):
    original_json_path:   str = Field(description="Path to the original parsed_resume.json")
    tailored_resume_text: str = Field(description="Full text of the tailored resume to validate")


class ATSKeywordScorerTool(BaseTool):
    """
    Scores a resume draft against a list of ATS keywords.
    Returns match percentage and lists present / missing / partial keywords.
    """
    name: str = "ATS Keyword Scorer"
    description: str = (
        "Scores a resume draft against a list of ATS keywords. "
        "Returns match percentage and lists which keywords are present, "
        "missing, or partially matched. Call after producing a draft."
    )
    args_schema: Type[BaseModel] = ATSScorerInput

    def _run(self, resume_text: str, keywords: List[str]) -> str:
        resume_lower = resume_text.lower()
        present, missing, partial = [], [], []

        for keyword in keywords:
            kw_lower = keyword.lower()
            if kw_lower in resume_lower:
                present.append(keyword)
                continue
            words = kw_lower.split()
            if len(words) > 1 and all(w in resume_lower for w in words):
                partial.append(keyword)
                continue
            found_fuzzy = False
            for word in words:
                for resume_word in resume_lower.split():
                    if SequenceMatcher(None, word, resume_word).ratio() > 0.88:
                        found_fuzzy = True
                        break
                if found_fuzzy:
                    break
            (partial if found_fuzzy else missing).append(keyword)

        total = len(keywords)
        score = ((len(present) + len(partial) * 0.5) / total * 100) if total > 0 else 0

        return json.dumps({
            "ats_score":      round(score, 1),
            "total_keywords": total,
            "exact_matches":  len(present),
            "partial_matches":len(partial),
            "missing_count":  len(missing),
            "present":  present,
            "partial":  partial,
            "missing":  missing,
            "recommendation": (
                "Good coverage — finalize the resume."
                if score >= 75
                else f"Coverage at {score:.0f}% — organically add missing keywords."
            ),
        }, indent=2)


class ResumeIntegrityCheckerTool(BaseTool):
    """
    Compares the tailored resume against the original parsed resume.
    Detects fabricated skills, inflated titles, invented metrics, or dropped experience.
    """
    name: str = "Resume Integrity Checker"
    description: str = (
        "Compares the tailored resume against the original parsed resume to detect "
        "fabricated skills, inflated titles, invented companies, or metrics that "
        "don't exist in the source. Run before finalizing. "
        "Empty issues list = resume is clean."
    )
    args_schema: Type[BaseModel] = IntegrityCheckerInput

    def _run(self, original_json_path: str, tailored_resume_text: str) -> str:
        try:
            with open(original_json_path) as f:
                original = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return f"Could not load original resume: {e}. Skipping integrity check."

        issues = []
        tailored_lower = tailored_resume_text.lower()

        # Check all original companies are still present
        for exp in original.get("experience", []):
            company = exp.get("company", "").lower()
            if company and company not in tailored_lower:
                issues.append(
                    f"Missing experience entry: '{exp.get('company')}' was removed."
                )

        # Check metrics in tailored resume exist in original
        original_text = " ".join(
            b for exp in original.get("experience", []) for b in exp.get("bullets", [])
        ) + " ".join(original.get("achievements", []))
        original_metrics = set(re.findall(r"\d+\.?\d*\s*%", original_text))
        tailored_metrics = re.findall(r"\d+\.?\d*\s*%", tailored_resume_text)
        for metric in tailored_metrics:
            normalised = re.sub(r"\s+", "", metric)
            if not any(re.sub(r"\s+", "", m) == normalised for m in original_metrics):
                issues.append(
                    f"Potentially fabricated metric: '{metric}' not in original resume."
                )

        # Check skills section for unknown skills
        original_skills = set()
        for skill_list in original.get("skills", {}).values():
            original_skills.update(s.lower() for s in skill_list)
        for exp in original.get("experience", []):
            for tech in exp.get("technologies", []):
                original_skills.add(tech.lower())

        skills_match = re.search(
            r"(?:skills|technical skills|core competencies)(.*?)(?:##|$)",
            tailored_resume_text, re.IGNORECASE | re.DOTALL
        )
        if skills_match:
            tokens = re.findall(r"\b([A-Z][a-zA-Z+#.]+)\b", skills_match.group(1))
            for token in tokens:
                if len(token) > 2 and token.lower() not in original_skills:
                    issues.append(
                        f"Possible fabricated skill: '{token}' not in original resume."
                    )

        if not issues:
            return json.dumps({
                "status": "PASS",
                "issues": [],
                "message": "No integrity issues detected.",
            }, indent=2)

        return json.dumps({
            "status": "REVIEW",
            "issues": issues,
            "message": f"{len(issues)} potential issue(s). Review each item.",
        }, indent=2)


# ── Instantiated tools ────────────────────────────────────────────────────────

ats_scorer        = ATSKeywordScorerTool()
integrity_checker = ResumeIntegrityCheckerTool()


# =============================================================================
# SECTION 4 — COVER LETTER WRITER TOOLS
# =============================================================================

class CLScorerInput(BaseModel):
    cover_letter_text: str       = Field(description="Full text of the cover letter draft")
    resume_text:       str       = Field(description="Full text of the tailored resume")
    ats_keywords:      List[str] = Field(description="ATS keyword list from job analysis")
    company_name:      str       = Field(description="Company name for specificity checks")
    culture_keywords:  List[str] = Field(description="Culture keywords from job analysis")


class ToneAnalyzerInput(BaseModel):
    cover_letter_text: str = Field(description="Full text of the cover letter")
    company_tone: str = Field(
        description="Company tone from job analysis: formal/casual/mission-driven/startup/corporate"
    )


class CoverLetterScorerTool(BaseTool):
    """
    Scores a cover letter draft across 5 dimensions: word count, ATS coverage,
    resume echo, company specificity, and tone alignment.
    """
    name: str = "Cover Letter Quality Scorer"
    description: str = (
        "Scores a cover letter draft across 5 dimensions: word count compliance, "
        "ATS keyword coverage, resume echo (repetition), company specificity, and "
        "tone alignment. Call after producing a draft to decide whether to iterate."
    )
    args_schema: Type[BaseModel] = CLScorerInput

    def _run(
        self,
        cover_letter_text: str,
        resume_text: str,
        ats_keywords: List[str],
        company_name: str,
        culture_keywords: List[str],
    ) -> str:
        flags    = []
        cl_lower = cover_letter_text.lower()

        word_count = len(cover_letter_text.split())
        if word_count < 200:
            flags.append(f"Too short: {word_count} words (minimum 200).")
        elif word_count > 400:
            flags.append(f"Too long: {word_count} words (maximum 400).")

        kw_hits = sum(1 for kw in ats_keywords if kw.lower() in cl_lower)
        if kw_hits < 4:
            flags.append(f"Low ATS coverage: {kw_hits} keywords. Aim for 4–8.")

        # Resume echo score
        resume_words = resume_text.lower().split()
        resume_phrases = {
            " ".join(resume_words[i:i+5])
            for i in range(len(resume_words) - 4)
            if not any(s in " ".join(resume_words[i:i+5])
                       for s in ["the", "and", "with", "for", "that"])
        }
        cl_words = cl_lower.split()
        echo_hits = sum(
            1 for i in range(len(cl_words) - 4)
            if " ".join(cl_words[i:i+5]) in resume_phrases
        )
        echo_score = min(1.0, echo_hits / max(1, len(cl_words) // 5))
        if echo_score > 0.25:
            flags.append("High resume repetition. Cover letter should interpret, not restate.")

        # Company specificity
        specificity_signals = [
            company_name.lower() in cl_lower,
            any(kw.lower() in cl_lower for kw in culture_keywords[:5]),
            bool(re.search(r"\b(product|platform|mission|team|launch|series)\b", cl_lower)),
            bool(re.search(r"\b(because|specifically|impressed by|drawn to)\b", cl_lower)),
        ]
        specificity_score = sum(specificity_signals) / len(specificity_signals)
        if specificity_score < 0.5:
            flags.append("Low company specificity. Add a reference to product, mission, or news.")

        # Tone alignment
        culture_hits = sum(1 for kw in culture_keywords if kw.lower() in cl_lower)
        tone_score   = min(1.0, culture_hits / max(1, min(5, len(culture_keywords))))
        if tone_score < 0.4:
            flags.append("Tone mismatch: few culture keywords present.")

        # Common anti-patterns
        opener = " ".join(cover_letter_text.split()[:30]).lower()
        if opener.startswith("i am writing"):
            flags.append("Weak opener: 'I am writing to apply...' Replace with a specific hook.")
        if opener.startswith("i am excited"):
            flags.append("Weak opener: 'I am excited to apply...' Lead with value instead.")

        if len(re.findall(r"\bI\b", cover_letter_text)) > 12:
            flags.append("Excessive first-person usage. Restructure some sentences.")

        cliches = [
            "passionate about", "team player", "hard worker", "think outside the box",
            "fast learner", "self-starter", "go-getter", "results-driven",
        ]
        found_cliches = [c for c in cliches if c in cl_lower]
        if found_cliches:
            flags.append(f"Clichés detected: {found_cliches}.")

        # Hook type detection
        hook_type = "direct_value_prop"
        if any(w in opener for w in ["raised", "launched", "announced", "series", "funding"]):
            hook_type = "company_news"
        elif any(w in opener for w in ["product", "platform", "built", "using"]):
            hook_type = "specific_product"
        elif any(w in opener for w in ["mission", "vision", "believe", "world"]):
            hook_type = "shared_mission"
        elif any(w in opener for w in ["referred", "met", "spoke", "connected"]):
            hook_type = "mutual_connection"

        return json.dumps({
            "word_count":                word_count,
            "paragraph_count":           cover_letter_text.count("\n\n") + 1,
            "hook_type":                 hook_type,
            "ats_keyword_count":         kw_hits,
            "tone_match_score":          round(tone_score, 2),
            "resume_echo_score":         round(echo_score, 2),
            "company_specificity_score": round(specificity_score, 2),
            "overall_quality":           "PASS" if not flags else "REVISE",
            "flags":                     flags,
        }, indent=2)


class ToneAnalyzerTool(BaseTool):
    """
    Checks whether the cover letter's vocabulary register matches
    the company's communication tone (formal/casual/startup/corporate/mission-driven).
    """
    name: str = "Cover Letter Tone Analyzer"
    description: str = (
        "Analyzes the linguistic register of the cover letter and checks whether "
        "it matches the company's expected tone. Returns tone signals and specific "
        "sentences to revise if there is a mismatch."
    )
    args_schema: Type[BaseModel] = ToneAnalyzerInput

    TONE_SIGNALS: Dict[str, Dict[str, List[str]]] = {
        "formal": {
            "positive": ["I would welcome", "I look forward to", "I am pleased",
                         "at your convenience", "I respectfully"],
            "negative": ["hey", "super", "awesome", "really excited", "cool", "pumped"],
        },
        "casual": {
            "positive": ["I'd love to", "excited to", "I'd bring", "ship", "learn"],
            "negative": ["I would be honoured", "herewith", "pursuant to",
                         "I respectfully submit"],
        },
        "mission-driven": {
            "positive": ["impact", "mission", "change", "believe", "world",
                         "community", "purpose", "values"],
            "negative": ["compensation first", "salary", "benefits above all"],
        },
        "startup": {
            "positive": ["ship", "build", "scale", "own", "move fast", "iterate",
                         "launch", "growth", "metrics"],
            "negative": ["I would be honoured", "distinguished organization",
                         "long-standing", "established institution"],
        },
        "corporate": {
            "positive": ["stakeholders", "strategic", "cross-functional", "alignment",
                         "deliverables", "enterprise", "leverage"],
            "negative": ["super pumped", "hack", "hustle", "grind", "crush it"],
        },
    }

    def _run(self, cover_letter_text: str, company_tone: str) -> str:
        tone_key = company_tone.lower().replace("-", "_").replace(" ", "_")
        signals  = self.TONE_SIGNALS.get(tone_key, self.TONE_SIGNALS["casual"])
        cl_lower = cover_letter_text.lower()

        positive_hits = [w for w in signals["positive"] if w in cl_lower]
        negative_hits = [w for w in signals["negative"] if w in cl_lower]
        tone_score    = (
            (len(positive_hits) / max(1, len(signals["positive"]))) * 0.7
            + (1 - len(negative_hits) / max(1, len(signals["negative"]))) * 0.3
        )

        suggestions = []
        if negative_hits:
            suggestions.append(
                f"Remove tone-mismatched phrases: {negative_hits}."
            )
        if tone_score < 0.4:
            suggestions.append(
                f"Add '{company_tone}' register signals: {signals['positive'][:4]}."
            )
        if not suggestions:
            suggestions.append(f"Tone well-aligned with '{company_tone}' register.")

        return json.dumps({
            "target_tone":             company_tone,
            "tone_alignment_score":    round(tone_score, 2),
            "positive_signals_found":  positive_hits,
            "tone_mismatches_found":   negative_hits,
            "verdict":   "ALIGNED" if tone_score >= 0.5 and not negative_hits else "MISMATCH",
            "suggestions": suggestions,
        }, indent=2)


# ── Instantiated tools ────────────────────────────────────────────────────────

cl_scorer     = CoverLetterScorerTool()
tone_analyzer = ToneAnalyzerTool()


# =============================================================================
# SECTION 5 — QA REVIEWER TOOLS
# =============================================================================

class ATSFinalScorerInput(BaseModel):
    resume_text:       str                 = Field(description="Full tailored resume text")
    cover_letter_text: str                 = Field(description="Full cover letter text")
    ats_keywords:      List[str]           = Field(description="ATS keywords from job analysis")
    required_skills:   List[Dict[str, Any]]= Field(description="Required skill objects from job analysis")


class ConsistencyInput(BaseModel):
    parsed_resume_json:   str = Field(description="Raw JSON string of parsed_resume.json")
    tailored_resume_text: str = Field(description="Full tailored resume text")
    cover_letter_text:    str = Field(description="Full cover letter text")


class FormattingLinterInput(BaseModel):
    resume_text:        str = Field(description="Full tailored resume text")
    cover_letter_text:  str = Field(description="Full cover letter text")
    candidate_name:     str = Field(description="Candidate's full name")


class GapDetectorInput(BaseModel):
    resume_text:               str                = Field(description="Full tailored resume text")
    cover_letter_text:         str                = Field(description="Full cover letter text")
    required_skills:           List[Dict[str, Any]]= Field(description="Required skills from job analysis")
    primary_responsibilities:  List[str]          = Field(description="Primary responsibilities from job analysis")


class ATSFinalScorerTool(BaseTool):
    """
    Comprehensive ATS audit across both resume and cover letter.
    Checks coverage, placement quality, density (stuffing detection),
    and section-level distribution.
    """
    name: str = "ATS Final Scorer"
    description: str = (
        "Comprehensive ATS audit across both the resume and cover letter. "
        "Checks keyword coverage, placement quality, stuffing detection, "
        "and section-level distribution. Returns a detailed score breakdown."
    )
    args_schema: Type[BaseModel] = ATSFinalScorerInput

    def _run(
        self,
        resume_text: str,
        cover_letter_text: str,
        ats_keywords: List[str],
        required_skills: List[Dict[str, Any]],
    ) -> str:
        resume_lower   = resume_text.lower()
        cl_lower       = cover_letter_text.lower()
        combined_lower = resume_lower + " " + cl_lower
        present, missing, partial = [], [], []

        for kw in ats_keywords:
            kw_lower = kw.lower()
            if kw_lower in resume_lower:
                present.append(kw)
            elif kw_lower in combined_lower:
                partial.append(f"{kw} (cover letter only — add to resume)")
            else:
                words = kw_lower.split()
                if len(words) > 1 and all(w in resume_lower for w in words):
                    partial.append(f"{kw} (fragmented — consolidate)")
                else:
                    missing.append(kw)

        total = len(ats_keywords) if ats_keywords else 1
        score = ((len(present) + len(partial) * 0.4) / total) * 100
        stuffing = any(resume_lower.count(kw.lower()) > 4 for kw in ats_keywords)

        required_names = {
            s["skill"].lower()
            for s in required_skills
            if s.get("is_required", False)
        }
        high_priority_missing = [kw for kw in missing if kw.lower() in required_names]

        return json.dumps({
            "ats_score":               round(score, 1),
            "total_keywords":          total,
            "exact_in_resume":         len(present),
            "partial_or_cl_only":      len(partial),
            "missing":                 len(missing),
            "keywords_present":        present,
            "keywords_partial":        partial,
            "keywords_missing":        missing,
            "high_priority_missing":   high_priority_missing,
            "keyword_stuffing_detected": stuffing,
            "recommended_additions":   high_priority_missing[:5],
            "verdict": "STRONG" if score >= 80 else "GOOD" if score >= 65 else "WEAK",
        }, indent=2)


class ConsistencyCheckerTool(BaseTool):
    """
    Verifies facts in the cover letter are consistent with the resume:
    company names, job titles, dates, and metrics.
    """
    name: str = "Cross-Document Consistency Checker"
    description: str = (
        "Verifies that facts stated in the cover letter are consistent with "
        "the resume: company names, job titles, dates, metrics. "
        "Also checks that the tailored resume hasn't dropped any original entries. "
        "Returns all mismatches found."
    )
    args_schema: Type[BaseModel] = ConsistencyInput

    def _run(
        self,
        parsed_resume_json: str,
        tailored_resume_text: str,
        cover_letter_text: str,
    ) -> str:
        issues: Dict[str, List[str]] = {
            "date_mismatches":    [],
            "title_mismatches":   [],
            "metric_mismatches":  [],
            "other":              [],
        }
        try:
            original = json.loads(parsed_resume_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Could not parse resume JSON.", "issues": issues})

        cl_lower     = cover_letter_text.lower()
        resume_lower = tailored_resume_text.lower()

        # Metrics in cover letter must exist in original
        cl_metrics    = re.findall(r"\d+\.?\d*\s*%", cover_letter_text)
        original_text = " ".join(
            b for exp in original.get("experience", []) for b in exp.get("bullets", [])
        )
        original_metrics = set(re.findall(r"\d+\.?\d*\s*%", original_text))
        for metric in cl_metrics:
            normalised = re.sub(r"\s+", "", metric)
            if not any(re.sub(r"\s+", "", m) == normalised for m in original_metrics):
                issues["metric_mismatches"].append(
                    f"Metric '{metric}' in cover letter not found in original resume."
                )

        # All original experience entries must still be in tailored resume
        for exp in original.get("experience", []):
            company = exp.get("company", "").lower()
            if company and company not in resume_lower:
                issues["other"].append(
                    f"Experience entry '{exp.get('company')}' was dropped from tailored resume."
                )

        # Years in cover letter should exist in the original
        cl_years       = re.findall(r"\b(19|20)\d{2}\b", cover_letter_text)
        original_years = set(re.findall(r"\b(19|20)\d{2}\b", original_text))
        for year in cl_years:
            if year not in original_years and int(year) < datetime.now().year:
                issues["date_mismatches"].append(
                    f"Year '{year}' in cover letter not found in original resume dates."
                )

        all_issues = (
            issues["date_mismatches"] + issues["title_mismatches"]
            + issues["metric_mismatches"] + issues["other"]
        )
        return json.dumps({
            "passed":            len(all_issues) == 0,
            "total_issues":      len(all_issues),
            "date_mismatches":   issues["date_mismatches"],
            "title_mismatches":  issues["title_mismatches"],
            "metric_mismatches": issues["metric_mismatches"],
            "other_issues":      issues["other"],
        }, indent=2)


class FormattingLinterTool(BaseTool):
    """
    Checks both documents for formatting issues: broken markdown,
    missing sections, weak verbs, orphaned bullets, clichés, and more.
    """
    name: str = "Document Formatting Linter"
    description: str = (
        "Checks both documents for formatting issues: broken markdown, "
        "missing required sections, weak verbs, orphaned bullets, "
        "excessive whitespace, and contact info completeness."
    )
    args_schema: Type[BaseModel] = FormattingLinterInput

    REQUIRED_RESUME_SECTIONS: ClassVar[List[str]] = ["## summary", "## experience", "## skills", "## education"]
    WEAK_VERBS: ClassVar[List[str]] = [
        "worked on", "helped with", "was responsible for", "assisted with",
        "participated in", "involved in", "contributed to", "supported the",
        "was part of",
    ]
    CLICHES: ClassVar[List[str]] = [
        "passionate about", "team player", "hard worker", "think outside the box",
        "fast learner", "self-starter", "go-getter", "results-driven",
        "proven track record", "detail-oriented", "synergy",
    ]
    GENERIC_OPENERS: ClassVar[List[str]] = [
        "i am writing to apply", "i am writing to express", "i am excited to apply",
        "i am passionate about", "please accept my application", "i would like to apply",
    ]

    def _run(
        self,
        resume_text: str,
        cover_letter_text: str,
        candidate_name: str,
    ) -> str:
        resume_issues, cl_issues = [], []
        resume_lower = resume_text.lower()
        cl_lower     = cover_letter_text.lower()

        # Resume checks
        for section in self.REQUIRED_RESUME_SECTIONS:
            if section not in resume_lower:
                resume_issues.append(f"Missing section: '{section.replace('## ', '').title()}'")

        if candidate_name.lower() not in resume_lower[:200]:
            resume_issues.append("Candidate name not found in resume header.")

        found_weak = [v for v in self.WEAK_VERBS if v in resume_lower]
        if found_weak:
            resume_issues.append(f"Weak verbs still present: {found_weak}")

        if re.findall(r"^-\s*$", resume_text, re.MULTILINE):
            resume_issues.append("Empty bullet lines detected.")

        if re.findall(r"\b(I|me|my|myself)\b", resume_text):
            resume_issues.append("First-person pronouns in resume — remove all.")

        if "\n\n\n" in resume_text:
            resume_issues.append("Triple blank lines detected — reduce to single.")

        role_headers = re.findall(r"^### .+", resume_text, re.MULTILINE)
        for i, header in enumerate(role_headers):
            start = resume_text.find(header)
            end   = resume_text.find(role_headers[i + 1]) if i + 1 < len(role_headers) else len(resume_text)
            bullets = len(re.findall(r"^- ", resume_text[start:end], re.MULTILINE))
            if bullets > 8:
                resume_issues.append(f"'{header.strip()}' has {bullets} bullets — max 6.")

        # Cover letter checks
        opener_150 = cover_letter_text.strip()[:150].lower()
        for opener in self.GENERIC_OPENERS:
            if opener in opener_150:
                cl_issues.append(f"Generic opener: '{opener}' — replace with a specific hook.")
                break

        cl_words = len(cover_letter_text.split())
        if cl_words < 200:
            cl_issues.append(f"Cover letter too short: {cl_words} words.")
        elif cl_words > 400:
            cl_issues.append(f"Cover letter too long: {cl_words} words.")

        if len(re.findall(r"\bI\b", cover_letter_text)) > 14:
            cl_issues.append("'I' overused in cover letter — restructure 2-3 sentences.")

        found_cliches = [c for c in self.CLICHES if c in cl_lower]
        if found_cliches:
            cl_issues.append(f"Clichés to remove: {found_cliches}")

        body_start = cover_letter_text.find("---")
        cl_body    = cover_letter_text[body_start:] if body_start != -1 else cover_letter_text
        if re.search(r"^- ", cl_body, re.MULTILINE):
            cl_issues.append("Bullet points in cover letter body — convert to prose.")

        if candidate_name.lower() not in cover_letter_text.lower()[-200:]:
            cl_issues.append("Candidate name not found in cover letter sign-off.")

        return json.dumps({
            "resume_issues":        resume_issues,
            "cover_letter_issues":  cl_issues,
            "resume_passed":        len(resume_issues) == 0,
            "cover_letter_passed":  len(cl_issues) == 0,
            "total_issues":         len(resume_issues) + len(cl_issues),
        }, indent=2)


class GapDetectorTool(BaseTool):
    """
    Scans both documents against required skills and responsibilities.
    Identifies unaddressed requirements and weak evidence areas.
    """
    name: str = "Requirement Gap Detector"
    description: str = (
        "Scans both documents against JD required skills and primary responsibilities. "
        "Identifies gaps and weak coverage areas. "
        "Returns mitigation strategies that don't require fabricating experience."
    )
    args_schema: Type[BaseModel] = GapDetectorInput

    SYNONYMS: ClassVar[Dict[str, List[str]]] = {
        "kubernetes":       ["k8s", "container orchestration"],
        "postgresql":       ["postgres", "relational database"],
        "javascript":       ["js", "node.js", "typescript"],
        "machine learning": ["ml", "ai", "predictive model"],
        "ci/cd":            ["continuous integration", "github actions", "jenkins"],
        "rest api":         ["restful", "api design", "http api"],
        "agile":            ["scrum", "sprint", "kanban"],
    }
    STOP_WORDS: ClassVar[set] = {
        "their", "these", "those", "which", "where", "while",
        "other", "about", "across", "within",
    }

    def _run(
        self,
        resume_text: str,
        cover_letter_text: str,
        required_skills: List[Dict[str, Any]],
        primary_responsibilities: List[str],
    ) -> str:
        combined     = (resume_text + " " + cover_letter_text).lower()
        gaps, weak, mitigations = [], [], []

        for skill_obj in required_skills:
            skill = skill_obj.get("skill", "")
            if not skill:
                continue
            count = combined.count(skill.lower())
            if count == 0:
                found_synonym = False
                for canonical, syns in self.SYNONYMS.items():
                    if skill.lower() in canonical or canonical in skill.lower():
                        if any(s in combined for s in syns):
                            weak.append(
                                f"'{skill}' not by name but synonym present — "
                                f"add exact term to Skills section."
                            )
                            found_synonym = True
                            break
                if not found_synonym:
                    gaps.append(skill)
                    mitigations.append(
                        f"'{skill}' required but absent. If candidate has exposure, "
                        f"add to Skills. If not, prepare adjacent-skill answer for interview."
                    )
            elif count == 1:
                weak.append(f"'{skill}' appears once — strengthen coverage.")

        for responsibility in primary_responsibilities:
            resp_words = [
                w for w in responsibility.lower().split()
                if len(w) > 4 and w not in self.STOP_WORDS
            ]
            coverage = sum(1 for w in resp_words if w in combined)
            ratio    = coverage / len(resp_words) if resp_words else 1.0
            if ratio < 0.4:
                gaps.append(f"Responsibility: {responsibility[:80]}...")
                mitigations.append(
                    f"Core responsibility '{responsibility[:60]}...' has minimal coverage. "
                    f"Add a direct bullet to the most relevant experience entry."
                )
            elif ratio < 0.65:
                weak.append(
                    f"Responsibility '{responsibility[:60]}...' weakly covered — "
                    f"add a more direct bullet."
                )

        return json.dumps({
            "unaddressed_requirements": gaps,
            "weak_evidence_areas":      weak,
            "mitigation_suggestions":   mitigations,
            "gap_count":                len(gaps),
            "weak_count":               len(weak),
            "verdict": "CLEAN" if not gaps else "GAPS" if len(gaps) <= 3 else "CRITICAL",
        }, indent=2)


# ── Instantiated tools ────────────────────────────────────────────────────────

ats_final_scorer    = ATSFinalScorerTool()
consistency_checker = ConsistencyCheckerTool()
formatting_linter   = FormattingLinterTool()
gap_detector        = GapDetectorTool()

# FileReadTool instances for QA (no fixed path — agent reads multiple files)
qa_parsed_resume_reader    = FileReadTool(file_path="outputs/parsed_resume.json")
qa_job_analysis_reader     = FileReadTool(file_path="outputs/job_analysis.json")
qa_tailored_resume_reader  = FileReadTool(file_path="outputs/tailored_resume.md")
qa_cover_letter_reader     = FileReadTool(file_path="outputs/cover_letter.md")