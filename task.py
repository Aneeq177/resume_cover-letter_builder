# =============================================================================
# tasks.py — All 5 task definitions for the Resume Crew pipeline
# =============================================================================
# Tasks are defined as functions that return Task objects so the context
# chain can be wired lazily. main.py calls build_tasks() to get all five.
#
# Import pattern in main.py:
#   from tasks import build_tasks
#   tasks = build_tasks()
# =============================================================================

import os
from crewai import Task

from agents import (
    resume_parser_agent,
    job_description_parser_agent,
    resume_tailor_agent,
    cover_letter_writer_agent,
    qa_reviewer_agent,
)


def build_tasks() -> dict:
    """
    Builds and returns all five Task objects with correct context chaining.
    Returns a dict keyed by task name for easy access in main.py.

    Usage:
        tasks = build_tasks()
        crew = Crew(tasks=list(tasks.values()), ...)
    """

    # =========================================================================
    # TASK 1 — Resume Parse
    # =========================================================================

    resume_parse_task = Task(
        description="""
        Parse the resume located at: {resume_path}

        STEP 1 — Full document read
        Use FileReadTool to read the entire resume for a complete first pass.

        STEP 2 — Targeted section retrieval
        Use ResumeSectionExtractorTool to pull each of these sections individually:
        - "Experience" (or "Work History" or "Employment")
        - "Skills" (or "Technical Skills" or "Core Competencies")
        - "Education"
        - "Projects" (if present)
        - "Certifications" (if present)
        - "Achievements" or "Awards" (if present)

        STEP 3 — Deep search for quantified metrics
        Use FileReadTool to re-read the resume focusing on quantified achievements
        (numbers, percentages, revenue figures, growth metrics, user counts).

        STEP 4 — Deep search for technologies
        Use ResumeSectionExtractorTool with section_name="Skills" to extract all
        technologies, frameworks, tools, platforms, and languages.

        STEP 5 — Assemble the final JSON
        Return a single JSON object with this exact structure:

        {{
          "contact": {{
            "name": "", "email": "", "phone": "",
            "linkedin": "", "github": "", "portfolio": "", "location": ""
          }},
          "summary": "Verbatim professional summary or null",
          "experience": [
            {{
              "company": "Employer name",
              "title": "Job title exactly as written",
              "dates": "Jan 2022 – Present",
              "location": "City, State",
              "bullets": ["Verbatim bullet — never paraphrase, preserve all numbers"],
              "technologies": ["React", "PostgreSQL", "AWS"]
            }}
          ],
          "skills": {{
            "technical": [],
            "languages": [],
            "tools": [],
            "soft": []
          }},
          "education": [
            {{
              "institution": "", "degree": "", "field": "",
              "dates": "", "gpa": null, "honors": null,
              "relevant_coursework": []
            }}
          ],
          "certifications": [],
          "projects": [
            {{
              "name": "", "description": "", "technologies": [],
              "link": null, "dates": null
            }}
          ],
          "achievements": [],
          "languages": []
        }}

        CRITICAL RULES:
        - NEVER paraphrase bullet points. Copy verbatim, numbers and all.
        - NEVER invent information. Missing fields → null or [].
        - NEVER merge two jobs into one entry.
        - ALWAYS include ALL experience entries, even 1-month internships.
        - Output ONLY the raw JSON. No markdown fences. No explanation.
        """,

        expected_output=(
            "A single valid JSON object containing the complete structured resume data. "
            "Every field from the schema must be present (null or [] if empty). "
            "Bullet points are verbatim. All quantified metrics are preserved. "
            "Parseable with json.loads() — no trailing commas, no markdown."
        ),

        agent=resume_parser_agent,
        output_file="outputs/parsed_resume.json",
    )

    # =========================================================================
    # TASK 2 — Job Analysis
    # =========================================================================

    jd_analysis_task = Task(
        description="""
        Analyze the job posting provided at: {job_input}

        {job_input} is one of:
        (a) A URL     → use JobDescriptionScraperTool
        (b) A file path → use FileReadTool
        (c) Raw text  → use directly

        STEP 1 — Acquire the job description
        Detect the input type and load the JD accordingly.

        STEP 2 — Extract ATS keywords
        Identify the top 20 keywords the ATS will scan for:
        - Technical skills named more than once
        - Exact tool/framework names (spelling matters)
        - Soft skills in the requirements section
        - Buzzwords from the responsibilities section
        Count frequency — it signals priority.

        STEP 3 — Research the company
        Extract the company name, then call CompanyIntelligenceTool with the
        company name and role title. If results are thin, use SerperDevTool
        directly: "[company] careers engineering blog" and "[company] crunchbase".

        STEP 4 — Identify red flags
        Flag: "rockstar/ninja", 10+ years for mid-level, 3+ unrelated domains,
        no team structure mentioned, missing salary for senior roles.

        STEP 5 — Synthesize the application angle
        Write a 2-3 sentence strategic note: what problem is the company solving
        by hiring for this role, and how should the candidate frame themselves
        as the solution?

        STEP 6 — Assemble the final JSON:
        {{
          "job_title": "",
          "company_name": "",
          "location": "",
          "employment_type": "",
          "seniority": "",
          "salary_range": null,
          "required_skills": [
            {{"skill": "", "level": "", "is_required": true, "mentioned_times": 1}}
          ],
          "preferred_skills": [
            {{"skill": "", "level": "", "is_required": false, "mentioned_times": 1}}
          ],
          "ats_keywords": [],
          "role_expectations": {{
            "primary_responsibilities": [],
            "success_metrics": [],
            "team_context": "",
            "growth_signals": [],
            "red_flags": []
          }},
          "company_intel": {{
            "name": "", "industry": "", "size": "", "founded": null,
            "headquarters": null, "recent_news": [], "tech_stack": [],
            "notable_facts": [], "glassdoor_signals": []
          }},
          "culture_signals": {{
            "values": [], "tone": "",
            "work_style": "", "keywords_to_mirror": []
          }},
          "application_angle": ""
        }}

        CRITICAL RULES:
        - ATS keywords must be EXACT strings from the JD — no synonyms.
        - Required vs preferred must reflect the JD's own language.
        - Output ONLY the raw JSON. No markdown fences. No preamble.
        """,

        expected_output=(
            "A single valid JSON object matching the JobAnalysis schema. "
            "ATS keywords are exact strings from the JD. "
            "Company intel is sourced from web research, not hallucinated. "
            "application_angle is strategic and specific. "
            "Parseable with json.loads()."
        ),

        agent=job_description_parser_agent,
        output_file="outputs/job_analysis.json",
    )

    # =========================================================================
    # TASK 3 — Resume Tailoring
    # =========================================================================

    resume_tailor_task = Task(
        description="""
        Tailor the resume using outputs from the two upstream agents.

        STEP 1 — Load context
        1a. FileReadTool → outputs/parsed_resume.json
        1b. FileReadTool → outputs/job_analysis.json
        Study the application_angle carefully — it guides every rewriting decision.

        STEP 2 — Detect candidate seniority
        Examine parsed_resume.json: count total years of work experience and check
        whether the candidate is currently enrolled in education or graduated within
        the last 2 years.

        STUDENT / ENTRY-LEVEL (< 3 years total experience OR currently enrolled):
        - Target: 1 page maximum. This is a hard constraint.
        - Keep only the 4-5 most relevant experience entries.
        - Entries with a single bullet and no quantified metric → consolidate into
          one "Campus Involvement" line at the bottom (format: "Role, Org (Year)").
          Do NOT give them their own section headers.
        - Skills section: list only verified hard skills (languages, tools, software).
          Keep soft skills to at most 3 that are genuinely evidenced in the bullets.
          No comma-separated walls of 15+ soft skills.
        - Flag any experience date that shows a future year (e.g., 2026-Present when
          today is {today_date}) as a likely data-entry error and correct to the
          nearest plausible academic term start (e.g., Sep 2025 or Jan 2026).

        EXPERIENCED (≥ 3 years, not currently enrolled):
        - All experience entries must remain. No entries may be dropped.
        - 2-page maximum.

        STEP 3 — Rewrite the professional summary
        Write a 3-sentence summary:
        - Sentence 1: Years of experience + domain + top 2 relevant JD technologies.
        - Sentence 2: Most relevant achievement, reframed in JD terminology.
        - Sentence 3: Value prop for this specific company, using 1-2 culture keywords.
        Under 60 words. No first-person pronouns.

        STEP 4 — Reorder experience
        Rank by relevance to the role. Most relevant first.
        Apply the entry-count rules from STEP 2 (student vs experienced).

        STEP 5 — Rewrite bullet points
        RULE A — Action verb + impact: [Verb] + [what] + [outcome].
        Replace: "worked on", "helped", "assisted", "was responsible for".
        Use: Built, Designed, Led, Reduced, Increased, Shipped, Migrated,
        Automated, Optimized, Launched, Scaled, Deployed, Architected.

        RULE B — ATS injection: substitute exact JD terms for vaguer ones.
          ✓ "managed the database" → "managed PostgreSQL schemas and query optimization"
          ✗ Never list keywords inside a bullet.

        RULE C — Preserve all metrics exactly as written.

        RULE D — Never fabricate. Reframe only from original content.

        RULE E — 4-6 bullets for top 2 roles, 2-3 for older/less relevant.
          For student candidates: consolidate thin entries per STEP 2 rules.

        STEP 6 — Rewrite skills section
        Order: Languages → Frameworks & Libraries → Tools & Platforms → Cloud & DevOps.
        Use exact ATS keyword names where the candidate genuinely has those skills.
        For student candidates: omit the soft-skills line entirely if it would exceed
        3 genuinely evidenced skills — no keyword dumping.

        STEP 7 — Education and projects
        Education: keep verbatim.
        Projects: rewrite descriptions to emphasise JD technologies. Most relevant first.

        STEP 8 — Score the draft
        Call ATSKeywordScorerTool with the full resume text and ats_keywords.
        If score < 75: add missing keywords organically to existing bullets.

        STEP 9 — Integrity check
        Call ResumeIntegrityCheckerTool with:
        - original_json_path: "outputs/parsed_resume.json"
        - tailored_resume_text: your full draft
        If REVIEW: correct or remove any flagged fabrications.

        STEP 10 — Output the final resume in clean markdown:

        # [Full Name]
        [email] · [phone] · [location] · [linkedin] · [github]

        ## Summary
        [3-sentence tailored summary]

        ## Experience

        ### [Job Title] | [Company] | [Dates]
        [Location]
        - [Rewritten bullet]

        ## Skills
        **Languages:** ...
        **Frameworks:** ...
        **Tools & Platforms:** ...
        **Cloud:** ...

        ## Education
        ### [Degree] in [Field] | [Institution] | [Dates]

        ## Projects (if present)
        ### [Name] | [Technologies]
        - [Description]

        ## Certifications (if present)
        - [Cert] — [Issuer], [Year]

        Output ONLY the markdown. No preamble, no meta-commentary.
        """,

        expected_output=(
            "A complete tailored resume in clean markdown. "
            "Student/entry-level candidates: 1 page maximum, max 5 experience entries, "
            "thin single-bullet entries consolidated into a Campus Involvement line, "
            "soft skills trimmed to ≤3 evidenced items, implausible future dates corrected. "
            "Experienced candidates: all entries retained, ≤2 pages. "
            "Passes ATS scoring at 75%+ keyword coverage. "
            "Every bullet begins with a strong action verb. "
            "All metrics from the original are preserved exactly. "
            "No fabricated information. No first-person pronouns. "
            "Skills section leads with most JD-relevant categories."
        ),

        context=[resume_parse_task, jd_analysis_task],
        agent=resume_tailor_agent,
        output_file="outputs/tailored_resume.md",
    )

    # =========================================================================
    # TASK 4 — Cover Letter Writing
    # =========================================================================

    cover_letter_task = Task(
        description="""
        Write a targeted cover letter using the three upstream outputs.

        STEP 1 — Load all context
        1a. FileReadTool → outputs/parsed_resume.json
            Focus on: 2-3 strongest achievements, career arc, quantified metrics.
        1b. FileReadTool → outputs/job_analysis.json
            Focus on: application_angle, company_intel (recent_news, notable_facts),
            culture_signals (tone, keywords_to_mirror), role_expectations.
        1c. FileReadTool → outputs/tailored_resume.md
            Read carefully — the cover letter must NOT repeat it verbatim.

        STEP 2 — Choose an opening hook (select ONE):

        HOOK A — Company News (strongest when available)
        Reference a specific recent development: funding, launch, acquisition.
        "When [Company] raised its Series B to scale its payments infrastructure
        last March, it confirmed what I'd already seen building high-throughput
        systems at [Candidate Company]..."

        HOOK B — Specific Product/Problem
        Reference a real product or customer problem the company solves.

        HOOK C — Shared Mission
        Only use for companies with a clear, distinctive mission.

        HOOK D — Direct Value Proposition
        Lead with the specific value brought to the role's core challenge.

        NEVER open with:
        "I am writing to apply", "I am excited to apply",
        "I have always been passionate about", "My name is [Name] and I..."

        STEP 3 — Write three paragraphs

        PARAGRAPH 1 — Hook + Bridge (3-5 sentences)
        Hook → candidate's current role (1 sentence) →
        most relevant achievement with a number (1 sentence) →
        connection to this specific role (1 sentence).
        Do NOT list job titles. Do NOT summarise the career history.

        PARAGRAPH 2 — The Specific Case (split into TWO short paragraphs)
        Paragraph 2a (2-3 sentences): Name the role's core challenge
        (from primary_responsibilities) and connect to one specific accomplishment
        NOT already leading the resume. Show the mechanism: what was done, why it
        transfers.
        Paragraph 2b (2-3 sentences): Pivot to a second, distinct skill area or
        background that rounds out the case. End with the outcome in the company's
        language (mirror keywords_to_mirror).
        NEVER merge both themes into a single paragraph — a wall of text loses the
        reader at sentence 4.

        PARAGRAPH 3 — The Close (2-3 sentences)
        NOT "I look forward to hearing from you."
        Choose ONE pattern:
        - First 90 days: what you'd focus on, tied to a JD responsibility
        - Conversation opener: specific company direction + your relevance
        - Confident restatement + direct call to action

        STEP 4 — Tone alignment
        Call ToneAnalyzerTool with the draft and culture_signals.tone.
        If MISMATCH: apply suggestions and revise flagged sentences.
        Mirror 2-3 exact phrases from culture_signals.keywords_to_mirror.

        STEP 5 — Quality score
        Call CoverLetterScorerTool with:
        - cover_letter_text, resume_text, ats_keywords, company_name,
          culture_keywords (keywords_to_mirror)

        Address any flags:
        - word_count outside 200-400 → trim or expand
        - resume_echo_score > 0.25 → rewrite to interpret, not restate
        - company_specificity_score < 0.5 → add a specific company reference
        - clichés flagged → remove immediately
        - ATS keyword count < 4 → add 2-3 keywords organically to paragraph 2
        One revision pass only.

        STEP 6 — Output in clean markdown:

        [City, State] · [Email] · [Phone]
        {today_date}

        Hiring Team
        [Company Name]

        ---

        [Paragraph 1]

        [Paragraph 2a]

        [Paragraph 2b]

        [Paragraph 3]

        [Full Name]

        CRITICAL RULES:
        - 250-400 words total.
        - No bullet points in the body.
        - No first-person opener.
        - No resume verbatim repetition.
        - Company name must appear at least twice.
        - NEVER invent statistics, enrollment figures, or institutional facts not
          present verbatim in outputs/parsed_resume.json. If a number is not in
          the parsed resume, do not use it — omit or rephrase without the figure.
        - NEVER use assertive language that tells the employer what standards they
          must hold (e.g. "non-negotiable", "must-have", "you need"). Demonstrate
          that the candidate meets the standards; do not assert the standards.
        - The close must name something specific about THIS role or THIS company —
          not generic adjectives ("helpful, organized, invested") that could
          describe anyone applying anywhere.
        - Output ONLY the markdown.
        """,

        expected_output=(
            "A complete cover letter in clean markdown, 250-400 words. "
            "Date line is exactly {today_date}. "
            "Opens with a specific, non-generic hook. "
            "Four paragraphs: hook + bridge, specific case (2a), supporting case (2b), confident close. "
            "Close references a specific detail about this role or company — no generic adjectives only. "
            "Tone matches company register. Mirrors 2-3 culture keywords. "
            "Contains 4-8 ATS keywords woven naturally into prose. "
            "Zero invented statistics — every number traceable to parsed_resume.json. "
            "No assertive employer-directed language (non-negotiable, must-have). "
            "Does not repeat the resume verbatim. No clichés. No bullet points."
        ),

        context=[resume_parse_task, jd_analysis_task, resume_tailor_task],
        agent=cover_letter_writer_agent,
        output_file="outputs/cover_letter.md",
    )

    # =========================================================================
    # TASK 5 — QA Review
    # =========================================================================

    qa_review_task = Task(
        description="""
        Run a full audit on all four upstream outputs. Apply targeted fixes.
        Produce two approved final documents and a QA report.

        STEP 1 — Load all context
        1a. FileReadTool → outputs/parsed_resume.json
        1b. FileReadTool → outputs/job_analysis.json
        1c. FileReadTool → outputs/tailored_resume.md
        1d. FileReadTool → outputs/cover_letter.md

        STEP 2 — ATS final audit
        Call ATSFinalScorerTool with resume, cover letter, ats_keywords, required_skills.
        If high_priority_missing keywords exist AND candidate has those skills in the
        original resume → add to the Skills section of the resume.
        Target: ATS score ≥ 75. Do not keyword-stuff to hit 80+.
        Document every addition in fixes_applied.

        STEP 3 — Consistency audit
        Call ConsistencyCheckerTool with parsed_resume_json (raw string), tailored resume,
        and cover letter.
        - DATE MISMATCH → correct the cover letter to match resume dates.
        - COVER LETTER DATE → must equal {today_date}. If any other date appears,
          replace it with {today_date}.
        - TITLE MISMATCH → use exact title from original resume.
        - METRIC MISMATCH → remove the metric from the cover letter; do not guess.
        - INVENTED STATISTIC → any number in the cover letter (e.g. "3,600 students",
          "serving X users") must be present verbatim in parsed_resume.json. If it is
          not found there, remove it and rephrase the sentence without the figure.
        - DROPPED EXPERIENCE → restore with 2 minimal bullets at the end (experienced
          candidates only — student candidates follow the 1-page rule from the tailor).
        - ASSERTIVE LANGUAGE → remove phrases like "non-negotiable", "you need",
          "must-have" from the cover letter. Rewrite to demonstrate the quality
          rather than assert a standard.
        - GENERIC CLOSE → if the close uses only generic adjectives ("helpful,
          organized, invested") with no company-specific detail, add one concrete
          reference to the role or company drawn from job_analysis.json.
        Document every correction.

        STEP 4 — Formatting lint
        Call FormattingLinterTool with resume, cover letter, and candidate name.
        Apply fixes directly for each issue:
        - Missing section header → add in correct position
        - Weak verb → replace with strongest applicable action verb
        - First-person in resume → rewrite without pronoun
        - Orphaned bullet → remove the empty line
        - Triple blank lines → reduce to single
        - Cliché in cover letter → rewrite sentence without it
        - Bullets in cover letter body → convert to prose
        - Generic opener → rewrite with value-prop hook using application_angle
        - RESUME LENGTH (student/entry-level candidates): count the number of
          distinct experience section headers. If the tailored resume still has
          more than 5 experience entries AND the candidate is student/entry-level
          (per parsed_resume.json), consolidate the excess thin entries (those
          with 1-2 bullets and no quantified metric) into a single "Campus
          Involvement" line. The final resume must render as 1 page.
        - SKILLS KEYWORD SOUP: if the Skills section contains more than 5 soft
          skills listed as plain text (e.g. "communication, teamwork, …"), trim
          to the 3 most relevant to the JD and remove the rest.
        Document every fix.

        STEP 5 — Gap detection
        Call GapDetectorTool with both documents, required_skills,
        and primary_responsibilities.
        - WEAK areas: add keyword to most appropriate bullet if obvious fit exists.
        - UNADDRESSED gaps: record in qa_report as candidate action items only.
          Do NOT fabricate.

        STEP 6 — Calculate composite score
        ATS score           × 0.35
        Consistency pass    × 0.25  (100 if passed, -15 per issue)
        Formatting pass     × 0.20  (100 if passed, -10 per issue)
        Gap coverage        × 0.20  (100 - gap_count×12 - weak_count×5)
        Clamp to [0, 100].

        Verdict:
        - Score ≥ 85 and 0 consistency issues → APPROVED
        - Score ≥ 70 and ≤ 2 consistency issues → APPROVED_WITH_FIXES
        - Score < 70 or critical gap → NEEDS_REVISION

        STEP 7 — Candidate recommendations
        Write 3-5 actionable recommendations beyond the documents
        (LinkedIn alignment, interview prep for gaps, portfolio hygiene, etc.)

        STEP 8 — Assemble final output using EXACTLY these delimiters:

        ===FINAL_RESUME===
        [Complete QA-approved tailored resume in markdown]
        ===END_RESUME===

        ===FINAL_COVER_LETTER===
        [Complete QA-approved cover letter in markdown]
        ===END_COVER_LETTER===

        ===QA_REPORT===
        {{
          "timestamp": "",
          "candidate_name": "",
          "target_role": "",
          "target_company": "",
          "ats_audit": {{
            "score": 0.0,
            "keywords_present": [],
            "keywords_missing": [],
            "keyword_density_warning": false,
            "recommended_additions": []
          }},
          "consistency_audit": {{
            "issues": [], "date_mismatches": [],
            "title_mismatches": [], "metric_mismatches": [], "passed": true
          }},
          "formatting_audit": {{
            "resume_issues": [], "cover_letter_issues": [], "passed": true
          }},
          "gap_audit": {{
            "unaddressed_requirements": [],
            "weak_evidence_areas": [],
            "mitigation_suggestions": []
          }},
          "overall_score": 0.0,
          "overall_verdict": "APPROVED",
          "fixes_applied": [],
          "recommendations": []
        }}
        ===END_REPORT===

        CRITICAL RULES:
        - Apply ONLY targeted, minimal fixes. No wholesale rewrites.
        - Every fix must be listed in fixes_applied.
        - If a fix requires candidate input, add it to recommendations instead.
        - The three delimited sections must be present for the output parser.
        - QA_REPORT must be valid JSON parseable with json.loads().
        """,

        expected_output=(
            "Three clearly delimited sections: "
            "===FINAL_RESUME=== ... ===END_RESUME===, "
            "===FINAL_COVER_LETTER=== ... ===END_COVER_LETTER===, "
            "===QA_REPORT=== ... ===END_REPORT===. "
            "Both documents are fully corrected and production-ready. "
            "QA report is valid JSON with all audit fields populated. "
            "All fixes documented in fixes_applied."
        ),

        context=[
            resume_parse_task,
            jd_analysis_task,
            resume_tailor_task,
            cover_letter_task,
        ],
        agent=qa_reviewer_agent,
        output_file="outputs/qa_full_output.txt",
    )

    return {
        "resume_parse":    resume_parse_task,
        "jd_analysis":     jd_analysis_task,
        "resume_tailor":   resume_tailor_task,
        "cover_letter":    cover_letter_task,
        "qa_review":       qa_review_task,
    }