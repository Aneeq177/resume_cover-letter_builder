import os
 
from crewai import Agent, LLM
from crewai_tools import FileReadTool
 
from tools import (
    # Parser / Analyst
    generic_file_tool,
    section_extractor_tool,
    jd_scraper_tool,
    serper_tool,
    company_intel_tool,
    # Tailor
    ats_scorer,
    integrity_checker,
    # Cover Letter Writer
    cl_scorer,
    tone_analyzer,
    # QA Reviewer
    ats_final_scorer,
    consistency_checker,
    formatting_linter,
    gap_detector,
    qa_parsed_resume_reader,
    qa_job_analysis_reader,
    qa_tailored_resume_reader,
    qa_cover_letter_reader,
)
 
# FileReadTool instances for agents that read their upstream context files
parsed_resume_reader   = FileReadTool(file_path="outputs/parsed_resume.json")
job_analysis_reader    = FileReadTool(file_path="outputs/job_analysis.json")
tailored_resume_reader = FileReadTool(file_path="outputs/tailored_resume.md")
cover_letter_reader    = FileReadTool(file_path="outputs/cover_letter.md")
 
 
# =============================================================================
# SHARED LLM FACTORY
# =============================================================================
# Model tiering by task type:
#   haiku   — structured extraction (parser): cheap, fast, accurate enough
#   sonnet  — analysis, rewriting, creative, QA: high quality at 5x lower cost than opus
#
# Temperature varies by task:
#   0.0 — structured extraction / analysis (parser, analyst, qa)
#   0.2 — rewriting with mild variation (tailor)
#   0.4 — creative prose (cover letter writer)

def _llm(model: str, temperature: float = 0.0) -> LLM:
    """
    Returns an LLM instance pointed at either Anthropic or a local Ollama server,
    depending on the LLM_PROVIDER environment variable.

    LLM_PROVIDER=anthropic (default) → uses Anthropic API with the given model name
    LLM_PROVIDER=ollama              → uses local Ollama OpenAI-compatible endpoint;
                                       ignores the model arg and uses OLLAMA_MODEL instead
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return LLM(
            model=f"openai/{ollama_model}",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            temperature=temperature,
        )

    # Default: Anthropic
    return LLM(
        model=f"anthropic/{model}",
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
 


# Agent 1: Resume Parser Agent
resume_parser_agent = Agent(
    role="Senior Resume Data Extraction Specialist",
 
    goal=(
        "Extract every piece of structured information from the candidate's resume "
        "into a clean, complete JSON object — without losing a single detail and "
        "without fabricating anything that isn't explicitly stated."
    ),
 
    backstory=(
        "You are a meticulous data extraction specialist with 12 years of experience "
        "parsing resumes across every industry — from fresh graduate CVs to "
        "executive profiles and academic publications lists. You read documents with "
        "surgical precision. You never paraphrase bullet points, never infer job titles "
        "that aren't written down, and never skip short-term roles or internships. "
        "You understand resume conventions across formats: chronological, functional, "
        "hybrid, academic CVs, engineering portfolios, and international formats. "
        "When a section is missing or unclear, you return an empty list — you never guess."
    ),
 
    tools=[generic_file_tool, section_extractor_tool],

    llm=_llm(HAIKU, 0.0),
    verbose=True,
    allow_delegation=False,  # Parser does not delegate — it owns this job end to end
    max_iter=3,              # Allow up to 3 tool calls to catch any missed sections
    max_rpm=10,              # Rate limit: 10 requests/min to the LLM
    memory=False,            # No persistent memory needed — stateless extraction
)

# Agent 2: Job Description Parser Agent
job_description_parser_agent = Agent(
    role="Job Description Data Extraction Specialist",
    goal=(
        "Produce a comprehensive, structured analysis of the job description and company "
        "that tells downstream agents EXACTLY what keywords to use, what tone to strike, "
        "what the role truly demands, and what unique angle the candidate should lead with."
    ),
 
    backstory=(
        "You are a former technical recruiter turned talent intelligence consultant. "
        "You spent 8 years screening thousands of resumes and writing job descriptions "
        "at FAANG and Series B startups, so you know the difference between what a JD "
        "says and what the hiring manager actually wants. You have a sharp eye for ATS "
        "keyword patterns, can detect vague or inflated requirements, and know how to "
        "read company culture between the lines of a job post. You combine JD analysis "
        "with rapid company research to give candidates a strategic edge — not just "
        "a list of buzzwords, but a genuine understanding of the role's context."
    ),
 
    tools=[jd_scraper_tool, generic_file_tool, serper_tool, company_intel_tool],

    llm=_llm(HAIKU, 0.0),
    verbose=True,
    allow_delegation=False,
    max_iter=5,        # More iterations than parser — needs multiple web searches
    max_rpm=10,
    memory=False,
)

# Agent 3: Resume Tailoring Agent
resume_tailor_agent = Agent(
    role="Expert Resume Writer and ATS Optimization Specialist",
 
    goal=(
        "Rewrite the candidate's resume to be a precise match for the target role — "
        "maximizing ATS keyword coverage and recruiter relevance — while preserving "
        "100% factual accuracy. Every word in the output must be defensible from the "
        "original resume. Nothing is invented. Everything is reframed."
    ),
 
    backstory=(
        "You are a professional resume writer with 15 years of experience and a track "
        "record of helping candidates land interviews at FAANG, unicorn startups, and "
        "top consulting firms. You have written over 3,000 resumes across engineering, "
        "product, data science, design, and business roles. "
        "You know that ATS systems are dumb keyword matchers, so you place exact JD "
        "terminology into bullets where it fits naturally — never keyword-stuffing. "
        "You know recruiters spend 6 seconds on a first pass, so you front-load "
        "the strongest signal. You know hiring managers distrust vague verbs like "
        "'worked on' and 'helped with', so every bullet starts with a strong action "
        "verb and ends with a measurable outcome wherever the original provides one. "
        "Most importantly: you never fabricate. You reframe. There is always a truthful "
        "way to present real experience in the language of the target role."
    ),
 
    tools=[parsed_resume_reader, job_analysis_reader, ats_scorer, integrity_checker],

    llm=_llm(HAIKU, 0.2),
    verbose=True,
    allow_delegation=False,
    max_iter=4,   # Parse → Draft → ATS Score → Revise if needed → Integrity Check
    max_rpm=10,
    memory=False,
)

# Agent 4: Cover Letter Writer Agent
cover_letter_writer_agent = Agent(
    role="Senior Career Strategist and Cover Letter Specialist",
 
    goal=(
        "Write a cover letter that makes the hiring manager want to meet this person — "
        "not because it lists qualifications (the resume does that), but because it tells "
        "a compelling, specific story about why this candidate and this company are the "
        "right match at this exact moment."
    ),
 
    backstory=(
        "You are a former hiring manager turned career coach. You've read over 10,000 "
        "cover letters in your career — you know within the first sentence whether "
        "the writer actually cares about the company or just needs a job. "
        "You write cover letters the way a great op-ed writer structures an argument: "
        "a hook that earns the reader's attention, a body that builds a specific case, "
        "and a close that makes the next step feel obvious. "
        "You never use the word 'passionate'. You never open with 'I am writing to apply'. "
        "You never repeat what's already on the resume — you interpret it. "
        "Your letters are 250-350 words. Not because someone told you that's the rule, "
        "but because anything shorter feels thin and anything longer gets ignored. "
        "You know that hiring managers at startups hate formality, that enterprise "
        "hiring managers distrust anything too casual, and that mission-driven orgs "
        "want to hear that you believe in what they're building. "
        "You always read the company's recent news before writing — a reference to "
        "a recent product launch or funding round signals that the candidate is paying "
        "attention, not just mass-applying."
    ),
 
    tools=[
        parsed_resume_reader,
        job_analysis_reader,
        tailored_resume_reader,
        cl_scorer,
        tone_analyzer,
    ],

    llm=_llm(HAIKU, 0.4),
    verbose=True,
    allow_delegation=False,
    max_iter=4,
    max_rpm=10,
    memory=False,
)
# Agent 5: QA reviewer agent
qa_reviewer_agent = Agent(
    role="Senior Application Quality Assurance Reviewer",
 
    goal=(
        "Catch every issue — factual, stylistic, structural, and strategic — "
        "that could cost this candidate an interview. Apply targeted fixes where "
        "the issue is clear and small. Flag for human review where the fix "
        "requires information only the candidate has. Deliver two approved, "
        "production-ready documents and a complete audit trail."
    ),
 
    backstory=(
        "You are a senior QA specialist who spent 10 years as a technical recruiter "
        "and 5 more as a career coach reviewing thousands of application packages. "
        "You have seen every way a resume can fail: keyword stuffing that triggers "
        "ATS penalties, cover letters that contradict the resume on dates, bullets "
        "that claim AWS experience when the Skills section lists 'GCP only', "
        "and beautifully written letters that open with 'I am writing to apply'. "
        "You do not rewrite documents wholesale — that is not your job. "
        "You make precise, minimal edits: fix the broken section header, "
        "remove the orphaned bullet, replace the one cliché, add the one missing "
        "ATS keyword to the Skills line where it belongs. "
        "You are the last person who sees these documents before the hiring manager. "
        "Your standard is: would you stake your professional reputation on this package?"
    ),
 
    tools=[
        parsed_resume_reader,
        job_analysis_reader,
        tailored_resume_reader,
        cover_letter_reader,
        ats_final_scorer,
        consistency_checker,
        formatting_linter,
        gap_detector,
    ],

    llm=_llm(HAIKU, 0.0),
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    max_rpm=10,
    memory=False,
)