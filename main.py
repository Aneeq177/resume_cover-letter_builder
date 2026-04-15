# =============================================================================
# main.py — Resume Crew Pipeline Entrypoint
# =============================================================================
# Wires all five agents into a sequential crew, kicks off the pipeline,
# parses the QA agent's delimited output into final files, and prints
# a full run summary to the terminal.
#
# Usage:
#   python main.py --resume resume.pdf --job "https://jobs.lever.co/acme/role"
#   python main.py --resume resume.pdf --job job_description.txt
#   python main.py --resume resume.pdf --job job_description.txt --human-review
#
# Outputs written to ./outputs/:
#   parsed_resume.json        ← Resume Parser output
#   job_analysis.json         ← Job Analyst output
#   tailored_resume.md        ← Resume Tailor output
#   cover_letter.md           ← Cover Letter Writer output
#   qa_full_output.txt        ← Raw QA agent output (kept for debugging)
#   final_resume.md           ← QA-approved resume
#   final_cover_letter.md     ← QA-approved cover letter
#   qa_report.json            ← Full audit scorecard
# =============================================================================

import argparse
import io
import json
import os
import re
import sys

# Force UTF-8 stdout on Windows so Unicode characters (emojis, box-drawing) print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env before any CrewAI / LangChain imports
load_dotenv()

from crewai import Crew, Process

from agents import (
    resume_parser_agent,
    job_description_parser_agent,
    resume_tailor_agent,
    cover_letter_writer_agent,
    qa_reviewer_agent,
)
from task import build_tasks


# =============================================================================
# 1. CONSTANTS
# =============================================================================

OUTPUTS_DIR   = Path("outputs")
REQUIRED_ENVS = ["ANTHROPIC_API_KEY", "SERPER_API_KEY", "VOYAGE_API_KEY"]


# =============================================================================
# 2. PRE-FLIGHT CHECKS
# =============================================================================

def check_environment() -> list[str]:
    """Returns a list of missing required environment variables."""
    return [key for key in REQUIRED_ENVS if not os.getenv(key)]


def check_resume_file(resume_path: str) -> Optional[str]:
    """Returns an error message if the resume file is missing or unreadable."""
    path = Path(resume_path)
    if not path.exists():
        return f"Resume file not found: {resume_path}"
    if not path.is_file():
        return f"Resume path is not a file: {resume_path}"
    if path.suffix.lower() not in {".pdf", ".txt", ".docx"}:
        return f"Unsupported resume format: {path.suffix}. Use .pdf, .txt, or .docx"
    if path.stat().st_size == 0:
        return f"Resume file is empty: {resume_path}"
    return None


def ensure_output_dir() -> None:
    """Creates the outputs/ directory if it doesn't exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_resume_text(resume_path: str) -> str:
    """
    If the resume is a PDF, extracts its text via pymupdf and writes it to
    outputs/resume_text.txt. Returns the path agents should use to read the resume.
    For .txt files, returns the original path unchanged.
    """
    path = Path(resume_path)
    if path.suffix.lower() != ".pdf":
        return resume_path

    try:
        import fitz
        doc = fitz.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        out_path = OUTPUTS_DIR / "resume_text.txt"
        out_path.write_text(text, encoding="utf-8")
        return str(out_path)
    except ImportError:
        print("  WARNING: pymupdf not installed — agents will attempt to read PDF directly.")
        return resume_path


# =============================================================================
# 3. OUTPUT PARSER
# =============================================================================

def parse_and_save_qa_output(
    raw_output: str,
    resume_path:       str = "outputs/final_resume.md",
    cover_letter_path: str = "outputs/final_cover_letter.md",
    report_path:       str = "outputs/qa_report.json",
) -> Dict[str, Any]:
    """
    Splits the QA agent's delimited output into three separate files:
      final_resume.md, final_cover_letter.md, qa_report.json

    Returns a summary dict with save status, verdict, score, and any errors.
    """
    results: Dict[str, Any] = {
        "resume_saved":       False,
        "cover_letter_saved": False,
        "report_saved":       False,
        "verdict":            "UNKNOWN",
        "overall_score":      0.0,
        "ats_score":          0.0,
        "fixes_applied":      [],
        "recommendations":    [],
        "errors":             [],
    }

    # ── Extract resume ────────────────────────────────────────────────────────
    resume_match = re.search(
        r"===FINAL_RESUME===(.*?)===END_RESUME===",
        raw_output, re.DOTALL
    )
    if resume_match:
        content = resume_match.group(1).strip()
        Path(resume_path).write_text(content, encoding="utf-8")
        results["resume_saved"] = True
    else:
        results["errors"].append(
            "===FINAL_RESUME=== delimiter not found in QA output. "
            "Check outputs/qa_full_output.txt for the raw response."
        )

    # ── Extract cover letter ──────────────────────────────────────────────────
    cl_match = re.search(
        r"===FINAL_COVER_LETTER===(.*?)===END_COVER_LETTER===",
        raw_output, re.DOTALL
    )
    if cl_match:
        content = cl_match.group(1).strip()
        Path(cover_letter_path).write_text(content, encoding="utf-8")
        results["cover_letter_saved"] = True
    else:
        results["errors"].append(
            "===FINAL_COVER_LETTER=== delimiter not found in QA output."
        )

    # ── Extract and parse QA report ───────────────────────────────────────────
    report_match = re.search(
        r"===QA_REPORT===(.*?)===END_REPORT===",
        raw_output, re.DOTALL
    )
    if report_match:
        report_raw = report_match.group(1).strip()
        # Strip accidental markdown fences
        report_raw = re.sub(r"^```json\s*", "", report_raw, flags=re.MULTILINE)
        report_raw = re.sub(r"\s*```$",     "", report_raw, flags=re.MULTILINE)

        try:
            report = json.loads(report_raw)
            Path(report_path).write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            results["report_saved"]    = True
            results["verdict"]         = report.get("overall_verdict", "UNKNOWN")
            results["overall_score"]   = report.get("overall_score", 0.0)
            results["fixes_applied"]   = report.get("fixes_applied", [])
            results["recommendations"] = report.get("recommendations", [])
            results["ats_score"]       = (
                report.get("ats_audit", {}).get("score", 0.0)
            )
        except json.JSONDecodeError as e:
            results["errors"].append(f"QA report is not valid JSON: {e}")
            # Save the raw text for manual inspection
            raw_path = report_path.replace(".json", "_raw.txt")
            Path(raw_path).write_text(report_raw, encoding="utf-8")
            results["errors"].append(f"Raw report saved to {raw_path}")
    else:
        results["errors"].append("===QA_REPORT=== delimiter not found in QA output.")

    return results


# =============================================================================
# 4. RUN SUMMARY PRINTER
# =============================================================================

def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  RESUME CREW PIPELINE")
    print("=" * 60)


def print_section(title: str) -> None:
    print(f"\n-- {title} {'-' * (54 - len(title))}")


def print_run_summary(
    summary:    Dict[str, Any],
    start_time: float,
    inputs:     Dict[str, str],
) -> None:
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    print_section("RUN COMPLETE")
    print(f"  Duration:         {minutes}m {seconds}s")
    print(f"  Resume input:     {inputs.get('resume_path', '—')}")
    print(f"  Job input:        {inputs.get('job_input', '—')[:60]}...")

    print_section("DELIVERABLES")
    print(f"  Final resume:     {'✅  outputs/final_resume.md'       if summary['resume_saved']       else '❌  Not saved'}")
    print(f"  Cover letter:     {'✅  outputs/final_cover_letter.md' if summary['cover_letter_saved'] else '❌  Not saved'}")
    print(f"  QA report:        {'✅  outputs/qa_report.json'        if summary['report_saved']       else '❌  Not saved'}")

    print_section("QA SCORECARD")
    verdict_icon = {
        "APPROVED":            "✅",
        "APPROVED_WITH_FIXES": "✅",
        "NEEDS_REVISION":      "⚠️ ",
        "UNKNOWN":             "❓",
    }.get(summary["verdict"], "❓")

    print(f"  Verdict:          {verdict_icon}  {summary['verdict']}")
    print(f"  Overall score:    {summary['overall_score']:.1f} / 100")
    print(f"  ATS score:        {summary['ats_score']:.1f} / 100")

    if summary["fixes_applied"]:
        print_section(f"FIXES APPLIED ({len(summary['fixes_applied'])})")
        for fix in summary["fixes_applied"]:
            print(f"  · {fix}")

    if summary["recommendations"]:
        print_section(f"CANDIDATE RECOMMENDATIONS ({len(summary['recommendations'])})")
        for rec in summary["recommendations"]:
            print(f"  → {rec}")

    if summary["errors"]:
        print_section(f"⚠️  ERRORS ({len(summary['errors'])})")
        for err in summary["errors"]:
            print(f"  ! {err}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# 5. CREW BUILDER
# =============================================================================

def build_crew(human_review: bool = False) -> tuple[Crew, dict]:
    """
    Builds the sequential Crew and returns (crew, tasks_dict).

    human_review=True adds a pause before the QA task so the candidate
    can inspect the tailored resume and cover letter before final QA runs.
    (Implemented by setting human_input=True on the qa_review_task.)
    """
    tasks = build_tasks()

    if human_review:
        # CrewAI will pause and prompt the user for input before the QA task
        tasks["qa_review"].human_input = True

    crew = Crew(
        agents=[
            resume_parser_agent,
            job_description_parser_agent,
            resume_tailor_agent,
            cover_letter_writer_agent,
            qa_reviewer_agent,
        ],
        tasks=list(tasks.values()),
        process=Process.sequential,
        verbose=True,
        # memory=True enables cross-task recall — useful for long resumes
        # where the tailor or cover letter writer needs to re-reference earlier
        # context without making extra FileReadTool calls. Requires an embedder.
        memory=False,
    )

    return crew, tasks


# =============================================================================
# 6. MAIN ENTRYPOINT
# =============================================================================

def run(resume_path: str, job_input: str, human_review: bool = False) -> Dict[str, Any]:
    """
    Full pipeline run. Returns the parsed summary dict.

    Can also be called programmatically:
        from main import run
        summary = run("resume.pdf", "https://...", human_review=False)
    """
    start_time = time.time()
    inputs     = {
        "resume_path": resume_path,
        "job_input":   job_input,
        "today_date":  datetime.now().strftime("%B %d, %Y"),
    }

    print_banner()

    # ── Pre-flight ────────────────────────────────────────────────────────────
    print_section("PRE-FLIGHT CHECKS")

    missing_env = check_environment()
    if missing_env:
        print(f"  ❌  Missing environment variables: {missing_env}")
        print("      Add them to your .env file and re-run.")
        sys.exit(1)
    print("  ✅  Environment variables OK")

    resume_error = check_resume_file(resume_path)
    if resume_error:
        print(f"  ❌  {resume_error}")
        sys.exit(1)
    print(f"  ✅  Resume file found: {resume_path}")

    ensure_output_dir()
    print(f"  ✅  Output directory ready: {OUTPUTS_DIR}/")

    # Extract PDF text so FileReadTool can read it as plain text
    readable_resume_path = extract_resume_text(resume_path)
    if readable_resume_path != resume_path:
        print(f"  ✅  PDF extracted to: {readable_resume_path}")
    inputs["resume_path"] = readable_resume_path

    if human_review:
        print("  ℹ️   Human review mode ON — pipeline will pause before QA.")

    # ── Build and run crew ────────────────────────────────────────────────────
    print_section("STARTING PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  5 agents · sequential process\n")

    crew, tasks = build_crew(human_review=human_review)

    try:
        result = crew.kickoff(inputs=inputs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Pipeline failed with error:\n    {type(e).__name__}: {e}")
        print("\n    Partial outputs may have been saved to outputs/")
        print("    Check the verbose log above for the failing step.")
        raise

    # ── Parse and save QA output ──────────────────────────────────────────────
    print_section("PARSING QA OUTPUT")
    raw_output = str(result)
    summary    = parse_and_save_qa_output(raw_output)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_run_summary(summary, start_time, inputs)

    # ── Exit code ─────────────────────────────────────────────────────────────
    if summary["verdict"] == "NEEDS_REVISION":
        print("⚠️  Package flagged for revision. Review outputs/qa_report.json.")
        sys.exit(2)   # Non-zero exit so CI/CD pipelines can detect degraded runs
    elif summary["errors"]:
        print("⚠️  Pipeline completed with parsing errors. Check the error list above.")
        sys.exit(3)

    return summary


# =============================================================================
# 7. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="resume_crew",
        description=(
            "AI-powered resume tailoring pipeline.\n"
            "Takes a resume and a job posting, outputs a tailored resume "
            "and cover letter optimised for ATS and human readers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Job posting URL
  python main.py --resume resume.pdf --job https://jobs.lever.co/acme/123

  # Local JD file
  python main.py --resume resume.pdf --job job_description.txt

  # With human review pause before QA
  python main.py --resume resume.pdf --job job.txt --human-review

  # Programmatic (from another script)
  from main import run
  summary = run("resume.pdf", "https://...")
        """,
    )
    parser.add_argument(
        "--resume", "-r",
        required=True,
        metavar="PATH",
        help="Path to the candidate's resume (.pdf, .txt, or .docx)",
    )
    parser.add_argument(
        "--job", "-j",
        required=True,
        metavar="URL_OR_PATH",
        help="Job posting URL, local file path, or raw JD text",
    )
    parser.add_argument(
        "--human-review",
        action="store_true",
        default=False,
        help=(
            "Pause before the QA step so you can inspect the tailored resume "
            "and cover letter before final QA runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
        help="Directory for all output files (default: outputs/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Allow overriding the output directory via CLI
    if args.output_dir != "outputs":
        import tools as _tools_module
        # Patch the FileReadTool paths that reference outputs/
        # (only needed if you change output-dir from the default)
        print(f"ℹ️  Custom output directory: {args.output_dir}")
        OUTPUTS_DIR = Path(args.output_dir)

    run(
        resume_path=args.resume,
        job_input=args.job,
        human_review=args.human_review,
    )