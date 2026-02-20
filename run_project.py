#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   Content Integrity & Authorship Intelligence Platform             ║
║   Master Training & Integration Script                             ║
║                                                                    ║
║   One command to rule them all:                                    ║
║       python run_project.py                                        ║
║                                                                    ║
║   This runs Person 1 → 2 → 3 → 4 in the correct order.           ║
║   Person 1 MUST finish first (others depend on its data).          ║
║   Person 2 & 3 can run in parallel after Person 1.                 ║
║   Person 4 runs last (needs all three modules).                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import subprocess
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
PERSON1_DIR = PROJECT_ROOT / "person_1"
PERSON2_DIR = PROJECT_ROOT / "person_2"
PERSON3_DIR = PROJECT_ROOT / "person_3"
PERSON4_DIR = PROJECT_ROOT / "person_4"

# Lock for thread-safe printing
print_lock = threading.Lock()


def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


def run_script(script_path, cwd, description, log_file=None, extra_args=None):
    """
    Run a Python script in a given directory.
    Returns (success: bool, elapsed: float, description: str)
    """
    safe_print(f"\n{'━' * 70}")
    safe_print(f"  ▶ {description}")
    safe_print(f"    Script: {script_path}")
    safe_print(f"    Dir:    {cwd}")
    if log_file:
        safe_print(f"    Log:    {log_file}")
    safe_print(f"{'━' * 70}")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    try:
        if log_file:
            log_path = PROJECT_ROOT / "logs" / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as lf:
                result = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=None,
                )
        else:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                timeout=None,
            )

        elapsed = time.time() - start
        if result.returncode == 0:
            safe_print(f"  ✓ {description} — completed in {format_time(elapsed)}")
            return True, elapsed, description
        else:
            safe_print(f"  ✗ {description} — failed (exit code {result.returncode})")
            return False, elapsed, description

    except Exception as e:
        elapsed = time.time() - start
        safe_print(f"  ✗ {description} — error: {e}")
        return False, elapsed, description


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║   Content Integrity & Authorship Intelligence Platform             ║")
    print("║   ─────────────────────────────────────────────────                 ║")
    print("║   Master Training Pipeline                                         ║")
    print("║                                                                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def print_plan(parallel):
    mode = "PARALLEL (Person 2 & 3 run simultaneously)" if parallel else "SEQUENTIAL"
    print(f"  Execution mode: {mode}")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Phase 1: Person 1 — Data Pipeline & AI Detection          │")
    print("  │           Downloads datasets, preprocesses, trains 4       │")
    print("  │           models + meta-classifier, evaluates ensemble     │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    if parallel:
        print("  │  Phase 2: Person 2 & 3 — IN PARALLEL                      │")
        print("  │           Person 2: Plagiarism index + model training      │")
        print("  │           Person 3: Humanization model training            │")
    else:
        print("  │  Phase 2: Person 2 — Plagiarism Detection Engine           │")
        print("  │           Builds index, trains Sentence-BERT + CrossEnc    │")
        print("  ├─────────────────────────────────────────────────────────────┤")
        print("  │  Phase 3: Person 3 — Humanization Module                   │")
        print("  │           Trains Flan-T5, PEGASUS, Mistral-7B              │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │  Phase 4: Person 4 — Integration Verification              │")
    print("  │           Runs health check on the full pipeline            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()


def run_phase_1():
    """Person 1: Data pipeline + AI detection. Must run first."""
    return run_script(
        "run_all.py", PERSON1_DIR,
        "PHASE 1 — Person 1: Data Pipeline & AI Detection Ensemble"
    )


def run_person_2(parallel=False):
    """Person 2: Plagiarism detection."""
    log = "person2_training.log" if parallel else None
    return run_script(
        "run_all.py", PERSON2_DIR,
        "Person 2: Plagiarism Detection Engine",
        log_file=log,
    )


def run_person_3(parallel=False):
    """Person 3: Humanization."""
    log = "person3_training.log" if parallel else None
    return run_script(
        "run_all.py", PERSON3_DIR,
        "Person 3: Humanization Module",
        log_file=log,
        extra_args=["--auto"],
    )


def run_phase_2_parallel():
    """Run Person 2 and Person 3 in parallel."""
    safe_print(f"\n{'━' * 70}")
    safe_print("  ▶ PHASE 2 — Person 2 & 3 running in PARALLEL")
    safe_print(f"    Logs: logs/person2_training.log, logs/person3_training.log")
    safe_print(f"{'━' * 70}")

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_person_2, True): "person_2",
            executor.submit(run_person_3, True): "person_3",
        }
        for future in as_completed(futures):
            name = futures[future]
            success, elapsed, desc = future.result()
            results[name] = (success, elapsed, desc)

    return results


def run_phase_2_sequential():
    """Run Person 2 then Person 3 sequentially."""
    r2 = run_person_2(False)
    r3 = run_person_3(False)
    return {
        "person_2": r2,
        "person_3": r3,
    }


def run_phase_4_verification():
    """Quick verification that the integrated pipeline loads."""
    verify_script = PERSON4_DIR / "_verify_pipeline.py"

    # Create a small verification script
    verify_script.write_text(
        '"""Quick pipeline health check."""\n'
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "from src.pipeline import ContentIntegrityPipeline\n"
        "from src.config import load_config\n\n"
        "print('Initializing pipeline...')\n"
        "pipeline = ContentIntegrityPipeline(load_config())\n"
        "health = pipeline.health_check()\n"
        "print(f'Health: {health}')\n\n"
        "modules_ok = sum([health['ai_detector'], health['plagiarism_detector'], health['humanizer']])\n"
        "deplag_ok = health.get('deplagiarizer', False)\n"
        "formats = health.get('supported_formats', [])\n"
        "print(f'Modules loaded: {modules_ok}/3 (deplagiarizer: {deplag_ok})')\n"
        "print(f'Supported formats: {formats}')\n\n"
        "if modules_ok == 3:\n"
        "    print('All modules loaded successfully!')\n"
        "    # Quick smoke test\n"
        "    test_text = 'Artificial intelligence has transformed the way we interact with technology in our daily lives.'\n"
        "    print(f'Running smoke test on: {test_text[:60]}...')\n"
        "    report = pipeline.analyze(test_text, check_ai=True, check_plagiarism=True, humanize=False, deplagiarize=False)\n"
        "    print(f'AI Score: {report.get(\"ai_detection\", {}).get(\"score\", \"N/A\")}')\n"
        "    print(f'Plagiarism Score: {report.get(\"plagiarism\", {}).get(\"score\", \"N/A\")}')\n"
        "    print('Smoke test passed!')\n"
        "elif modules_ok > 0:\n"
        "    print('Partial modules loaded — some training may have failed.')\n"
        "    print('The pipeline will work in degraded mode for available modules.')\n"
        "else:\n"
        "    print('No modules loaded — training may not have completed.')\n"
        "    sys.exit(1)\n"
    )

    result = run_script(
        "_verify_pipeline.py", PERSON4_DIR,
        "PHASE 4 — Integration Verification & Smoke Test"
    )

    # Clean up temp script
    try:
        verify_script.unlink()
    except Exception:
        pass

    return result


def print_summary(results, total_time):
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                       TRAINING SUMMARY                             ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    for name, (success, elapsed, desc) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"║  {status}  {name:<20s}  {format_time(elapsed):>10s}  ║")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Total time: {format_time(total_time):>10s}                                       ║")

    all_passed = all(s for s, _, _ in results.values())
    if all_passed:
        print("║                                                                    ║")
        print("║  🎉 ALL PHASES COMPLETED SUCCESSFULLY                              ║")
        print("║                                                                    ║")
        print("║  You can now use the platform:                                     ║")
        print("║    cd person_4                                                     ║")
        print("║    python main.py --input 'your text' --full                       ║")
        print("║    python run_server.py   (for web UI at localhost:8000)            ║")
    else:
        print("║                                                                    ║")
        print("║  ⚠  Some phases failed. Check logs above for details.              ║")
        print("║     The pipeline will still work for modules that trained OK.       ║")

    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Master training script — runs all 4 persons in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_project.py                  # Run all (parallel mode for P2 & P3)
  python run_project.py --sequential     # Run all sequentially
  python run_project.py --skip-p1        # Skip Person 1 (data already ready)
  python run_project.py --only p1        # Run only Person 1
  python run_project.py --only p2 p3     # Run only Person 2 and 3
        """,
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run Person 2 & 3 sequentially instead of in parallel",
    )
    parser.add_argument(
        "--skip-p1", action="store_true",
        help="Skip Person 1 (use if datasets are already downloaded & preprocessed)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the final integration verification step",
    )
    parser.add_argument(
        "--only", nargs="+", choices=["p1", "p2", "p3", "p4"],
        help="Run only specific persons (e.g., --only p1 p2)",
    )

    args = parser.parse_args()
    parallel = not args.sequential

    print_banner()
    print_plan(parallel)

    total_start = time.time()
    all_results = {}

    # Determine what to run
    if args.only:
        run_set = set(args.only)
    else:
        run_set = {"p1", "p2", "p3", "p4"}
        if args.skip_p1:
            run_set.discard("p1")
        if args.skip_verify:
            run_set.discard("p4")

    # ── Phase 1: Person 1 ──
    if "p1" in run_set:
        success, elapsed, desc = run_phase_1()
        all_results["Person 1 (Data + AI Detection)"] = (success, elapsed, desc)
        if not success:
            print("\n  ⚠ Person 1 failed. Person 2 & 3 may not have data.")
            print("    Continuing anyway — they have their own downloaders.\n")

    # ── Phase 2: Person 2 & 3 ──
    run_p2 = "p2" in run_set
    run_p3 = "p3" in run_set

    if run_p2 and run_p3 and parallel:
        phase2_results = run_phase_2_parallel()
        for name, result in phase2_results.items():
            label = "Person 2 (Plagiarism)" if name == "person_2" else "Person 3 (Humanization)"
            all_results[label] = result
    else:
        if run_p2:
            r = run_person_2(False)
            all_results["Person 2 (Plagiarism)"] = r
        if run_p3:
            r = run_person_3(False)
            all_results["Person 3 (Humanization)"] = r

    # ── Phase 4: Verification ──
    if "p4" in run_set:
        success, elapsed, desc = run_phase_4_verification()
        all_results["Person 4 (Integration)"] = (success, elapsed, desc)

    total_time = time.time() - total_start
    print_summary(all_results, total_time)


if __name__ == "__main__":
    main()
