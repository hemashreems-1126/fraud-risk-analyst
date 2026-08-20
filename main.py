"""
main.py
--------
Runs the whole AI Risk Manager pipeline, start to finish:

  1. generate_data.py   -> creates transactions.csv
  2. score_transactions.py -> flags the suspicious ones
  3. risk_agent.py       -> AI agent investigates each flagged one

This is the one file you'd point someone to and say "just run this."
"""

import subprocess
import sys


def run(script_name: str):
    print(f"\n{'=' * 50}")
    print(f"Running {script_name}")
    print("=" * 50)
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        raise SystemExit(f"{script_name} failed -- fix the error above before continuing.")


if __name__ == "__main__":
    run("generate_data.py")
    run("score_transactions.py")
    run("risk_agent.py")
    print("\nAll done. Check risk_report.csv for the final results.")
