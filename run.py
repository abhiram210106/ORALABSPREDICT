"""
run.py
------
Unified execution entrypoint for OralAbsPredict.
Supports starting the Streamlit dashboard, running the model training pipelines,
and executing the comprehensive test suite.
"""

import sys
import subprocess
import argparse


def launch_app():
    """Starts the Streamlit web dashboard."""
    print("Starting OralAbsPredict Streamlit Web Application...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app/app.py"]
    subprocess.run(cmd)


def train_models():
    """Runs the training pipelines for both HIA and HOB."""
    print("Initiating training for Human Intestinal Absorption (HIA)...")
    subprocess.run([sys.executable, "-m", "src.train_hia"], check=True)

    print("\nInitiating training for Human Oral Bioavailability (HOB)...")
    subprocess.run([sys.executable, "-m", "src.train_hob"], check=True)

    print("\nTraining completed successfully for all endpoints!")


def run_tests():
    """Executes the test suite using pytest."""
    print("Running OralAbsPredict test suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="OralAbsPredict — Unified CLI Launcher"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="app",
        choices=["app", "train", "test"],
        help="Action to execute: 'app' (launch Streamlit UI), 'train' (train ML models), 'test' (run tests)"
    )

    args = parser.parse_args()

    if args.action == "app":
        launch_app()
    elif args.action == "train":
        train_models()
    elif args.action == "test":
        run_tests()


if __name__ == "__main__":
    main()
