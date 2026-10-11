"""
Main Pipeline Runner.
Executes data preprocessing, Scikit-Learn classical models training,
TensorFlow linear classifier benchmark, and model persistence.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_and_evaluate

def main():
    print("Starting Loan Prediction Machine Learning Pipeline...\n")
    try:
        metrics = train_and_evaluate()
        print("Pipeline execution completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
