"""
Loan Prediction Web Application Entrypoint.
Starts the Flask server for the interactive loan assessment UI.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print("  LOAN PREDICTION AI — WEB APPLICATION & API")
    print(f"  Local URL: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=False)
