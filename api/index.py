# api/index.py
import os, sys
# add project root to import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app  # re-use your existing Flask app from app.py
