# api/index.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
from app import app  # reuse your Flask app object
