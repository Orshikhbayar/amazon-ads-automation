import os, sys
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
from app import app  # reuse Flask app that defines /api/retrieve and /api/generate
