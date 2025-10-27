# api/[...path].py — route all /api/* to your Flask app
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
from app import app  # expose the Flask WSGI app
