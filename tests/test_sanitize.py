import sys
import os

# Add the project root to sys.path to import gemini_docs_mcp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_docs_mcp.server import sanitize_term

def test_sanitize_term():
    assert sanitize_term("google") == "google"
    assert sanitize_term("@google/genai") == '"@google/genai"'
    assert sanitize_term("gemini-2.5-flash") == '"gemini-2.5-flash"'

if __name__ == "__main__":
    test_sanitize_term()
