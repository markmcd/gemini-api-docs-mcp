import sys
import os


from gemini_docs_mcp.server import sanitize_term

def test_sanitize_term():
    assert sanitize_term("google") == "google"
    assert sanitize_term("@google/genai") == '"@google/genai"'
    assert sanitize_term("gemini-2.5-flash") == '"gemini-2.5-flash"'

if __name__ == "__main__":
    test_sanitize_term()
