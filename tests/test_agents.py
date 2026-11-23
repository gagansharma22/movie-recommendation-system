import os
import sys
from unittest.mock import patch, MagicMock

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_scrape_url_html_parsing():
    """Verify that scrape_url strips scripts, styles, and extracts clean text."""
    from tools import scrape_url
    
    mock_html = """
    <html>
        <head><title>Test</title><style>.test { color: red; }</style></head>
        <body>
            <script>console.log("ignore me");</script>
            <nav>Navigation links</nav>
            <main>
                <h1>Autonomous AI Agents</h1>
                <p>Multi-agent research systems coordinate specialized LLM agents.</p>
            </main>
            <footer>Footer content</footer>
        </body>
    </html>
    """
    
    with patch("requests.Session") as mock_session:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = mock_html
        mock_session.return_value.get.return_value = mock_resp
        
        result = scrape_url.invoke({"url": "https://example.com/test"})
        assert "Autonomous AI Agents" in result
        assert "Multi-agent research systems coordinate specialized LLM agents" in result
        assert "console.log" not in result
        assert "Navigation links" not in result


def test_prompts_structure():
    """Verify writer and critic prompts have expected placeholders."""
    from agents import writer_prompt, critic_prompt
    
    writer_vars = writer_prompt.input_variables
    assert "topic" in writer_vars
    assert "research" in writer_vars
    
    critic_vars = critic_prompt.input_variables
    assert "report" in critic_vars


def test_web_search_error_handling():
    """Verify web_search handles network/API exceptions gracefully."""
    from tools import web_search
    
    with patch("tools.tavily.search", side_effect=Exception("API limit exceeded")):
        result = web_search.invoke({"query": "test query"})
        assert "Search encountered an issue" in result
