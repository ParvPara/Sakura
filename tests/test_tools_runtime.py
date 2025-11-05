import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from Sakura.tools_runtime import run_tool, format_tool_result_for_synthesis

def test_run_tool_search():
    def mock_search(query):
        return {"items": [{"title": "Test", "snippet": "Result"}]}
    
    decision = {"action": "search", "args": {"query": "test query"}}
    result = run_tool(decision, search_fn=mock_search)
    
    assert result["tool_used"] == "web_search"
    assert result["result"] is not None

def test_run_tool_call():
    def mock_call(target):
        return {"status": "initiated", "target": target}
    
    decision = {"action": "call", "args": {"target": "Eric"}}
    result = run_tool(decision, call_fn=mock_call)
    
    assert result["tool_used"] == "call"
    assert result["result"]["status"] == "initiated"

def test_run_tool_dm():
    def mock_dm(target, intent):
        return {"status": "sent", "target": target, "intent": intent}
    
    decision = {"action": "dm", "args": {"target": "Eric", "intent": "test"}}
    result = run_tool(decision, dm_fn=mock_dm)
    
    assert result["tool_used"] == "dm"
    assert result["result"]["status"] == "sent"

def test_run_tool_reply_no_tool():
    decision = {"action": "reply"}
    result = run_tool(decision)
    
    assert result["tool_used"] is None
    assert result["result"] is None

def test_format_tool_result_search():
    result = {
        "items": [
            {"title": "Title 1", "snippet": "Snippet 1"},
            {"title": "Title 2", "snippet": "Snippet 2"}
        ]
    }
    
    formatted = format_tool_result_for_synthesis("web_search", result)
    
    assert "Title 1" in formatted
    assert "Snippet 1" in formatted

def test_format_tool_result_error():
    result = {"error": "Something went wrong"}
    
    formatted = format_tool_result_for_synthesis("web_search", result)
    
    assert "failed" in formatted.lower()
    assert "Something went wrong" in formatted

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
