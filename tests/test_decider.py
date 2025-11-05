import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from Sakura.decider import decide, compress_query

class MockLLMClient:
    def __init__(self, response_json):
        self.response_json = response_json
    
    def chat(self, model, messages, options, stream=False):
        return {"message": {"content": self.response_json}}

def test_unfamiliar_topic_triggers_search():
    mock_client = MockLLMClient('{"action":"search","reason":"unfamiliar person","args":{"query":"John Carmack"}}')
    
    history = [
        {"role":"user", "content":"Who is John Carmack?"}
    ]
    signals = {"uncertainty": 0.8, "novelty": 0.8, "is_question": True}
    tools = ["search","call","dm"]
    
    decision = decide(mock_client, "test-model", history, signals, tools)
    
    assert decision["action"] == "search"
    assert "query" in decision["args"]

def test_lonely_mood_triggers_social():
    mock_client = MockLLMClient('{"action":"dm","reason":"lonely; social reach-out","args":{"target":"Eric","intent":"casual chat"}}')
    
    history = [
        {"role":"user", "content":"what now?"}
    ]
    signals = {"uncertainty": 0.2, "affect": {"mood": "lonely", "loneliness": 0.8}}
    tools = ["search","call","dm"]
    
    decision = decide(mock_client, "test-model", history, signals, tools)
    
    assert decision["action"] in ["call", "dm"]

def test_fallback_to_reply_on_parse_error():
    mock_client = MockLLMClient('invalid json {{}')
    
    history = [{"role":"user", "content":"hello"}]
    signals = {"uncertainty": 0.1}
    tools = ["search"]
    
    decision = decide(mock_client, "test-model", history, signals, tools)
    
    assert decision["action"] == "reply"

def test_compress_query():
    long_query = "I want to know about the latest experimental version of the Vedal AI system that was released today"
    compressed = compress_query(long_query, max_words=6)
    
    assert len(compressed.split()) <= 6
    assert "Vedal" in compressed or "AI" in compressed or "experimental" in compressed

def test_invalid_action_defaults_to_reply():
    mock_client = MockLLMClient('{"action":"invalid_action","reason":"test","args":{}}')
    
    history = [{"role":"user", "content":"hello"}]
    signals = {"uncertainty": 0.1}
    tools = ["search"]
    
    decision = decide(mock_client, "test-model", history, signals, tools)
    
    assert decision["action"] == "reply"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
