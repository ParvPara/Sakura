import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from Sakura.signals import AffectStore, compute_signals, novelty_score, extract_proper_like, tokenize

def test_affect_store_loneliness_increases_when_quiet():
    affect = AffectStore()
    initial_loneliness = affect.loneliness
    
    for _ in range(5):
        affect.tick({"chat_quiet": True})
    
    assert affect.loneliness > initial_loneliness
    assert affect.loneliness <= 1.0

def test_affect_store_mood_becomes_lonely():
    affect = AffectStore()
    affect.loneliness = 0.8
    
    assert affect.mood() == "lonely"

def test_affect_store_positive_events_reduce_loneliness():
    affect = AffectStore()
    affect.loneliness = 0.8
    
    for _ in range(5):
        affect.tick({"positive": True})
    
    assert affect.loneliness < 0.8

def test_novelty_score_high_for_unseen_terms():
    memory_terms = ["Python", "JavaScript", "React"]
    new_terms = ["Rust", "Golang", "Svelte"]
    
    score = novelty_score(new_terms, memory_terms)
    
    assert score >= 0.5

def test_novelty_score_low_for_known_terms():
    memory_terms = ["Python", "JavaScript", "React"]
    known_terms = ["Python", "React"]
    
    score = novelty_score(known_terms, memory_terms)
    
    assert score <= 0.5

def test_extract_proper_nouns():
    text = "Vedal is working on Neuro and testing with Evil"
    proper_nouns = extract_proper_like(text)
    
    assert "Vedal" in proper_nouns
    assert "Neuro" in proper_nouns
    assert "Evil" in proper_nouns

def test_compute_signals_high_uncertainty_for_novel_query():
    affect = AffectStore()
    memory_terms = ["old", "known", "terms"]
    
    user_text = "What is NewTech?"
    signals = compute_signals(user_text, memory_terms, affect, {"conversation": True})
    
    assert signals["uncertainty"] >= 0.3
    assert signals["novelty"] > 0

def test_compute_signals_detects_questions():
    affect = AffectStore()
    memory_terms = []
    
    user_text = "What is the weather like?"
    signals = compute_signals(user_text, memory_terms, affect)
    
    assert signals["is_question"] == True

def test_tokenize():
    text = "Hello, this is a test-message with URLs https://example.com"
    tokens = tokenize(text)
    
    assert "hello" in tokens
    assert "test-message" in tokens
    assert len(tokens) > 0

def test_affect_store_engagement_increases_with_tools():
    affect = AffectStore()
    initial_engagement = affect.engagement
    
    affect.tick({"tool_used": True})
    
    assert affect.engagement > initial_engagement

def test_should_reach_out_respects_cooldown():
    affect = AffectStore()
    affect.loneliness = 0.8
    affect.last_dm_ts = 0.0
    
    assert affect.should_reach_out(cooldown_seconds=1) == True
    
    import time
    affect.last_dm_ts = time.time()
    assert affect.should_reach_out(cooldown_seconds=300) == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
