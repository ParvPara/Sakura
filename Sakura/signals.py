import re
import time
from typing import Dict, List

class AffectStore:
    def __init__(self):
        self.loneliness = 0.3
        self.social_charge = 0.5
        self.last_call_ts = 0.0
        self.last_dm_ts = 0.0
        self.boredom = 0.2
        self.engagement = 0.5

    def tick(self, events: Dict[str,bool]):
        self.social_charge = max(0.0, self.social_charge - 0.05)
        self.engagement = max(0.0, self.engagement - 0.03)
        
        if events.get("chat_quiet"):
            self.loneliness = min(1.0, self.loneliness + 0.1)
            self.boredom = min(1.0, self.boredom + 0.15)
        
        if events.get("positive"):
            self.social_charge = min(1.0, self.social_charge + 0.2)
            self.loneliness = max(0.0, self.loneliness - 0.1)
            self.engagement = min(1.0, self.engagement + 0.2)
        
        if events.get("conversation"):
            self.boredom = max(0.0, self.boredom - 0.1)
            self.engagement = min(1.0, self.engagement + 0.1)
        
        if events.get("tool_used"):
            self.engagement = min(1.0, self.engagement + 0.15)

    def mood(self) -> str:
        if self.loneliness > 0.7:
            return "lonely"
        elif self.boredom > 0.6:
            return "bored"
        elif self.engagement > 0.7:
            return "engaged"
        else:
            return "neutral"
    
    def should_reach_out(self, cooldown_seconds: float = 300) -> bool:
        now = time.time()
        if now - self.last_dm_ts < cooldown_seconds:
            return False
        return self.loneliness > 0.6 or self.boredom > 0.7

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", s.lower())

def novelty_score(terms: List[str], memory_terms: List[str]) -> float:
    mem = set(t.lower() for t in memory_terms)
    if not terms:
        return 0.0
    unseen = sum(1 for t in terms if t.lower() not in mem)
    return min(1.0, unseen / max(1, len(terms)))

def extract_proper_like(s: str) -> List[str]:
    props = re.findall(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b", s)
    return list(dict.fromkeys(props))

def has_question_words(s: str) -> bool:
    question_words = {"what","where","when","who","why","how","which","whose"}
    tokens = tokenize(s)
    return any(w in question_words for w in tokens)

def compute_signals(
    user_text: str, 
    memory_terms: List[str], 
    affect: AffectStore, 
    events: Dict[str,bool] = None
) -> Dict[str,object]:
    affect.tick(events or {})
    
    toks = tokenize(user_text)
    props = extract_proper_like(user_text)
    
    nov = novelty_score(props, memory_terms)
    
    is_question = has_question_words(user_text) or user_text.strip().endswith("?")
    
    uncertainty = round(0.7*nov + 0.3*int(is_question), 2)
    
    return {
        "uncertainty": uncertainty,
        "novelty": round(nov, 2),
        "is_question": is_question,
        "affect": {
            "loneliness": round(affect.loneliness, 2),
            "boredom": round(affect.boredom, 2),
            "engagement": round(affect.engagement, 2),
            "mood": affect.mood()
        },
        "timestamps": {"now": time.time()},
        "proper_nouns": props[:5]
    }

def extract_memory_terms(memory_system) -> List[str]:
    try:
        if not memory_system:
            return []
        
        terms = []
        
        if hasattr(memory_system, 'people') and memory_system.people:
            for person in memory_system.people.values():
                if hasattr(person, 'name'):
                    terms.append(person.name)
        
        if hasattr(memory_system, 'conversation_history'):
            recent = memory_system.conversation_history[-50:] if len(memory_system.conversation_history) > 50 else memory_system.conversation_history
            for msg in recent:
                if isinstance(msg, dict) and 'content' in msg:
                    props = extract_proper_like(msg['content'])
                    terms.extend(props)
        
        return list(set(terms))
    except Exception as e:
        print(f"[SIGNALS] Error extracting memory terms: {e}")
        return []
