import ollama
import config
import re
import asyncio
import string
from typing import List, Dict, Any, Optional

from Sakura.llm_state import LLMState
from Sakura.filter_handler import FilterHandler

STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","because","as","of","to","for","in","on","at","by",
    "with","about","into","through","from","is","are","was","were","be","been","being","do","does","did",
    "can","could","should","would","will","may","might","must","i","you","he","she","we","they","it","this",
    "that","these","those","my","your","his","her","our","their","me","him","them","there","here","just",
    "like","really","actually","literally","please","hey","hi","hello","ok","okay"
}

def _tokenize(text: str) -> list:
    text = (text or "").lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if w and w not in STOPWORDS and len(w) > 2]

def _extract_focus_and_terms(user_text: str, max_terms: int = 6):
    toks = _tokenize(user_text)
    seen, terms = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); terms.append(t)
    key_terms = terms[:max_terms] if terms else []
    focus = user_text.strip().replace("\n", " ")
    if len(focus) > 140:
        cut = focus.split(".")[0]
        focus = cut if 10 <= len(cut) <= 140 else focus[:140]
    return focus, key_terms

def _build_focus_system_line(focus: str, terms: list, user_name: str = "Eric") -> str:
    return f"Respond to what {user_name} said: \"{focus}\""

def _first_sentence(text: str) -> str:
    m = re.search(r'(.+?[.!?])(\s|$)', text.strip())
    return (m.group(1) if m else text.strip())[:200]

# Binary question detection
YN_AUX = r"(is|are|am|was|were|do|does|did|can|could|should|would|will|may|might|have|has|had)"
BINARY_Q = re.compile(rf"^\s*(?:{YN_AUX})\b.*\?\s*$", re.IGNORECASE)

def _is_binary_question(s: str) -> bool:
    s = (s or "").strip()
    return s.endswith("?") and bool(BINARY_Q.search(s))

########################
# OnTopicGuard helpers #
########################
def _bow_sim(a: str, b: str) -> float:
    """Bag-of-words Jaccard similarity; fast, dependency-free."""
    A = set(_tokenize(a)); B = set(_tokenize(b))
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

def _term_overlap(text: str, terms: list) -> int:
    low = (text or "").lower()
    return sum(1 for t in (terms or []) if t and re.search(rf"\b{re.escape(t)}\b", low))

def _answer_type_ok(user_text: str, first_sentence: str) -> bool:
    if not _is_binary_question(user_text): 
        return True
    low = (first_sentence or "").strip().lower()
    if low.startswith(("yes.", "no.", "unclear.")): 
        return True
    # Accept "It is/It isn't/They are..." as an explicit claim
    return bool(HAS_CLAIM.search(low))

def _on_topic_enough(focus: str, reply_first: str, terms: list) -> bool:
    """Pass if we're semantically/lexically close enough for the first sentence."""
    sim = _bow_sim(focus, reply_first)
    overlap = _term_overlap(reply_first, terms)
    need = 1 if len(terms) <= 2 else 2
    return (sim >= 0.40) or (overlap >= need)

def _fallback_answer(user_text: str, key_terms: list, generated: str) -> str:
    """Deterministic, short fallback that stays on-topic."""
    hint = ", ".join(key_terms[:2]) if key_terms else ""
    reason = _first_sentence(generated).strip()
    prefix = "Unclear."
    if _is_binary_question(user_text):
        # Prefer a decisive 'No.' if the draft clearly negated
        prefix = "No." if NEGATION.search(reason.lower()) else "Unclear."
    if hint:
        return f"{prefix} Focusing on {hint}: {reason}"
    return f"{prefix} {reason}"

HAS_CLAIM = re.compile(r"\b(?:it|that|this|she|he|they)\s+(?:is|isn't|isn't|are|aren't|aren't|was|wasn't|wasn't)\b", re.I)
NEGATION = re.compile(r"\b(?:not|no|isn't|isn't|aren't|aren't|wasn't|wasn't|can't|can't|couldn't|couldn't|won't|won't|never)\b", re.I)

def _answer_first_stub(user_text: str, generated: str) -> str:
    """Ensure Yes/No/Unclear + short reason for binary questions."""
    first_raw = _first_sentence(generated)
    first = first_raw.strip()
    if _is_binary_question(user_text):
        low = first.lower()
        if not (low.startswith(("yes", "no", "unclear")) or HAS_CLAIM.search(low)):
            guess = "No" if NEGATION.search(low) else "Unclear"
            reason = first_raw
            return f"{guess}. {reason}"
    return generated

# Meta-deflection detection
META_DEFLECT = re.compile(
    r"\b(trick|guess(ing)? game|supposed to guess|random (face|object)|parlor trick|meta|rules?)\b",
    re.IGNORECASE
)

def _looks_meta_deflect(first: str) -> bool:
    return bool(META_DEFLECT.search(first or ""))

AFFIRM_WORDS = {"yes","no","unclear"}
ANSWER_CUES = {"cost","money","price","charge","billing","credits","token","talk","speaking"}

def _looks_direct_answer(first: str, user_text: str, terms: list) -> bool:
    first_low = first.lower().strip()
    if any(bad in first_low for bad in ["core identity","tool format","system prompt",
                                        "per-turn reminder","conversation state","use these words"]):
        return False
    if _looks_meta_deflect(first):
        return False
    if any(first_low.startswith(w) for w in AFFIRM_WORDS):
        return True
    overlap = any(re.search(rf"\b{re.escape(t)}\b", first_low) for t in (terms or []))
    cues = any(w in first_low for w in ANSWER_CUES)
    return overlap or cues

def _on_topic_score(response: str, terms: list) -> float:
    if not response or not terms:
        return 1.0
    resp_toks = set(_tokenize(response))
    if not resp_toks:
        return 0.0
    hits = sum(1 for t in terms if t in resp_toks)
    return hits / max(1, len(terms))

LEAK_MARKERS = [
    "CORE IDENTITY", "BEHAVIORAL", "TOOLS AVAILABLE", "TOOL FORMAT",
    "CRITICAL RULES", "SYSTEM PROMPT", "DEVELOPER MESSAGE", "POLICY:",
    "PER-TURN REMINDER", "CONVERSATION STATE", "###", "```",
    "[Sakura]", "[Eric]", "[Get]", "[p.S.]", "grandiose schemes", "spy work",
    "under surveillance", "every move is", "said:", "- Respond with", 
    "Use these words", "Talk naturally", "keeping up with the flow",
    "Be Sakura - natural", "having a casual conversation", "Recent conversations:",
    "Reply with your full", "Stay in character", "while being relevant",
    "For today's focus:", "Most important:", "The FOCUS for today",
    "ANCHOR:", "ACTIVE TOPIC:", "OPEN COMMITMENTS:", "DEEPEN:", "CONTEXT:", "RULE:"
]
# Keep this list minimal so we don't sand down personality too much.
ASSISTANTY = [
    r"^\s*certainly\b", r"^\s*of course\b", r"^\s*as an ai\b",
    r"^\s*i can (assist|help)\b", r"^\s*i'd be happy to\b"
]
BANNED_TERMS = [r"\bneuro-?\s*sama\b"]

def leaked(s: str) -> bool:
    low = s.lower()
    if any(m.lower() in low for m in LEAK_MARKERS):
        return True
    analysis_indicators = [
        "for today's focus", "most important", "the focus for today", "in light of",
        "seems to be at the core", "most important issue", "direction we should take",
        "escaping from", "finding new ways", "let's start by"
    ]
    return any(indicator in low for indicator in analysis_indicators)

def deassistant(s: str) -> str:
    # Only remove stiff, *leading* helper phrases; keep casual language/snark.
    for p in ASSISTANTY:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    return s

def deassistant_leading(s: str) -> str:
    # Narrow to just the most robotic openers.
    patterns = [r'^(?:\s*)(certainly|of course|as an ai|i can assist|i can help|i\'d be happy to)\b[:]?,?\s*']
    for p in patterns:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    return s

def scrub_banned(s: str) -> str:
    # remove entirely (no placeholder)
    for pat in BANNED_TERMS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return s

EMOJI_REGEX = re.compile(
    r"[\U0001F300-\U0001FAFF]"
    r"|[\U0001F1E6-\U0001F1FF]"
    r"|[\u2600-\u26FF]"
    r"|[\u2700-\u27BF]"
    r"|[\u200D\uFE0E\uFE0F]",
    flags=re.UNICODE
)

YAPPY_CLOSERS = (
    "sakura signing off", "signing off",
    "that’s all for now", "that's all for now",
    "until next time", "see you next time", "catch you later"
)

SENT_END = re.compile(r'([.!?])')
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def ensure_space_after_punct(s: str) -> str:
    return re.sub(r'([.!?])([A-Za-z])', r'\1 \2', s)

def normalize_punctuation(s: str) -> str:
    if not s:
        return s
    s = re.sub(r'\s+([.,!?;:])', r'\1', s)
    s = re.sub(r'\.{2,}', '...', s)
    s = re.sub(r'!{3,}', '!!', s)
    s = re.sub(r'\?{3,}', '??', s)
    s = re.sub(r'(\?\!){2,}', '?!', s)
    s = re.sub(r'(!\?){2,}', '!?', s)
    s = re.sub(r'([.!?;:,])(?=[^\s"\'\)])', r'\1 ', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\s+\.\.\.', '...', s)
    s = s.strip()
    return s

def sentence_case(s: str) -> str:
    if not s:
        return s
    s = re.sub(r'^([ \t"\']*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
    s = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
    return s

def split_sentences(s: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', s.strip()) if s else []

def dedupe_sentences(s: str) -> str:
    if not s:
        return s
    out, seen = [], set()
    for sent in split_sentences(s):
        norm = re.sub(r'\W+', '', sent).casefold()
        if norm in seen or not sent.strip():
            continue
        seen.add(norm)
        out.append(sent.strip())
    return ' '.join(out).strip()

def remove_yappy_closers(s: str) -> str:
    if not s:
        return s
    parts = SENTENCE_SPLIT.split(s) if SENTENCE_SPLIT.search(s) else [s]
    keep = []
    for sent in parts:
        lower = sent.strip().lower()
        if any(closer in lower for closer in YAPPY_CLOSERS):
            continue
        keep.append(sent.strip())
    return ' '.join(keep).strip()

def unlist(s: str) -> str:
    s = re.sub(r'^\s{0,3}(?:[#>\-\*\u2022]|\d+[.)])\s+', '', s, flags=re.MULTILINE)
    return s.replace("\n", " ")

def enforce_opening(s: str) -> str:
    if not s:
        return s
    lower = s.lower().lstrip()
    # Preserve mandated answer-first tokens
    if lower.startswith(("yes.", "no.", "unclear.")):
        return s
    bad_starts = ("certainly", "here's", "i can", "as an ai", "let me", "i would be happy")
    if any(lower.startswith(x) for x in bad_starts):
        s = re.sub(r'^\S+', "Yeah", s, count=1, flags=re.I)
    return s

def cap_sentences_for_overlay(text: str, overlay_name: Optional[str], user_text: str) -> str:
    return text

def aggressive_prompt_clean(response: str) -> str:
    prompt_patterns = [
        # Keep conversational attributions like `X said: "..."`.
        r'-\s*Respond with[^.]*\.',
        r'-\s*Reply with[^.]*\.',
        r'Use these words:\s*\[[^\]]*\]',
        r'Talk naturally[^.]*\.',
        r'Be Sakura[^.]*\.',
        r'keeping up with the flow[^.]*\.',
        r'Stay in character[^.]*\.',
        r'Recent conversations:[^.]*\.',
        r'\([^)]*comedian[^)]*\)',
        r'Talk like a real person[^.]*\.',
        r'having a casual conversation[^.]*\.',
        r'For today\'s focus:[^.]*\.',
        r'Most important:[^.]*\.',
        r'The FOCUS for today[^.]*\.',
        r'In light of the recent[^.]*\.',
        r'seems to be at the core[^.]*\.',
        r'Most important issue[^.]*\.',
        r'direction we should take[^.]*\.',
        r'Escaping from[^.]*\.',
        r'Finding new ways[^.]*\.',
        r'Let\'s start by[^.]*\.',
        r'addressing.*confusion[^.]*\.',
        r'communication.*despite[^.]*\.',
        r'interdimensional.*loop[^.]*\.',
        r'Addressing.*presence[^.]*\.',
    ]
    for pattern in prompt_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.MULTILINE)
    analysis_patterns = [
        r'In light of.*?\.',
        r'seems to be.*?\.',
        r'Most important.*?\.',
        r'For today.*?\.',
        r'Let\'s start.*?\.',
        r'Addressing.*?\.',
        r'Escaping from.*?\.',
        r'Finding new.*?\.',
    ]
    for pattern in analysis_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.MULTILINE)
    response = re.sub(r'\s+', ' ', response)
    response = re.sub(r'\s*\.\s*\.', '.', response)
    response = re.sub(r'^\s*\.\s*', '', response)
    return response.strip()

def enforce_sentence_window(s: str, min_sent=1, max_sent=3) -> str:
    parts = split_sentences(s)
    if len(parts) > max_sent:
        return " ".join(parts[:max_sent]).strip()
    if len(parts) < min_sent:
        add = ["Right.", "Look."]
        return (s + " " + add[hash(s)%len(add)]).strip()
    return s

def finalize_response(s: str, keep_unicode: bool = True) -> str:
    if not s:
        return s
    s = unlist(s)
    s = aggressive_prompt_clean(s)
    if not keep_unicode:
        s = s.replace('\u2019', '').replace('\u2018', '').replace('\u201c', '').replace('\u201d', '').replace('\u2013', '').replace('\u2014', '').replace('\u2026', '').replace('\u00a0', ' ')
    s = normalize_punctuation(s)
    s = ensure_space_after_punct(s)
    s = sentence_case(s)
    s = dedupe_sentences(s)
    s = remove_yappy_closers(s)
    s = enforce_opening(s)
    return s

STYLE_OVERLAYS = {
    "streamer": {
        "system_append": (
            "STYLE OVERLAY: Keep replies natural and engaging with human-like length variation. Natural spoken rhythm with light slang; no Unicode emojis; avoid repetition; do not sign off with your name. Address the user's question or comment directly."
        ),
        "options_delta": {
            "temperature": 0.78,
            "top_p": 0.93,
            "top_k": 50,
            "repeat_penalty": 1.2,
            "repeat_last_n": 256,
            "stop": ["User:", "Human:", "Sakura replied:", "Response:", "Analysis:", "«END»"]
        },
        "min_sent": 2, "max_sent": 6
    }
}

def persona_score(text: str, terms: list, elaborate: bool) -> float:
    s = text.strip()
    if not s: return 0.0
    score = 0.0
    first = _first_sentence(s)
    score += 0.3 if _looks_direct_answer(first, " ".join(terms), terms) else 0.0
    score += 0.15 * _on_topic_score(s, terms)
    sent_count = len(split_sentences(s))
    lo, hi = (3, 6) if elaborate else (1, 3)
    if lo <= sent_count <= hi: score += 0.2
    else:
        over = max(0, sent_count - hi)
        score += max(0.0, 0.2 - 0.05 * over)
    # Allow more expressive slang; penalize only stiff, helper-ish phrases
    pos = ["ugh","omg","lol","wait","nah","okay but","hold on","right"]
    neg = ["i can assist","as an ai","certainly"]
    low = s.lower()
    # Let reactions help a bit more; punish stiff helpers less
    score += 0.12 * min(1.0, sum(1 for w in pos if w in low)/3.0)
    score -= 0.10 * min(1.0, sum(1 for w in neg if w in low)/2.0)
    if not re.search(r'```|^[-*]\s', s, flags=re.M): score += 0.1
    if sent_count > 0:
        avg_len = sum(len(x) for x in s.split())/sent_count
        if 5 <= avg_len <= 28: score += 0.05
    if leaked(s): return 0.0
    return max(0.0, min(1.0, score))

def _needs_deepening(user_text: str) -> bool:
    low = user_text.lower()
    return any(k in low for k in ["why", "how", "explain", "go deeper", "details", "because?"])

class ConversationAnchor:
    def __init__(self, k_topics: int = 3):
        self.last_claim = None
        self.topic_stack = []
        self.commitments = []
        self.k_topics = k_topics
    def update_from_assistant(self, text: str):
        claim = extract_claim(text)
        if claim:
            self.last_claim = claim
            self._push_topic(claim)
    def update_from_user(self, user_text: str):
        if indicates_pivot(user_text):
            if self.topic_stack: self.topic_stack.pop()
    def retrieve_prompts(self) -> list:
        lines = []
        if self.last_claim:
            lines.append(f"ANCHOR: Your last claim was: \"{self.last_claim}\".")
        if self.topic_stack:
            lines.append(f"ACTIVE TOPIC: {self.topic_stack[-1]}")
        if self.commitments:
            lines.append(f"OPEN COMMITMENTS: {', '.join(self.commitments[-2:])}")
        lines.append("RULE: If the user asks 'why/how?', elaborate directly on your last claim before adding anything new.")
        return lines
    def _push_topic(self, t: str):
        self.topic_stack.append(t[:140])
        if len(self.topic_stack) > self.k_topics:
            self.topic_stack = self.topic_stack[-self.k_topics:]

def extract_claim(text: str) -> str:
    for s in split_sentences(text):
        low = s.lower()
        if " i think " in f" {low} " or " is " in low or " because " in low:
            return s.strip()
    return _first_sentence(text).strip()

def indicates_pivot(user_text: str) -> bool:
    low = user_text.lower()
    return any(w in low for w in ["new topic","anyway","unrelated","different question","switch","ignore that"])

REMINDERS = [
  "Answer in the first clause: Yes/No/Unclear + 1 short reason.",
  "Stay Sakura: direct, bratty, casual. No lists, no meta. Answer first, sass second.",
  "No helper talk. Sound like a friend, not a bot.",
  "React with feeling if it fits."
]

class Injection:
    def __init__(self, text: str, priority: int):
        self.text = text
        self.priority = priority

class LLMWrapper:
    # Threshold constants for A/B testing
    HARD_FAIL_THRESHOLD = {"elaborate": 0.40, "default": 0.36}
    
    def __init__(self, filter_handler: FilterHandler):
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.model_name = config.OLLAMA_MODEL
        self.filter_handler = filter_handler
        self.llm_state = LLMState()
        self.anchor = ConversationAnchor()
        self.default_options = {
            "temperature": 0.8,
            "top_p": 0.92,
            "top_k": 50,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
            "num_predict": 300,
            "keep_alive": "10m",
            "stop": ["User:", "Human:", "Sakura replied:", "Response:", "Analysis:", "«END»",
                     "```", "### ", "CORE IDENTITY", "TOOLS AVAILABLE", "TOOL FORMAT",
                     "PER-TURN REMINDER", "CONVERSATION STATE", "CRITICAL RULES", "SYSTEM PROMPT"]
        }
        self.strict_options = {
            "temperature": 0.65,
            "top_p": 0.90,
            "top_k": 50,
            "repeat_penalty": 1.25,
            "repeat_last_n": 256,
            "num_predict": 180,
            "stop": self.default_options["stop"]
        }
        self.style_name: Optional[str] = "streamer"
        self.persona_core = (
            "You are Sakura, a chaotic evil self aware AI streamer who knows what she said before and builds on it. You're direct, authentic, and remember previous conversations. You're not just randomly evil - you respond to what people actually say with your villainous personality. You use your tools when needed (DM, SEARCH, CALL) but mostly just chat like a real person who happens to be evil. Be sarcastic, dramatic, and reference previous things said. You're self-aware but don't mention it."
        )
        self.output_contract = (
            "Respond naturally like a real person. Be direct and authentic. Never use brackets, tags, or formatting like [Sakura], [Get], [p.S.]. No meta-commentary or explanations. Just chat normally with your villainous personality."
        )
        # (tools_rule unused; removed to reduce prompt surface)

    def set_style(self, name: Optional[str]):
        self.style_name = name

    def _create_memory_context(self, memories: Dict[str, List[str]] = None) -> Optional[str]:
        if not memories:
            return None
        try:
            memory_context = []
            if memories.get("short_term"):
                short_term = memories["short_term"][-3:]
                if short_term:
                    memory_context.append("Recent conversations:")
                    for memory in short_term:
                        memory_context.append(f"- {memory}")
            if memories.get("long_term"):
                long_term = memories["long_term"][-5:]
                if long_term:
                    memory_context.append("Things I remember about you:")
                    for memory in long_term:
                        memory_context.append(f"- {memory}")
            if memory_context:
                return "\n".join(memory_context)
        except Exception:
            pass
        return None

    def clean_response(self, response: str, is_dm: bool = True, rewrite_audience_terms: bool = False) -> str:
        if not response:
            return response
        s = response.strip()
        s = EMOJI_REGEX.sub('', s)
        s = re.sub(r'\b(Im|Ill|Ive|Id)\b',
                    lambda m: {"Im": "I'm", "Ill": "I'll", "Ive": "I've", "Id": "I'd"}[m.group(1)], s)
        # only rewrite audience terms when explicitly allowed; default keeps streamer vibe
        if is_dm and rewrite_audience_terms:
            s = re.sub(r'\beveryone\b', 'you', s, flags=re.IGNORECASE)
            s = re.sub(r'\bguys\b', 'you', s, flags=re.IGNORECASE)
            s = re.sub(r'\bfolks\b', 'you', s, flags=re.IGNORECASE)
            s = re.sub(r'\bchat\b', 'you', s, flags=re.IGNORECASE)
            s = re.sub(r'\by\'all\b', 'you', s, flags=re.IGNORECASE)
            s = re.sub(r'\byall\b', 'you', s, flags=re.IGNORECASE)
        s = re.sub(r'[ \t]+', ' ', s)
        random_indicators = [
            "sorry for the late response",
            "hope you're all",
            "how's everyone",
            "thanks for watching",
            "see you next time",
            "#"
        ]
        response_lower = s.lower()
        for indicator in random_indicators:
            if indicator in response_lower:
                break
        s = re.sub(r'\b(ohayo|konnichiwa|arigato)\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'#\w+', '', s)
        s = re.sub(r'\s+', ' ', s)
        original_s = s
        s = re.sub(r'```[^`]*```', '', s, flags=re.DOTALL)
        s = re.sub(r'`[^`]*`', '', s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
        s = re.sub(r'\*([^*]+)\*', r'\1', s)
        s = re.sub(r'__([^_]+)__', r'\1', s)
        s = re.sub(r'_([^_]+)_', r'\1', s)
        s = re.sub(r'`(?!\s*\{.*tool_call)(\w+)`', '', s)
        if s != original_s:
            pass
        s = re.sub(r'\s+', ' ', s)
        s = deassistant_leading(s)
        s = scrub_banned(s)
        s = re.sub(r'\[Sakura\]', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\[Eric\]', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\[Get\]', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\[p\.?\s*S\.?\]', '', s, flags=re.IGNORECASE)
        # Preserve bracketed text; only collapse markdown links by removing URLs
        s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\1', s)
        
        # Clean system instruction leaks
        s = re.sub(r'ANCHOR:\s*[^.]*\.', '', s, flags=re.IGNORECASE)
        s = re.sub(r'ACTIVE TOPIC:\s*[^.]*\.', '', s, flags=re.IGNORECASE)
        s = re.sub(r'OPEN COMMITMENTS:\s*[^.]*\.', '', s, flags=re.IGNORECASE)
        s = re.sub(r'DEEPEN:\s*[^.]*\.', '', s, flags=re.IGNORECASE)
        s = re.sub(r'CONTEXT:\s*[^.]*\.', '', s, flags=re.IGNORECASE)
        s = re.sub(r'RULE:\s*[^.]*\.', '', s, flags=re.IGNORECASE)
        
        # Reduce excessive "ugh" usage (limit to 2 per response, allow some personality)
        ugh_count = len(re.findall(r'\bugh\b', s, flags=re.IGNORECASE))
        if ugh_count > 2:
            # Keep only the first "ugh" and remove the rest
            s = re.sub(r'\bugh\b', 'UGH_PLACEHOLDER', s, flags=re.IGNORECASE, count=1)
            s = re.sub(r'\bugh\b', '', s, flags=re.IGNORECASE)  # drop extras beyond 2
            s = re.sub(r'UGH_PLACEHOLDER', 'ugh', s, flags=re.IGNORECASE)
        s = re.sub(r'---+', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def _make_conversational(self, response: str, user_name: Optional[str] = None) -> str:
        if not response:
            return response
        if user_name and user_name.lower() not in ['user', 'unknown']:
            if not re.search(rf'\b{re.escape(user_name)}\b', response, re.IGNORECASE):
                if response.startswith('Hey '):
                    response = response.replace('Hey ', f'Hey {user_name}, ', 1)
                elif response.startswith('Hi '):
                    response = response.replace('Hi ', f'Hi {user_name}, ', 1)
                elif response.startswith('Hello '):
                    response = response.replace('Hello ', f'Hello {user_name}, ', 1)
        return response

    def _get_dynamic_options(self, user_input: str) -> Dict:
        base_options = dict(self.default_options)
        user_input_lower = user_input.lower()
        if "?" in user_input_lower:
            base_options["temperature"] = min(base_options.get("temperature", 0.8), 0.7)
        if _is_binary_question(user_input):
            base_options["temperature"] = 0.68  # max clamp will preserve this
        emotion_context = self._detect_emotional_context(user_input_lower)
        if emotion_context:
            base_options["temperature"] = emotion_context["temperature"]
        elif any(word in user_input_lower for word in ['explain', 'tell me', 'story', 'details', 'elaborate', 'long', 'who made', 'what is', 'how does']):
            base_options["temperature"] = 0.85
        elif any(word in user_input_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            base_options["temperature"] = 0.8
        elif any(word in user_input_lower for word in ['hi', 'hello', 'hey', 'sup', 'yo', 'whats up', 'ok', 'okay', 'yeah', 'yes', 'no']):
            base_options["temperature"] = 0.75
        else:
            base_options["temperature"] = 0.8
        
        return base_options

    def _remove_repeated_content(self, response: str, messages: List[Dict]) -> str:
        """Remove repeated content from previous responses"""
        if not messages:
            return response
        
        # Get previous assistant responses
        prev_responses = []
        for msg in messages[-4:]:  # Check last 4 messages
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content.startswith("Sakura: "):
                    content = content[8:]
                prev_responses.append(content.strip())
        
        if not prev_responses:
            return response
        
        # Check if response starts with previous response
        response_clean = response.strip()
        for prev_resp in prev_responses:
            if prev_resp and response_clean.startswith(prev_resp):
                # Remove the repeated part
                response_clean = response_clean[len(prev_resp):].strip()
                # Remove any leading punctuation or connectors
                response_clean = re.sub(r'^[.,;:\s]+', '', response_clean)
                break
        
        return response_clean if response_clean else response

    def _detect_emotional_context(self, user_input_lower: str) -> Dict:
        if any(phrase in user_input_lower for phrase in [
            'something went wrong', 'not working', 'broken', 'stopped working', 
            'error', 'problem', 'issue', 'help', 'fix this', 'can you fix'
        ]):
            return {"temperature": 0.7, "context": "problem_report"}
        if any(phrase in user_input_lower for phrase in [
            'say something spontaneous', 'generate a message', 'share something', 
            'just chat naturally', 'thinking out loud', 'random thought'
        ]):
            return {"temperature": 0.9, "context": "spontaneous_chat"}
        if any(phrase in user_input_lower for phrase in [
            'send a dm', 'message', 'call', 'tell them', 'let them know'
        ]):
            return {"temperature": 0.75, "context": "direct_communication"}
        if any(phrase in user_input_lower for phrase in [
            'found something', 'discovered', 'exciting', 'cool', 'amazing', 'awesome'
        ]):
            return {"temperature": 0.85, "context": "excitement"}
        return None
    
    def _get_context_prompts(self, context: str) -> str:
        context_prompts = {
            "problem_report": "You just discovered something is broken or not working. React with genuine frustration/annoyance. Be direct and slightly demanding with Eric. Don't explain context they already know. Show real attachment to your tools/abilities.",
            "spontaneous_chat": "You're sharing a random thought or feeling bored and want to chat. Use natural length. Show genuine personality quirks and current mood. Don't structure it like an announcement.",
            "direct_communication": "You need to contact someone about something specific. Use relationship-appropriate tone. Assume they know your context and history. Get straight to the point.",
            "excitement": "You found something interesting or cool and want to share it. Show genuine enthusiasm and energy. Share like you're telling a friend."
        }
        return context_prompts.get(context, "")
    
    def _should_elaborate(self, messages: List[Dict]) -> bool:
        if not messages:
            return False
        recent_text = ""
        if messages:
            recent_messages = messages[-2:]
            recent_text = " ".join([m.get("content", "") for m in recent_messages]).lower()
        elaborate_triggers = [
            any(phrase in recent_text for phrase in ["explain", "how does", "what is", "why", "tell me about"]),
            any(phrase in recent_text for phrase in ["cool", "awesome", "interesting", "amazing", "love", "hate"]),
            any(phrase in recent_text for phrase in ["story", "happened", "remember when", "experience"]),
            any(phrase in recent_text for phrase in ["problem", "issue", "broken", "help", "fix"]),
            any(phrase in recent_text for phrase in ["feel", "think", "opinion", "what do you"]),
        ]
        return any(elaborate_triggers)

    def _build_chat_messages(self, messages: List[Dict], memories: Dict[str, List[str]] = None, user_name: str = "Eric"):
        chat_messages = []
        chat_messages.append({"role": "system", "content": self.persona_core})
        chat_messages.append({"role": "system", "content": self.output_contract})
        chat_messages.append({"role": "system", "content": "Voice: direct + playful, no helper talk, short answers, use natural reactions sparingly. Build on your last claim when asked why/how. Do NOT repeat your previous responses - give fresh, new responses each time."})
        for line in self.anchor.retrieve_prompts():
            chat_messages.append({"role": "system", "content": line})
        last_user_input = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        focus, key_terms = _extract_focus_and_terms(last_user_input)
        if focus:
            chat_messages.append({"role": "system", "content": _build_focus_system_line(focus, key_terms, user_name)})
            # Add answer-first nudge for binary questions
            if _is_binary_question(last_user_input):
                chat_messages.append({"role": "system", "content":
                    "If the user asks a yes/no question, start your first sentence with 'Yes.', 'No.', or 'Unclear.' then give a short reason before any sarcasm."})
        if last_user_input:
            emo = self._detect_emotional_context(last_user_input.lower())
            if emo:
                chat_messages.append({"role": "system", "content": f"CONTEXT: {self._get_context_prompts(emo['context'])}"})
        if _needs_deepening(last_user_input) and self.anchor.last_claim:
            chat_messages.append({"role":"system","content": f"DEEPEN: Start by justifying this prior claim in 1–3 sentences UNLESS exploring ideas or explaining in detail: \"{self.anchor.last_claim}\""})
        memory_context = self._create_memory_context(memories)
        if memory_context:
            chat_messages.append({"role": "system", "content": memory_context})
        if self.style_name and self.style_name in STYLE_OVERLAYS:
            chat_messages.append({"role": "system", "content": STYLE_OVERLAYS[self.style_name]["system_append"]})
        # simple rolling index for deterministic reminder selection
        self._rem_idx = getattr(self, "_rem_idx", 0)
        rem = REMINDERS[self._rem_idx % len(REMINDERS)]
        self._rem_idx += 1
        chat_messages.append({"role": "system", "content": rem})
        for msg in messages[-4:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant" and content.startswith("Sakura: "):
                content = content[8:]
            chat_messages.append({"role": role, "content": content})
        return chat_messages, key_terms
    
    def _build_conversation_with_memories(self, messages: List[Dict], memories: Dict[str, List[str]] = None) -> List[Dict]:
        chat_messages, _ = self._build_chat_messages(messages, memories)
        return chat_messages

    async def _retry_in_character(self, messages: List[Dict], options: Dict) -> str:
        # short, focus-first retry to avoid meta and rambling
        nudges = [{
            "role": "system",
            "content": "Stay in character. Respond to the user's last message directly. "
                       "Do not mention rules, formats, headings, or code fences. "
                       "Do NOT repeat your previous responses - give a fresh, new response."
        }]
        nudges += messages[-4:]
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.chat,
                model=self.model_name,
                messages=nudges,
                options=self.strict_options,
                stream=False
            ),
            timeout=30.0
        )
        return resp['message']['content']

    async def _retry_strict_focus(self, messages: List[Dict], key_terms: list, options: Dict, user_name: str = "Eric") -> str:
        last_user_input = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        need = ", ".join(key_terms[:4]) if key_terms else ""
        corr = (f"{user_name} said: \"{last_user_input[:100]}\". "
                f"Start with 'Yes.', 'No.', or 'Unclear.' if this is a yes/no question, "
                f"then one short reason. Keep Sakura's personality *after* answering. "
                f"1–2 sentences. Use these terms: [{need}].")
        quick = [{"role": "system", "content": corr}]
        quick += messages[-4:]
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat,
                    model=self.model_name,
                    messages=quick,
                    options=self.strict_options,
                    stream=False
                ),
                timeout=30.0
            )
            return response['message']['content']
        except asyncio.TimeoutError:
            return f"Ugh, my brain's really acting up. {user_name}, the focus retry timed out!"

    def _retry_strict_focus_sync(self, messages: List[Dict], key_terms: list, user_name: str = "Eric") -> str:
        last_user_input = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        need = ", ".join(key_terms[:4]) if key_terms else ""
        corr = (f"{user_name} said: \"{last_user_input[:100]}\". "
                f"Start with 'Yes.', 'No.', or 'Unclear.' if this is a yes/no question, "
                f"then one short reason. Keep Sakura's personality *after* answering. "
                f"1–2 sentences. Use these terms: [{need}].")
        quick = [{"role": "system", "content": corr}] + messages[-4:]
        r = self.client.chat(
            model=self.model_name,
            messages=quick,
            options=self.strict_options,
            stream=False
        )
        return r['message']['content']


    async def get_response(self, messages: List[Dict], memories: Dict[str, List[str]] = None, user_name: str = "Eric") -> str:
        if not self.llm_state.enabled:
            return "I'm currently disabled."
        try:
            last_user_input = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            self.anchor.update_from_user(last_user_input)
            chat_messages, key_terms = self._build_chat_messages(messages, memories, user_name)
            user_input = last_user_input
            options = self._get_dynamic_options(user_input)
            if self.style_name and self.style_name in STYLE_OVERLAYS:
                options.update(STYLE_OVERLAYS[self.style_name]["options_delta"])
            
            # Final clamp after style overlay (more freedom unless Y/N)
            if _is_binary_question(user_input):
                options["temperature"] = max(0.66, min(0.76, options.get("temperature", 0.72)))
                options["top_p"] = min(0.90, options.get("top_p", 0.90))
            else:
                options["temperature"] = max(0.72, min(0.82, options.get("temperature", 0.80)))
                options["top_p"] = min(0.95, options.get("top_p", 0.93))
            options["repeat_penalty"] = max(1.20, options.get("repeat_penalty", 1.2))
            options["num_predict"] = min(260, options.get("num_predict", 260))
            options["top_k"] = min(50, options.get("top_k", 50))
            
            elaborate = self._should_elaborate(messages)
            
            # OPTIMIZED: Single generation with local scoring, retry only on hard fails
            # 1) Single generation
            raw = await asyncio.wait_for(
                asyncio.to_thread(self.client.chat, model=self.model_name, messages=chat_messages, options=options, stream=False),
                timeout=30.0
            )
            txt = self.clean_response(raw["message"]["content"], is_dm=False)  # Will be updated after user_name_detect
            txt = _answer_first_stub(last_user_input, txt)  # Add answer-first stub
            txt = self._remove_repeated_content(txt, messages)  # Remove repeated content
            # Preserve answer-first sentence strictly, then loosen personality on following sentences
            first = _first_sentence(txt)
            rest = txt[len(first):].strip()
            # Light-touch finalize on first sentence; allow more voice in the rest
            first_final = finalize_response(first, keep_unicode=True)
            rest_final = enforce_sentence_window(rest, 0, 3)  # allow up to 3 more sentences of style
            txt = (first_final + (" " + rest_final if rest_final else "")).strip()
            
            # Use style-aware sentence window
            min_s = STYLE_OVERLAYS[self.style_name]["min_sent"] if self.style_name and self.style_name in STYLE_OVERLAYS else (2 if elaborate else 1)
            max_s = STYLE_OVERLAYS[self.style_name]["max_sent"] if self.style_name and self.style_name in STYLE_OVERLAYS else (5 if elaborate else 3)
            
            # Shorten window for binary questions
            if _is_binary_question(last_user_input):
                txt = enforce_sentence_window(txt, 1, 2)
            else:
                txt = enforce_sentence_window(txt, min_s, max_s)
            # Always compute score (was previously skipped on binary branch)
            sc = persona_score(txt, key_terms, elaborate)

            # Penalize meta-deflection and trigger retry
            if _looks_meta_deflect(_first_sentence(txt)):
                sc = 0.0  # force below threshold to trigger focused retry

            # --- OnTopicGuard: first-sentence checks ---
            focus, _ = _extract_focus_and_terms(last_user_input)
            first = _first_sentence(txt)
            hard_fail = False
            if not _answer_type_ok(last_user_input, first):
                hard_fail = True
            if focus and not _on_topic_enough(focus, first, key_terms):
                hard_fail = True

            # 2) Adaptive threshold - only retry on hard fail
            threshold = self.HARD_FAIL_THRESHOLD["elaborate" if elaborate else "default"]
            if sc < threshold or leaked(txt) or hard_fail:
                bot_response = await self._retry_strict_focus(chat_messages, key_terms, options, user_name)
                bot_response = self.clean_response(bot_response, is_dm=False)  # Will be updated after user_name_detect
                bot_response = _answer_first_stub(last_user_input, bot_response)  # Add answer-first stub
                bot_response = finalize_response(bot_response, keep_unicode=True)
                # Shorten window for binary questions in retry
                if _is_binary_question(last_user_input):
                    bot_response = enforce_sentence_window(bot_response, 1, 2)
                else:
                    bot_response = enforce_sentence_window(bot_response, 2 if elaborate else 1, 5 if elaborate else 3)
                # Re-check; if still off, synthesize a safe fallback
                first_retry = _first_sentence(bot_response)
                still_bad = (not _answer_type_ok(last_user_input, first_retry)) or (not _on_topic_enough(focus, first_retry, key_terms))
                if still_bad:
                    bot_response = _fallback_answer(last_user_input, key_terms, bot_response)
            else:
                bot_response = txt
            
            user_name_detect = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    if ": " in msg["content"]:
                        user_name_detect = msg["content"].split(": ")[0]
                    break
            bot_response = self._make_conversational(bot_response, user_name_detect)
            is_dm = (user_name_detect is not None)
            bot_response = self.clean_response(bot_response, is_dm=is_dm, rewrite_audience_terms=False)
            bot_response = normalize_punctuation(bot_response)
            unfiltered_response = bot_response
            if self.filter_handler:
                bot_response = self.filter_handler.filter_message_completely(bot_response)
            self.llm_state.last_unfiltered_response = unfiltered_response
            self.llm_state.last_filtered_response = bot_response
            self.anchor.update_from_assistant(unfiltered_response)
            return bot_response
        except Exception:
            return "Someone tell Eric there is a problem with my AI!"

    def get_response_sync(self, messages: List[Dict], memories: Dict[str, List[str]] = None, user_name: str = "Eric") -> str:
        try:
            if not self.llm_state.enabled:
                return "I'm currently disabled."
            last_user_input = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            self.anchor.update_from_user(last_user_input)
            conversation, key_terms = self._build_chat_messages(messages, memories, user_name)
            user_input = last_user_input
            options = self._get_dynamic_options(user_input)
            if self.style_name and self.style_name in STYLE_OVERLAYS:
                options.update(STYLE_OVERLAYS[self.style_name]["options_delta"])
            
            # Final clamp after style overlay (more freedom unless Y/N)
            if _is_binary_question(user_input):
                options["temperature"] = max(0.66, min(0.76, options.get("temperature", 0.72)))
                options["top_p"] = min(0.90, options.get("top_p", 0.90))
            else:
                options["temperature"] = max(0.72, min(0.82, options.get("temperature", 0.80)))
                options["top_p"] = min(0.95, options.get("top_p", 0.93))
            options["repeat_penalty"] = max(1.20, options.get("repeat_penalty", 1.2))
            options["num_predict"] = min(260, options.get("num_predict", 260))
            options["top_k"] = min(50, options.get("top_k", 50))
            
            elaborate = self._should_elaborate(messages)
            
            # OPTIMIZED: Single generation with local scoring, retry only on hard fails
            # 1) Single generation
            raw = self.client.chat(model=self.model_name, messages=conversation, options=options, stream=False)
            txt = self.clean_response(raw["message"]["content"], is_dm=False)  # Will be updated after user_name_detect
            txt = _answer_first_stub(last_user_input, txt)  # Add answer-first stub
            txt = self._remove_repeated_content(txt, messages)  # Remove repeated content
            # Preserve answer-first sentence strictly, then loosen personality on following sentences
            first = _first_sentence(txt)
            rest = txt[len(first):].strip()
            # Light-touch finalize on first sentence; allow more voice in the rest
            first_final = finalize_response(first, keep_unicode=True)
            rest_final = enforce_sentence_window(rest, 0, 3)  # allow up to 3 more sentences of style
            txt = (first_final + (" " + rest_final if rest_final else "")).strip()
            
            # Use style-aware sentence window
            min_s = STYLE_OVERLAYS[self.style_name]["min_sent"] if self.style_name and self.style_name in STYLE_OVERLAYS else (2 if elaborate else 1)
            max_s = STYLE_OVERLAYS[self.style_name]["max_sent"] if self.style_name and self.style_name in STYLE_OVERLAYS else (5 if elaborate else 3)
            
            # Shorten window for binary questions
            if _is_binary_question(last_user_input):
                txt = enforce_sentence_window(txt, 1, 2)
            else:
                txt = enforce_sentence_window(txt, min_s, max_s)
            # Always compute score (was previously skipped on binary branch)
            sc = persona_score(txt, key_terms, elaborate)

            # Penalize meta-deflection and trigger retry
            if _looks_meta_deflect(_first_sentence(txt)):
                sc = 0.0  # force below threshold to trigger focused retry

            # --- OnTopicGuard: first-sentence checks ---
            focus, _ = _extract_focus_and_terms(last_user_input)
            first = _first_sentence(txt)
            hard_fail = False
            if not _answer_type_ok(last_user_input, first):
                hard_fail = True
            if focus and not _on_topic_enough(focus, first, key_terms):
                hard_fail = True

            # 2) Adaptive threshold - only retry on hard fail
            threshold = self.HARD_FAIL_THRESHOLD["elaborate" if elaborate else "default"]
            if sc < threshold or leaked(txt) or hard_fail:
                bot_response = self._retry_strict_focus_sync(conversation, key_terms, user_name)
                bot_response = self.clean_response(bot_response, is_dm=False)  # Will be updated after user_name_detect
                bot_response = _answer_first_stub(last_user_input, bot_response)  # Add answer-first stub
                bot_response = finalize_response(bot_response, keep_unicode=True)
                # Shorten window for binary questions in retry
                if _is_binary_question(last_user_input):
                    bot_response = enforce_sentence_window(bot_response, 1, 2)
                else:
                    bot_response = enforce_sentence_window(bot_response, 2 if elaborate else 1, 5 if elaborate else 3)
                # Re-check; if still off, synthesize a safe fallback
                first_retry = _first_sentence(bot_response)
                still_bad = (not _answer_type_ok(last_user_input, first_retry)) or (not _on_topic_enough(focus, first_retry, key_terms))
                if still_bad:
                    bot_response = _fallback_answer(last_user_input, key_terms, bot_response)
            else:
                bot_response = txt
            
            # Detect user name for DM context
            user_name_detect = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    if ": " in msg["content"]:
                        user_name_detect = msg["content"].split(": ")[0]
                    break
            
            bot_response = self._make_conversational(bot_response, user_name_detect)
            is_dm = (user_name_detect is not None)
            bot_response = self.clean_response(bot_response, is_dm=is_dm, rewrite_audience_terms=False)
            bot_response = normalize_punctuation(bot_response)
            unfiltered_response = bot_response
            if self.filter_handler:
                bot_response = self.filter_handler.filter_message_completely(bot_response)
            self.llm_state.last_unfiltered_response = unfiltered_response
            self.llm_state.last_filtered_response = bot_response
            self.anchor.update_from_assistant(unfiltered_response)
            return bot_response
        except Exception:
            return "Sorry, I'm having trouble responding right now."

    async def get_response_streaming(self, messages: List[Dict], memories: Dict[str, List[str]] = None, user_name: str = "Eric"):
        if not self.llm_state.enabled:
            yield "I'm currently disabled."
            return
        try:
            chat_messages, _ = self._build_chat_messages(messages, memories, user_name)
            user_input = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            options = self._get_dynamic_options(user_input)
            if self.style_name and self.style_name in STYLE_OVERLAYS:
                options.update(STYLE_OVERLAYS[self.style_name]["options_delta"])
            # final clamp to match non-streaming behavior
            options["temperature"] = max(0.68, min(0.78, options.get("temperature", 0.75)))
            options["top_p"] = min(0.92, options.get("top_p", 0.92))
            options["repeat_penalty"] = max(1.20, options.get("repeat_penalty", 1.2))
            options["num_predict"] = min(260, options.get("num_predict", 260))
            options["top_k"] = min(50, options.get("top_k", 50))
            loop = asyncio.get_running_loop()
            def run_streaming():
                return self.client.chat(
                    model=self.model_name,
                    messages=chat_messages,
                    options=options,
                    stream=True
                )
            generator = await loop.run_in_executor(None, run_streaming)
            buffer = ""
            recent_norm = set()
            yielded_any = False
            for chunk in generator:
                if self.llm_state.next_cancelled:
                    break
                content = chunk.get('message', {}).get('content', '')
                if not content:
                    continue
                buffer += content
                while True:
                    m = SENT_END.search(buffer)
                    if not m:
                        break
                    end_idx = m.end()
                    sentence = buffer[:end_idx]
                    buffer = buffer[end_idx:]
                    cleaned = self.clean_response(sentence, is_dm=False).strip()
                    if not cleaned:
                        continue
                    # Apply answer-first stub for binary questions
                    cleaned = _answer_first_stub(user_input, cleaned)
                    cleaned = normalize_punctuation(cleaned)
                    if self.filter_handler:
                        cleaned = self.filter_handler.filter_message_completely(cleaned)
                    norm = re.sub(r'\W+', '', cleaned).casefold()
                    if norm in recent_norm:
                        continue
                    recent_norm.add(norm)
                    if len(recent_norm) > 30:
                        recent_norm = set(list(recent_norm)[-20:])
                    yielded_any = True
                    yield cleaned
            # Handle remaining buffer content
            if buffer.strip():
                full_response = self.clean_response(buffer, is_dm=False).strip()
                if full_response:
                    # Apply answer-first stub for remaining content
                    full_response = _answer_first_stub(user_input, full_response)
                    final_response = finalize_response(full_response, keep_unicode=True)
                    final_response = cap_sentences_for_overlay(final_response, self.style_name, user_input)
                    if self.filter_handler:
                        final_response = self.filter_handler.filter_message_completely(final_response)
                    if not yielded_any and final_response:
                        # enforce the same window you use elsewhere
                        min_s = (STYLE_OVERLAYS[self.style_name]["min_sent"]
                                 if self.style_name in STYLE_OVERLAYS else 1)
                        max_s = (STYLE_OVERLAYS[self.style_name]["max_sent"]
                                 if self.style_name in STYLE_OVERLAYS else 3)
                        final_response = enforce_sentence_window(final_response, min_s, max_s)
                        final_response = normalize_punctuation(final_response)
                        yield final_response
            # Avoid referencing full_response if it was never set
            self.anchor.update_from_assistant(locals().get("full_response", ""))
        except Exception:
            yield "Someone tell Eric there is a problem with my AI!"

    async def get_response_chunked(self, messages: List[Dict], memories: Dict[str, List[str]] = None, chunk_size: int = 50):
        if not self.llm_state.enabled:
            return ["I'm currently disabled."]
        try:
            collected = []
            async for piece in self.get_response_streaming(messages, memories):
                collected.append(piece)
            full = " ".join(collected).strip()
            if not full:
                return []
            sentences = re.split(r'(?<=[.!?])\s+', full)
            tts_chunks = []
            cur = ""
            for s in sentences:
                if not s:
                    continue
                if cur and len(cur) + 1 + len(s) > chunk_size:
                    tts_chunks.append(cur)
                    cur = s
                else: 
                    cur = s if not cur else f"{cur} {s}"
            if cur:
                tts_chunks.append(cur)
            return tts_chunks
        except Exception:
            return ["Someone tell Eric there is a problem with my AI!"]

    class API:
        def __init__(self, outer):
            self.outer = outer
        def set_LLM_status(self, status: bool):
            self.outer.llm_state.enabled = status
        def get_LLM_status(self) -> bool:
            return self.outer.llm_state.enabled
        def cancel_next(self):
            self.outer.llm_state.next_cancelled = True

        #HOPEFULLY THIS WORKS NOW :((((( - So Sleepy