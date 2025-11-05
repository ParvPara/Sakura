import json
from typing import Dict, List, Any, Optional

DECIDER_SYS = """You are Sakura's executive function.
Choose exactly one action for the NEXT step: ["reply","search","call","dm"].
Use tools when:
- a proper noun or topic appears with unclear context that you genuinely don't know about, or
- affect.mood == "lonely" and a safe contact exists (for 'call' or 'dm'), or
- the user is asking about something you're uncertain about.

DEFAULT to "reply" for normal conversation. Only use tools when truly needed.
When action="search", compress query to: who/what + distinctive noun phrase.
When action="call", provide target user.
When action="dm", provide target user and brief message intent.
Return ONLY JSON: {"action": "...", "reason": "...", "args": {...}}."""

DECIDER_FEWSHOTS = [
    {"role":"system","content":
     'Example: User: "Who is John Carmack?"; Signals: novelty=0.8\n'
     'Decision: {"action":"search","reason":"unfamiliar person","args":{"query":"John Carmack"}}'},
    {"role":"system","content":
     'Example: State: mood=lonely; User: "what now?"\n'
     'Decision: {"action":"dm","reason":"lonely; social reach-out","args":{"target":"Eric","intent":"casual chat"}}'},
    {"role":"system","content":
     'Example: User: "what do you think about AI?"\n'
     'Decision: {"action":"reply","reason":"opinion question, no tool needed","args":{}}'},
    {"role":"system","content":
     'Example: User: "just chatting with you"\n'
     'Decision: {"action":"reply","reason":"conversational, no tool needed","args":{}}'},
    {"role":"system","content":
     'Example: User: "tell me about the latest updates"\n'
     'Decision: {"action":"reply","reason":"vague request, can chat about it","args":{}}'},
]

def decide(
    llm_client, 
    model: str, 
    history: List[Dict[str,Any]], 
    signals: Dict[str,Any], 
    tools: List[str],
    user_name: str = "User"
) -> Dict[str,Any]:
    msgs = [{"role":"system","content": DECIDER_SYS}]
    msgs += DECIDER_FEWSHOTS
    msgs.append({"role":"system","content": f"SIGNALS={json.dumps(signals, ensure_ascii=False)}"})
    msgs.append({"role":"system","content": f"AVAILABLE_TOOLS={json.dumps(tools, ensure_ascii=False)}"})
    msgs.append({"role":"system","content": f"CURRENT_USER={user_name}"})
    msgs += history[-4:]
    
    try:
        r = llm_client.chat(
            model=model, 
            messages=msgs, 
            options={"temperature":0.2, "num_predict":200}, 
            stream=False
        )
        content = r["message"].get("content","{}").strip()
        
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        start_idx = content.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(content)):
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                content = content[start_idx:end_idx]
        
        decision = json.loads(content)
    except Exception as e:
        print(f"[DECIDER] Failed to parse decision: {e}")
        print(f"[DECIDER] Raw content: {content[:200]}")
        decision = {"action":"reply","reason":"fallback due to parse error","args":{}}
    
    valid_actions = {"reply","search","call","dm"}
    if decision.get("action") not in valid_actions:
        print(f"[DECIDER] Invalid action '{decision.get('action')}', defaulting to reply")
        decision["action"] = "reply"
    
    print(f"[DECIDER] Decision: action={decision.get('action')}, reason={decision.get('reason','')}")
    return decision

def compress_query(raw_query: str, max_words: int = 8) -> str:
    words = raw_query.split()
    if len(words) <= max_words:
        return raw_query
    
    important_words = []
    for word in words:
        if word[0].isupper() or len(word) > 4 or word in {"experimental","beta","latest","new","version","update"}:
            important_words.append(word)
    
    if important_words:
        return " ".join(important_words[:max_words])
    return " ".join(words[:max_words])
