import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Sakura.decider import decide
from Sakura.signals import AffectStore, compute_signals
from Sakura.tools_runtime import run_tool, format_tool_result_for_synthesis
import ollama

def mock_search(query: str) -> dict:
    print(f"  [MOCK SEARCH] Query: '{query}'")
    return {
        "items": [
            {"title": f"Latest on {query}", "url": "https://example.com", "snippet": f"Recent updates about {query} include..."},
            {"title": f"{query} details", "url": "https://example.com/2", "snippet": "More information..."}
        ]
    }

def mock_call(target: str) -> dict:
    print(f"  [MOCK CALL] Calling {target}...")
    return {"target": target, "status": "initiated"}

def mock_dm(target: str, intent: str) -> dict:
    print(f"  [MOCK DM] Sending DM to {target} with intent: {intent}")
    return {"target": target, "status": "sent", "intent": intent}

def run_demo_conversation(utterances: list, affect: AffectStore):
    try:
        client = ollama.Client()
        model = "sakura"
    except Exception as e:
        print(f"Warning: Could not connect to Ollama: {e}")
        print("Demo will use mock responses\n")
        client = None
        model = None
    
    memory_terms = ["Python", "AI", "Neuro", "Vedal", "Eric"]
    
    for i, (utterance, events) in enumerate(utterances, 1):
        print(f"\n{'='*70}")
        print(f"TURN {i}")
        print(f"{'='*70}")
        print(f"User: {utterance}")
        print()
        
        signals = compute_signals(utterance, memory_terms, affect, events)
        print(f"Signals: uncertainty={signals['uncertainty']}, novelty={signals['novelty']}, "
              f"is_question={signals['is_question']}, mood={signals['affect']['mood']}")
        
        history = [{"role": "user", "content": utterance}]
        tools = ["search", "call", "dm"]
        
        if client:
            decision = decide(client, model, history, signals, tools, user_name="Demo User")
        else:
            if signals['novelty'] > 0.7 or "who is" in utterance.lower():
                decision = {"action": "search", "reason": "unfamiliar topic", "args": {"query": "search query"}}
            elif signals['affect']['mood'] == "lonely":
                decision = {"action": "dm", "reason": "lonely; social reach-out", "args": {"target": "Eric", "intent": "casual chat"}}
            else:
                decision = {"action": "reply", "reason": "conversational", "args": {}}
        
        print(f"Decision: {decision['action']} - {decision.get('reason', '')}")
        
        tool_feedback = run_tool(
            decision, 
            search_fn=mock_search,
            call_fn=mock_call,
            dm_fn=mock_dm
        )
        
        if tool_feedback["tool_used"]:
            formatted_result = format_tool_result_for_synthesis(
                tool_feedback["tool_used"], 
                tool_feedback["result"]
            )
            print(f"\nTool Result:\n{formatted_result}")
            print("\n[Sakura would now synthesize this into a natural response]")
        else:
            print("\n[No tool used - Sakura replies normally]")

def main():
    print("="*70)
    print("AUTONOMOUS DECISION LOOP DEMO")
    print("="*70)
    
    affect = AffectStore()
    
    scenario_1 = [
        ("Who is John Carmack?", {"conversation": True}),
        ("That's interesting, what else do you know?", {"conversation": True}),
    ]
    
    print("\n\nSCENARIO 1: Unfamiliar Topic Query")
    print("-" * 70)
    run_demo_conversation(scenario_1, affect)
    
    print("\n\nSCENARIO 2: Loneliness-Driven Social Reach-out")
    print("-" * 70)
    
    affect2 = AffectStore()
    
    for _ in range(8):
        affect2.tick({"chat_quiet": True})
    
    scenario_2 = [
        ("I'm bored...", {"chat_quiet": True}),
    ]
    
    run_demo_conversation(scenario_2, affect2)
    
    print("\n\nSCENARIO 3: Normal Conversation (No Tools)")
    print("-" * 70)
    
    affect3 = AffectStore()
    
    scenario_3 = [
        ("Hey Sakura, how are you doing?", {"positive": True, "conversation": True}),
        ("That's good to hear!", {"positive": True, "conversation": True}),
    ]
    
    run_demo_conversation(scenario_3, affect3)
    
    print("\n\nSCENARIO 4: Multiple Tool Calls")
    print("-" * 70)
    
    affect4 = AffectStore()
    
    scenario_4 = [
        ("What's the latest on GPT-5?", {"conversation": True}),
        ("What about Claude 4?", {"conversation": True}),
        ("And what about Gemini 2?", {"conversation": True}),
    ]
    
    run_demo_conversation(scenario_4, affect4)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
