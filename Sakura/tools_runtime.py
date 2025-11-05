import json
from typing import Dict, Any, Callable, Optional

def run_tool(
    decision: Dict[str,Any], 
    search_fn: Optional[Callable] = None,
    call_fn: Optional[Callable] = None, 
    dm_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    action = decision.get("action")
    
    if action == "reply":
        return {"tool_used": None, "result": None}
    
    try:
        if action == "search":
            if not search_fn:
                return {"tool_used": None, "result": {"error":"search not available"}}
            
            query = (decision.get("args") or {}).get("query","").strip()
            if not query:
                return {"tool_used": None, "result":{"error":"empty_query"}}
            
            print(f"[TOOLS] Executing search: '{query}'")
            res = search_fn(query)
            return {"tool_used":"web_search", "result": res}
        
        elif action == "call":
            if not call_fn:
                return {"tool_used": None, "result": {"error":"call not available"}}
            
            target = (decision.get("args") or {}).get("target") or "Eric"
            print(f"[TOOLS] Executing call: target='{target}'")
            res = call_fn(target)
            return {"tool_used":"call", "result": res}
        
        elif action == "dm":
            if not dm_fn:
                return {"tool_used": None, "result": {"error":"dm not available"}}
            
            args = decision.get("args") or {}
            target = args.get("target") or "Eric"
            intent = args.get("intent") or "casual message"
            print(f"[TOOLS] Executing DM: target='{target}', intent='{intent}'")
            res = dm_fn(target, intent)
            return {"tool_used":"dm", "result": res}
        
        else:
            return {"tool_used": None, "result": {"error":f"unknown action: {action}"}}
    
    except Exception as e:
        print(f"[TOOLS] Error executing {action}: {e}")
        return {"tool_used": None, "result": {"error": str(e)}}

def format_tool_result_for_synthesis(tool_name: str, result: Any) -> str:
    if result is None:
        return ""
    
    if isinstance(result, dict) and "error" in result:
        return f"Tool '{tool_name}' failed: {result['error']}"
    
    if tool_name == "web_search":
        if isinstance(result, dict):
            items = result.get("items", [])
            if items:
                summaries = []
                for i, item in enumerate(items[:3], 1):
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    summaries.append(f"{i}. {title}: {snippet}")
                return "Search results:\n" + "\n".join(summaries)
            return "Search returned no results."
        return str(result)
    
    elif tool_name == "call":
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            target = result.get("target", "user")
            return f"Call to {target}: {status}"
        return str(result)
    
    elif tool_name == "dm":
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            target = result.get("target", "user")
            return f"DM to {target}: {status}"
        return str(result)
    
    return json.dumps(result, ensure_ascii=False)
