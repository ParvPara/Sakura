#!/usr/bin/env python3
"""
Enhanced Search Tool for Sakura's agentic system
Integrates with tool guard for permissions, rate limits, and safety
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from schemas import ToolResponse, ToolStatus, ToolContext
from router.tool_guard import ToolGuard

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    description: str
    source: str

@dataclass
class SearchResponse:
    """Complete search response"""
    success: bool
    results: List[SearchResult]
    query: str
    total_results: int
    error: Optional[str] = None

class SearchTool:
    """Enhanced search tool with guard integration"""
    
    def __init__(self, websearch, tool_guard: ToolGuard):
        self.websearch = websearch
        self.tool_guard = tool_guard
        logger.info("Search Tool initialized with guard integration")
    
    async def execute_search(self, context: ToolContext, query: str) -> ToolResponse:
        """Execute search with full guard checks and execution"""
        try:
            # Check if search is allowed through tool guard
            allowed, status, meta = self.tool_guard.check_search_allowed(context, query)
            
            if not allowed:
                # Log the denial
                self.tool_guard.log_tool_attempt(context, "SEARCH", False, status, meta)
                
                return ToolResponse(
                    ok=False,
                    tool="SEARCH",
                    status=ToolStatus(status),
                    reason=self._get_denial_reason(status, meta),
                    meta=meta
                )
            
            # Search is allowed - attempt to execute
            search_response = await self._perform_search(query)
            
            if search_response.success:
                # Consume rate limit token
                self.tool_guard._consume_token(context.user_id, "search")
                
                # Log successful execution
                self.tool_guard.log_tool_attempt(context, "SEARCH", True, "executed", {
                    "query": query,
                    "results_count": search_response.total_results,
                    "user_id": context.user_id
                })
                
                # Format results for response
                formatted_results = []
                for result in search_response.results:
                    formatted_results.append({
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.description[:200] + "..." if len(result.description) > 200 else result.description,
                        "source": result.source
                    })
                
                return ToolResponse(
                    ok=True,
                    tool="SEARCH",
                    status=ToolStatus.EXECUTED,
                    reason=f"Search completed successfully - found {search_response.total_results} results",
                    meta={
                        "query": query,
                        "results": formatted_results,
                        "total_results": search_response.total_results,
                        "user_id": context.user_id
                    }
                )
            else:
                # Search failed
                self.tool_guard.log_tool_attempt(context, "SEARCH", False, "error", {
                    "error": search_response.error,
                    "query": query,
                    "user_id": context.user_id
                })
                
                return ToolResponse(
                    ok=False,
                    tool="SEARCH",
                    status=ToolStatus.ERROR,
                    reason=f"Search failed: {search_response.error}",
                    meta={
                        "error": search_response.error,
                        "query": query,
                        "user_id": context.user_id
                    }
                )
                
        except Exception as e:
            logger.error(f"Search tool execution error: {e}")
            
            return ToolResponse(
                ok=False,
                tool="SEARCH",
                status=ToolStatus.ERROR,
                reason=f"Search tool error: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def _perform_search(self, query: str) -> SearchResponse:
        """Actually perform the web search"""
        try:
            # Use the existing websearch instance
            search_response = self.websearch.search(query, count=5)
            
            if not search_response or not search_response.success:
                return SearchResponse(
                    success=False,
                    results=[],
                    query=query,
                    total_results=0,
                    error="Web search failed or returned no results"
                )
            
            # Convert search results to our format
            results = []
            for result in search_response.results[:5]:  # Limit to 5 results
                search_result = SearchResult(
                    title=result.title,
                    url=result.url,
                    description=result.description,
                    source="web_search"
                )
                results.append(search_result)
            
            return SearchResponse(
                success=True,
                results=results,
                query=query,
                total_results=len(results)
            )
            
        except Exception as e:
            logger.error(f"Error performing search for '{query}': {e}")
            return SearchResponse(
                success=False,
                results=[],
                query=query,
                total_results=0,
                error=f"Search error: {str(e)}"
            )
    
    def _get_denial_reason(self, status: str, meta: Dict[str, Any]) -> str:
        """Get human-readable reason for search denial"""
        if status == "revoked":
            return "Search tool is currently disabled"
        elif status == "forbidden":
            if meta.get("reason") == "policy":
                return f"Search query violates ToS: {meta.get('violation', 'unknown')}"
            else:
                return "Search not allowed for this query"
        elif status == "rate_limited":
            cooldown = meta.get("cooldown_secs", 0)
            if meta.get("global"):
                return f"Global search rate limit exceeded. Try again in {cooldown} seconds"
            else:
                return f"User search rate limit exceeded. Try again in {cooldown} seconds"
        else:
            return f"Search denied: {status}"
    
    async def get_search_status(self, context: ToolContext) -> Dict[str, Any]:
        """Get search status and rate limit information for a user"""
        try:
            # Get rate limit info
            rate_limit_info = self.tool_guard.get_rate_limit_info(context.user_id, "search")
            
            # Check if search tool is enabled
            search_enabled = self.tool_guard._is_tool_enabled("search")
            
            return {
                "user_id": context.user_id,
                "user_name": context.user_name,
                "search_enabled": search_enabled,
                "rate_limits": rate_limit_info,
                "can_search": search_enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting search status: {e}")
            return {
                "user_id": context.user_id,
                "user_name": context.user_name,
                "error": str(e)
            }
    
    def format_search_results_for_llm(self, search_response: ToolResponse) -> str:
        """Format search results for LLM consumption"""
        if not search_response.ok or search_response.status != ToolStatus.EXECUTED:
            return f"Search failed: {search_response.reason}"
        
        results = search_response.meta.get("results", [])
        query = search_response.meta.get("query", "unknown")
        
        if not results:
            return f"No search results found for '{query}'"
        
        formatted = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result['title']}**\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   Source: {result['url']}\n\n"
        
        formatted += f"Found {len(results)} results. You can cite these sources in your response."
        
        return formatted
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on query"""
        suggestions = []
        
        # Basic query enhancement
        if "what is" in query.lower():
            suggestions.append(query + " definition")
            suggestions.append(query + " meaning")
        elif "how to" in query.lower():
            suggestions.append(query + " tutorial")
            suggestions.append(query + " guide")
        elif "latest" in query.lower() or "news" in query.lower():
            suggestions.append(query + " 2024")
            suggestions.append(query + " current")
        
        # Add the original query
        suggestions.insert(0, query)
        
        return suggestions[:5]  # Limit to 5 suggestions
