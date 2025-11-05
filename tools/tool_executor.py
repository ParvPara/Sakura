#!/usr/bin/env python3
"""
Tool Executor for Sakura's agentic system
Coordinates tool execution and manages the autonomy loop
"""

import logging
from typing import Dict, Any, Optional, List
import json

from schemas import ToolCall, ToolResponse, ToolContext, ToolStatus
from router.tool_guard import ToolGuard
from tools.dm_tool import DMTool
from tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Executes tools and manages the autonomy loop"""
    
    def __init__(self, bot, tool_guard: ToolGuard, websearch):
        self.bot = bot
        self.tool_guard = tool_guard
        
        # Initialize tools
        from tools.call_tool import CallTool
        
        self.dm_tool = DMTool(bot, tool_guard)
        self.search_tool = SearchTool(websearch, tool_guard)
        self.call_tool = CallTool(bot, tool_guard)
        
        # Tool registry
        self.tools = {
            "DM": self.dm_tool.execute_dm,
            "SEARCH": self.search_tool.execute_search,
            "CALL": self.call_tool.execute_invite
        }
        
        logger.info("Tool Executor initialized")
    
    async def execute_tool_call(self, tool_call: ToolCall, context: ToolContext) -> ToolResponse:
        """Execute a tool call with full validation and execution"""
        try:
            # Validate tool call
            if not tool_call:
                return ToolResponse(
                    ok=False,
                    tool="UNKNOWN",
                    status=ToolStatus.INVALID_ARGS,
                    reason="No tool call provided",
                    meta={"error": "missing_tool_call"}
                )
            
            # Check if tool exists
            if tool_call.name not in self.tools:
                return ToolResponse(
                    ok=False,
                    tool=tool_call.name,
                    status=ToolStatus.INVALID_ARGS,
                    reason=f"Unknown tool: {tool_call.name}",
                    meta={"available_tools": list(self.tools.keys())}
                )
            
            # Execute the tool
            logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.args}")
            
            if tool_call.name == "DM":
                message = tool_call.args.get("message", "")
                target_person = tool_call.args.get("target_person")
                return await self.dm_tool.execute_dm(context, message, target_person)
            
            elif tool_call.name == "CALL":
                message = tool_call.args.get("message", "")
                target_user = tool_call.args.get("target_user")
                return await self.call_tool.execute_invite(context, message, target_user)
            
            elif tool_call.name == "SEARCH":
                query = tool_call.args.get("query", "")
                return await self.search_tool.execute_search(context, query)
            
            else:
                return ToolResponse(
                    ok=False,
                    tool=tool_call.name,
                    status=ToolStatus.ERROR,
                    reason=f"Tool execution not implemented: {tool_call.name}",
                    meta={"error": "not_implemented"}
                )
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            
            return ToolResponse(
                ok=False,
                tool=tool_call.name if tool_call else "UNKNOWN",
                status=ToolStatus.ERROR,
                reason=f"Tool execution failed: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def execute_autonomous_tool_call(self, context: ToolContext, tool_name: str, **kwargs) -> ToolResponse:
        """Execute a tool call autonomously (without LLM input)"""
        try:
            # Create a tool call object
            tool_call = ToolCall(
                name=tool_name,
                args=kwargs,
                confidence=1.0,  # High confidence for autonomous calls
                reasoning="Autonomous tool execution"
            )
            
            # Execute the tool
            return await self.execute_tool_call(tool_call, context)
            
        except Exception as e:
            logger.error(f"Autonomous tool execution error: {e}")
            
            return ToolResponse(
                ok=False,
                tool=tool_name,
                status=ToolStatus.ERROR,
                reason=f"Autonomous execution failed: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def get_tool_status(self, context: ToolContext) -> Dict[str, Any]:
        """Get status of all tools for a user"""
        try:
            dm_status = await self.dm_tool.get_dm_status(context)
            search_status = await self.search_tool.get_search_status(context)
            
            return {
                "user_id": context.user_id,
                "user_name": context.user_name,
                "tools": {
                    "dm": dm_status,
                    "search": search_status
                },
                "whitelist": self.tool_guard.get_whitelist()
            }
            
        except Exception as e:
            logger.error(f"Error getting tool status: {e}")
            return {
                "user_id": context.user_id,
                "user_name": context.user_name,
                "error": str(e)
            }
    
    async def add_user_to_whitelist(self, user_id: str) -> bool:
        """Add a user to the DM whitelist"""
        return await self.dm_tool.add_user_to_whitelist(user_id)
    
    async def remove_user_from_whitelist(self, user_id: str) -> bool:
        """Remove a user from the DM whitelist"""
        return await self.dm_tool.remove_user_from_whitelist(user_id)
    
    def get_whitelist(self) -> List[str]:
        """Get current DM whitelist"""
        return self.dm_tool.get_whitelist()
    
    def format_tool_response_for_llm(self, tool_response: ToolResponse) -> str:
        """Format tool response for LLM consumption"""
        if not tool_response.ok:
            return f"Tool execution failed: {tool_response.reason}"
        
        if tool_response.tool == "SEARCH":
            return self.search_tool.format_search_results_for_llm(tool_response)
        elif tool_response.tool == "DM":
            return f"DM sent successfully: {tool_response.reason}"
        else:
            return f"Tool {tool_response.tool} executed successfully: {tool_response.reason}"
    
    def get_tool_help(self) -> str:
        """Get help information for all available tools"""
        help_text = "**Available Tools:**\n\n"
        
        help_text += "**DM Tool**\n"
        help_text += "- Send private Discord messages\n"
        help_text += "- Requires user to be whitelisted\n"
        help_text += "- Rate limited: 1 DM per 90 seconds per user\n\n"
        
        help_text += "**SEARCH Tool**\n"
        help_text += "- Perform web searches for information\n"
        help_text += "- Rate limited: 3 searches per minute per user\n"
        help_text += "- Returns 5 results with sources\n\n"
        
        help_text += "**Usage Examples:**\n"
        help_text += "- Ask for definitions, news, prices, facts\n"
        help_text += "- Request private conversations\n"
        help_text += "- Search for current information\n\n"
        
        help_text += "**Rate Limits:**\n"
        help_text += "- Global: 6 DMs per hour, 60 searches per hour\n"
        help_text += "- Per-user: 1 DM per 90s, 3 searches per minute"
        
        return help_text
    
    def log_tool_execution(self, tool_call: ToolCall, tool_response: ToolResponse, context: ToolContext):
        """Log tool execution for audit trail"""
        logger.info(
            f"Tool Execution: {tool_call.name} -> {tool_response.status} "
            f"for user {context.user_name} ({context.user_id}) "
            f"Reason: {tool_response.reason}"
        )
        
        # Log additional details for debugging
        if tool_response.meta:
            logger.debug(f"Tool execution meta: {tool_response.meta}")
