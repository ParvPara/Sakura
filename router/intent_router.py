#!/usr/bin/env python3
"""
Enhanced Intent Router for Sakura's agentic system
Combines fast Neuro-style classification with tool calling protocol
"""

import time
import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from schemas import RouterDecision, ToolCall, ToolContext

logger = logging.getLogger(__name__)

@dataclass
class IntentScore:
    """Score for a specific intent"""
    score: int
    patterns_matched: List[str]
    confidence: float

class IntentRouter:
    """Fast, reliable intent classification with tool calling protocol"""
    
    def __init__(self):
        # DM intent patterns (high priority)
        self.dm_patterns = [
            'send me', 'message me', 'contact me', 'dm me', 'reach me',
            'get in touch', 'write me', 'text me', 'ping me', 'notify me',
            'discord message', 'private message', 'direct message',
            'talk privately', 'in private', 'personal message',
            'tell me privately', 'don\'t say it here', 'keep it private',
            'slide into my dms', 'hit me up', 'reach out to me'
        ]
        
        # Search intent patterns (more specific to avoid false positives)
        self.search_patterns = [
            'weather today', 'latest news', 'search for', 'find information', 'look up',
            'what is the price', 'stock price', 'movie reviews', 'current events',
            'define this', 'meaning of', 'find out about', 'where to buy',
            'google search', 'web search', 'search results', 'information about'
        ]
        
        # Chat intent patterns (enhanced to catch greetings)
        self.chat_patterns = [
            'hello', 'hi', 'hey', 'how are you', 'how you doing', 'good morning', 'good evening',
            'tell me a joke', 'what time', 'thanks', 'thank you', 'sup', 'what\'s up',
            'goodbye', 'bye', 'see you', 'how are you doing', 'how\'s it going',
            'nice to meet you', 'pleasure', 'appreciate it', 'hey there', 'yo',
            'good to see you', 'how have you been', 'what\'s new', 'how\'s your day'
        ]
        
        logger.info("Intent Router initialized with Neuro-style patterns")
    
    def route_message(self, text: str, context: ToolContext) -> RouterDecision:
        """Route a message with fast Neuro-style classification and tool calling"""
        start_time = time.time()
        
        # Fast intent detection with optimized scoring
        dm_score = self._calculate_dm_score(text.lower())
        call_score = self._calculate_call_score(text.lower())
        search_score = self._calculate_search_score(text.lower())
        chat_score = self._calculate_chat_score(text.lower())
        
        # Check for person mentions and voice presence
        person_mention_result = self._check_person_mentions(text, context)
        
        # Fast decision logic with clear thresholds and person mention priority
        if person_mention_result['should_dm']:
            action = "DM"
            confidence = 95  # High confidence when mentioning someone not in call
            reasoning = f"Person mention DM: {person_mention_result['reason']}"
            
            # Generate enhanced tool call for person-specific DM
            tool_call = self._generate_person_dm_tool_call(context, text, person_mention_result, confidence)
            
        elif call_score.score > 0 and call_score.score >= max(dm_score.score, search_score.score):
            action = "CALL"
            confidence = min(85 + (call_score.score * 3), 95)
            reasoning = f"Voice call intent detected (score: {call_score.score}, patterns: {call_score.patterns_matched})"
            
            # Generate tool call for voice call
            tool_call = self._generate_call_tool_call(context, text, confidence)
            
        elif dm_score.score > 0:
            action = "DM"
            confidence = min(90 + (dm_score.score * 2), 98)
            reasoning = f"DM intent detected (score: {dm_score.score}, patterns: {dm_score.patterns_matched})"
            
            # Generate tool call for DM
            tool_call = self._generate_dm_tool_call(context, text, confidence)
            
        elif search_score.score > 0 and search_score.score > chat_score.score:
            action = "SEARCH"
            confidence = min(60 + (search_score.score * 10), 95)  # Lower base, higher multiplier for genuine searches
            reasoning = f"Search intent detected (score: {search_score.score}, patterns: {search_score.patterns_matched})"
            
            # Generate tool call for search
            tool_call = self._generate_search_tool_call(context, text, confidence)
            
        else:
            action = "CHAT"
            confidence = 95
            reasoning = "Normal conversation intent - no tools needed"
            tool_call = None
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Intent Router: {action} (confidence: {confidence}%, latency: {latency_ms}ms)")
        
        return RouterDecision(
            action=action,
            confidence=confidence / 100.0,  # Convert to 0.0-1.0 range
            reasoning=reasoning,
            latency_ms=latency_ms,
            method="neuro_fast",
            tool_call=tool_call
        )
    
    def _check_person_mentions(self, text: str, context: ToolContext) -> Dict[str, Any]:
        """Check if message mentions someone who should be DMed"""
        import re
        
        # Only applicable for voice channels
        if context.channel_type != "voice" or not context.voice_members:
            return {'should_dm': False, 'reason': 'Not in voice context'}
        
        voice_member_names = [m['name'].lower() for m in context.voice_members]
        voice_member_display_names = [m['display_name'].lower() for m in context.voice_members]
        
        # Patterns that suggest asking someone to do something
        action_patterns = [
            r'ask (\w+) to',
            r'tell (\w+) to',
            r'get (\w+) to',
            r'have (\w+) do',
            r'remind (\w+) to',
            r'let (\w+) know',
            r'message (\w+) about',
            r'dm (\w+) about',
            r'send (\w+) a message',
            r'contact (\w+) about'
        ]
        
        text_lower = text.lower()
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                mentioned_name = match.strip()
                
                # Skip if mentioned person is already in voice
                if mentioned_name in voice_member_names or mentioned_name in voice_member_display_names:
                    continue
                
                # Check if it's a known person from people.json
                if self._is_known_person(mentioned_name):
                    return {
                        'should_dm': True,
                        'reason': f"User wants to contact {mentioned_name} who is not in voice call",
                        'target_person': mentioned_name,
                        'action_requested': text,
                        'voice_members': context.voice_members
                    }
        
        return {'should_dm': False, 'reason': 'No person mentions requiring DM'}
    
    def _is_known_person(self, name: str) -> bool:
        """Check if the mentioned name is a known person"""
        import json
        import os
        
        try:
            people_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'people.json')
            if os.path.exists(people_file):
                with open(people_file, 'r') as f:
                    people_data = json.load(f)
                
                # Check if name matches any known person
                for person in people_data.get('people', []):
                    if name.lower() in [person.get('name', '').lower(), person.get('nickname', '').lower()]:
                        return True
            
            # Also consider common names that might be Discord users
            common_names = ['eric', 'alex', 'sarah', 'mike', 'john', 'jane', 'bob', 'alice', 'tom', 'lisa']
            return name.lower() in common_names
            
        except Exception as e:
            logger.warning(f"Could not check people.json: {e}")
            return True  # Assume it's a person if we can't check
    
    def _generate_person_dm_tool_call(self, context: ToolContext, text: str, person_result: Dict[str, Any], confidence: float) -> ToolCall:
        """Generate a DM tool call for contacting a specific person"""
        target_person = person_result.get('target_person', 'someone')
        
        # Create a message explaining the situation
        message = f"Hi! {context.user_name} asked me to contact you about: {text}. They're currently in a voice call but wanted to make sure you got this message."
        
        return ToolCall(
            name="DM",
            args={
                "target_person": target_person,
                "message": message,
                "reason": f"Contacting {target_person} as requested by {context.user_name}",
                "context": person_result
            },
            confidence=confidence / 100.0,
            reasoning=f"User requested to contact {target_person} who is not in the current voice call"
        )
    
    def _calculate_dm_score(self, text_lower: str) -> IntentScore:
        """Calculate DM intent score with pattern matching"""
        patterns_matched = []
        score = 0
        
        for pattern in self.dm_patterns:
            if pattern in text_lower:
                patterns_matched.append(pattern)
                score += 3  # Base score for each pattern
        
        # Bonus for obvious contact requests
        contact_words = ['send', 'message', 'contact', 'reach', 'dm', 'private']
        for word in contact_words:
            if word in text_lower:
                score += 2
        
        # Bonus for explicit privacy requests
        privacy_words = ['privately', 'private', 'don\'t say', 'keep it']
        for word in privacy_words:
            if word in text_lower:
                score += 3
        
        return IntentScore(
            score=score,
            patterns_matched=patterns_matched,
            confidence=min(score * 5, 95)  # Convert score to confidence
        )
    
    def _calculate_search_score(self, text_lower: str) -> IntentScore:
        """Calculate search intent score with pattern matching"""
        patterns_matched = []
        score = 0
        
        for pattern in self.search_patterns:
            if pattern in text_lower:
                patterns_matched.append(pattern)
                score += 2
        
        # Bonus for question words only if they're part of search-like patterns
        search_question_patterns = [
            'what is the', 'how do i', 'when did', 'where can i', 'why does', 'who is the'
        ]
        for pattern in search_question_patterns:
            if pattern in text_lower:
                score += 1
        
        return IntentScore(
            score=score,
            patterns_matched=patterns_matched,
            confidence=min(score * 4, 90)
        )
    
    def _calculate_chat_score(self, text_lower: str) -> IntentScore:
        """Calculate chat intent score with pattern matching"""
        patterns_matched = []
        score = 0
        
        for pattern in self.chat_patterns:
            if pattern in text_lower:
                patterns_matched.append(pattern)
                score += 1
        
        return IntentScore(
            score=score,
            patterns_matched=patterns_matched,
            confidence=min(score * 3, 85)
        )
    
    def _calculate_call_score(self, text_lower: str) -> IntentScore:
        """Calculate voice channel call intent score"""
        patterns_matched = []
        score = 0
        
        # High priority call patterns (these override DM patterns)
        high_priority_patterns = [
            r'\b(call\s+me)\b',
            r'\b(give\s+me\s+a\s+call)\b',
            r'\b(can\s+you\s+call)\b',
            r'\b(send\s+me\s+(a\s+)?(call|voice)\s+invite)\b',
            r'\b(call\s+invite)\b',
            r'\b(voice\s+invite)\b'
        ]
        
        # Check high priority patterns first (these get higher scores)
        for pattern in high_priority_patterns:
            if re.search(pattern, text_lower):
                patterns_matched.append(pattern)
                score += 5  # Higher score to override DM patterns
        
        # Regular call patterns
        call_patterns = [
            r'\b(join\s+(us|me|the\s+call|voice|vc))\b',
            r'\b(come\s+(to\s+)?voice)\b', 
            r'\b(hop\s+in(\s+the\s+call)?)\b',
            r'\b(call\s+.+\s+to\s+(voice|call|vc))\b',
            r'\b(ask\s+.+\s+to\s+join)\b',
            r'\b(get\s+.+\s+(in\s+)?(voice|call|vc))\b',
            r'\b(bring\s+.+\s+(to\s+)?voice)\b'
        ]
        
        for pattern in call_patterns:
            if re.search(pattern, text_lower):
                patterns_matched.append(pattern)
                score += 2
        
        # Context patterns that suggest voice call
        context_patterns = [
            r'\b(everyone\s+should\s+join)\b',
            r'\b(come\s+here)\b',
            r'\b(join\s+the\s+conversation)\b',
            r'\b(we\'re\s+talking\s+about)\b'
        ]
        
        for pattern in context_patterns:
            if re.search(pattern, text_lower):
                patterns_matched.append(pattern)
                score += 1
        
        return IntentScore(
            score=score,
            patterns_matched=patterns_matched,
            confidence=min(score * 4, 90)
        )
    
    def _generate_dm_tool_call(self, context: ToolContext, text: str, confidence: float) -> ToolCall:
        """Generate a DM tool call - content will be generated by LLM"""
        
        return ToolCall(
            name="DM",
            args={
                "user_id": context.user_id,
                "message": "LLM_GENERATED",  # Placeholder - will be replaced by LLM content
                "reason": "User requested private conversation"
            },
            confidence=confidence / 100.0,
            reasoning=f"User requested private conversation: {text[:100]}..."
        )
    
    def _generate_call_tool_call(self, context: ToolContext, text: str, confidence: float) -> ToolCall:
        """Generate a CALL tool call with fallback call message"""
        
        # Try to extract target user from text
        target_user = self._extract_target_user_from_text(text)
        
        # Generate a fallback call message
        if target_user:
            fallback_message = f"Hey {target_user}! {context.user_name} wants you to join us in the voice channel. We're having a great conversation - come hop in!"
        else:
            fallback_message = f"Hey! {context.user_name} wants you to join us in the voice channel. We're having a great time - come join the conversation!"
        
        return ToolCall(
            name="CALL",
            args={
                "target_user": target_user,
                "message": fallback_message,
                "reason": "User wants to invite someone to voice channel"
            },
            confidence=confidence / 100.0,
            reasoning=f"User wants to invite someone to voice: {text[:100]}..."
        )
    
    def _generate_search_tool_call(self, context: ToolContext, text: str, confidence: float) -> ToolCall:
        """Generate a search tool call based on context and message"""
        # Extract search query from text
        search_query = self._extract_search_query(text)
        
        return ToolCall(
            name="SEARCH",
            args={
                "query": search_query,
                "user_id": context.user_id,
                "context": text[:200]  # Include some context
            },
            confidence=confidence / 100.0,
            reasoning=f"User requested information search: {text[:100]}..."
        )
    

    
    def _extract_search_query(self, text: str) -> str:
        """Extract search query from user message"""
        # Remove common prefixes
        query = text.strip()
        
        # Remove question words at the beginning
        question_prefixes = ['what is', 'what\'s', 'how to', 'when is', 'where is', 'who is', 'why is']
        for prefix in question_prefixes:
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break
        
        # Remove "search for" prefix
        if query.lower().startswith('search for'):
            query = query[11:].strip()
        
        # Clean up the query
        query = re.sub(r'[?.,!]', '', query).strip()
        
        # If query is too short, use original text
        if len(query) < 3:
            query = text.strip()
        
        return query
    
    def _extract_target_user_from_text(self, text: str) -> Optional[str]:
        """Extract target user name from invite text"""
        import re
        
        # Patterns to extract user names from invite requests
        patterns = [
            r'(?:invite|ask|get|call|bring)\s+(\w+)\s+(?:to\s+)?(?:join|voice|call|vc|the\s+call)',
            r'(?:invite|ask|get|call|bring)\s+(\w+)',  # More general pattern
            r'(\w+)\s+(?:should\s+)?(?:join|come)',
            r'(?:tell|ask)\s+(\w+)\s+to\s+(?:join|come|hop\s+in)',
            r'(\w+)\s+(?:wants\s+to\s+)?(?:join|come)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                user_name = match.group(1)
                # Skip common words that aren't user names
                if user_name not in ['us', 'me', 'here', 'this', 'that', 'the', 'voice', 'call', 'everyone', 'to', 'in', 'on', 'at', 'with']:
                    return user_name
        
        return None
    
    def parse_tool_call_from_llm(self, response_text: str) -> Optional[ToolCall]:
        """Parse tool call from LLM response text"""
        try:
            # Look for JSON tool call in the response - handle nested JSON properly
            # First try to find the complete JSON block
            json_start = response_text.find('```json')
            if json_start != -1:
                json_start = response_text.find('{', json_start)
                if json_start != -1:
                    # Find the matching closing brace by counting braces
                    brace_count = 0
                    json_end = json_start
                    for i, char in enumerate(response_text[json_start:], json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        logger.info(f"[DEBUG] Parsing JSON: {json_str}")
                        data = json.loads(json_str)
                        
                        if "tool_call" in data:
                            tool_data = data["tool_call"]
                            return ToolCall(
                                name=tool_data.get("name", ""),
                                args=tool_data.get("args", {}),
                                confidence=tool_data.get("confidence", 0.8),
                                reasoning=tool_data.get("reasoning", "LLM generated tool call")
                            )
            
            # Look for inline tool call format
            tool_call_match = re.search(r'<(\w+)\s+([^>]+)>', response_text)
            if tool_call_match:
                tool_name = tool_call_match.group(1).upper()
                args_str = tool_call_match.group(2)
                
                # Parse simple args (key=value format)
                args = {}
                for arg in args_str.split():
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        args[key.strip()] = value.strip().strip('"\'')
                
                return ToolCall(
                    name=tool_name,
                    args=args,
                    confidence=0.8,
                    reasoning="Parsed from inline tool call format"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse tool call from LLM response: {e}")
            return None
    
    def remove_tool_call_from_response(self, response_text: str) -> str:
        """Remove JSON tool call from response text to get clean chat response"""
        try:
            # Log the original for debugging
            logger.info(f"[DEBUG] Removing tool calls from: {repr(response_text[:200])}")
            
            # Check if response has meaningful text outside of tool calls
            # Remove code blocks first to see what's left
            temp_cleaned = re.sub(r'```[^`]*?```', '', response_text, flags=re.DOTALL)
            # Check if there's substantial text remaining after JSON removal
            temp_no_json = re.sub(r'\{[^{}]*\}', '', temp_cleaned)
            meaningful_text = temp_no_json.strip()
            
            # Only return empty if there's truly no meaningful content
            if not meaningful_text or len(meaningful_text) < 10:
                logger.info("[DEBUG] Response is entirely tool calls, returning empty")
                return ""
            
            # Remove any code blocks (```json, ```, etc.) - more comprehensive
            cleaned = re.sub(r'```[^`]*?```', '', response_text, flags=re.DOTALL)
            
            # Remove JSON objects with nested braces (more robust)
            # This handles nested JSON better
            while re.search(r'\{[^{}]*\}', cleaned):
                cleaned = re.sub(r'\{[^{}]*\}', '', cleaned)
            
            # Remove inline tool call format
            cleaned = re.sub(r'<\w+[^>]*>', '', cleaned)
            
            # Remove system hints that might leak through
            cleaned = re.sub(r'\[SYSTEM:[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'<SYSTEM_HINT>.*?</SYSTEM_HINT>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any remaining JSON-like patterns
            cleaned = re.sub(r'"[^"]*":\s*"[^"]*"', '', cleaned)
            cleaned = re.sub(r'"[^"]*":\s*\{[^}]*\}', '', cleaned)
            
            # Remove tool call keywords
            cleaned = re.sub(r'\b(tool_call|args|name|DM|CALL|SEARCH)\b', '', cleaned, flags=re.IGNORECASE)
            
            # Clean up extra whitespace and newlines
            cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
            cleaned = re.sub(r'^\s*\n', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
            
            logger.info(f"[DEBUG] Cleaned result: {repr(cleaned[:200])}")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to remove tool call from response: {e}")
            return response_text
    
    def validate_tool_call(self, tool_call: ToolCall, context: ToolContext) -> Tuple[bool, str, Dict]:
        """Validate a tool call before execution"""
        if not tool_call:
            return False, "invalid_args", {"reason": "No tool call provided"}
        
        if tool_call.name not in ["DM", "SEARCH"]:
            return False, "invalid_args", {"reason": f"Unknown tool: {tool_call.name}"}
        
        if tool_call.name == "DM":
            # Validate DM tool call
            if "user_id" not in tool_call.args:
                return False, "invalid_args", {"reason": "Missing user_id for DM"}
            if "message" not in tool_call.args:
                return False, "invalid_args", {"reason": "Missing message for DM"}
            
            # Check if user_id matches context
            if tool_call.args["user_id"] != context.user_id:
                return False, "invalid_args", {"reason": "User ID mismatch"}
        
        elif tool_call.name == "SEARCH":
            # Validate search tool call
            if "query" not in tool_call.args:
                return False, "invalid_args", {"reason": "Missing query for search"}
            
            if len(tool_call.args["query"].strip()) < 2:
                return False, "invalid_args", {"reason": "Search query too short"}
        
        return True, "ok", {}
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics for monitoring"""
        return {
            "patterns": {
                "dm_patterns": len(self.dm_patterns),
                "search_patterns": len(self.search_patterns),
                "chat_patterns": len(self.chat_patterns)
            },
            "method": "neuro_fast",
            "target_latency_ms": "<100"
        }
