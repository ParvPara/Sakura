"""
Web Search Tool using Brave Search API
Provides web search functionality for the Sakura bot
"""

import requests
import json
import time
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import quote_plus
import config

@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    description: str
    source: str
    published_date: Optional[str] = None
    language: Optional[str] = None

@dataclass
class SearchResponse:
    """Represents a complete search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    success: bool
    error_message: Optional[str] = None

class BraveWebSearch:
    """Web search tool using Brave Search API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Brave Web Search tool
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from config
        """
        self.api_key = api_key or getattr(config, 'BRAVE_SEARCH_API_KEY', None)
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        } if self.api_key else {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum 500ms between requests
        
        # Search options
        self.default_params = {
            "count": 5,  # Number of results to return
            "search_lang": "en_US",  # Search language
            "ui_lang": "en",  # UI language
            "safesearch": "moderate",  # Safe search level
            "freshness": "pd",  # Past day
            "text_decorations": False,  # No text decorations
            "spellcheck": True,  # Enable spellcheck
        }
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-.,!?;:()]', '', text)
        
        return text.strip()
    
    def _parse_search_result(self, result: Dict[str, Any]) -> SearchResult:
        """Parse a single search result from the API response"""
        return SearchResult(
            title=self._clean_text(result.get('title', '')),
            url=result.get('url', ''),
            description=self._clean_text(result.get('description', '')),
            source=result.get('source', ''),
            published_date=result.get('published_date'),
            language=result.get('language')
        )
    
    def search(self, query: str, count: int = 5, search_type: str = "web") -> SearchResponse:
        """
        Perform a web search using Brave Search API
        
        Args:
            query: Search query string
            count: Number of results to return (max 10)
            search_type: Type of search ("web", "news", "videos", "images")
            
        Returns:
            SearchResponse object with results and metadata
        """
        if not self.api_key:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0,
                success=False,
                error_message="Brave Search API key not configured"
            )
        
        if not query or not query.strip():
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0,
                success=False,
                error_message="Empty search query"
            )
        
        # Rate limiting
        self._rate_limit()
        
        # Prepare search parameters
        params = self.default_params.copy()
        params.update({
            "q": query.strip(),
            "count": min(count, 10),  # API limit is 10
            "search_lang": "en_US",
        })
        
        # Adjust endpoint based on search type
        endpoint = self.base_url
        if search_type == "news":
            endpoint = "https://api.search.brave.com/news/search"
        elif search_type == "videos":
            endpoint = "https://api.search.brave.com/videos/search"
        elif search_type == "images":
            endpoint = "https://api.search.brave.com/images/search"
        
        start_time = time.time()
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            search_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse results based on search type
                if search_type == "web":
                    results = data.get('web', {}).get('results', [])
                    total_results = data.get('web', {}).get('total', 0)
                elif search_type == "news":
                    results = data.get('news', {}).get('results', [])
                    total_results = data.get('news', {}).get('total', 0)
                elif search_type == "videos":
                    results = data.get('videos', {}).get('results', [])
                    total_results = data.get('videos', {}).get('total', 0)
                elif search_type == "images":
                    results = data.get('images', {}).get('results', [])
                    total_results = data.get('images', {}).get('total', 0)
                else:
                    results = []
                    total_results = 0
                
                # Parse search results
                parsed_results = []
                for result in results:
                    try:
                        parsed_result = self._parse_search_result(result)
                        if parsed_result.title and parsed_result.url:
                            parsed_results.append(parsed_result)
                    except Exception as e:
                        print(f"[WEBSEARCH] Error parsing result: {e}")
                        continue
                
                return SearchResponse(
                    query=query,
                    results=parsed_results,
                    total_results=total_results,
                    search_time=search_time,
                    success=True
                )
                
            else:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        error_msg = error_data['message']
                except:
                    pass
                
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=time.time() - start_time,
                    success=False,
                    error_message=error_msg
                )
                
        except requests.exceptions.Timeout:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                success=False,
                error_message="Search request timed out"
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                success=False,
                error_message=f"Network error: {str(e)}"
            )
        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def search_web(self, query: str, count: int = 5) -> SearchResponse:
        """Perform a general web search"""
        return self.search(query, count, "web")
    
    def search_news(self, query: str, count: int = 5) -> SearchResponse:
        """Perform a news search"""
        return self.search(query, count, "news")
    
    def search_videos(self, query: str, count: int = 5) -> SearchResponse:
        """Perform a video search"""
        return self.search(query, count, "videos")
    
    def search_images(self, query: str, count: int = 5) -> SearchResponse:
        """Perform an image search"""
        return self.search(query, count, "images")
    
    def format_results(self, search_response: SearchResponse, max_length: int = 2000) -> str:
        """
        Format search results into a readable string
        
        Args:
            search_response: SearchResponse object
            max_length: Maximum length of formatted string
            
        Returns:
            Formatted string with search results
        """
        if not search_response.success:
            return f"❌ Search failed: {search_response.error_message}"
        
        if not search_response.results:
            return f"🔍 No results found for '{search_response.query}'"
        
        # Build formatted output
        lines = []
        lines.append(f"🔍 **Search Results for '{search_response.query}'**")
        lines.append(f"📊 Found {search_response.total_results} results in {search_response.search_time:.2f}s")
        lines.append("")
        
        for i, result in enumerate(search_response.results, 1):
            # Truncate description if too long
            desc = result.description
            if len(desc) > 150:
                desc = desc[:147] + "..."
            
            lines.append(f"**{i}. {result.title}**")
            lines.append(f"🔗 {result.url}")
            lines.append(f"📝 {desc}")
            if result.source:
                lines.append(f"📰 Source: {result.source}")
            lines.append("")
        
        formatted = "\n".join(lines)
        
        # Truncate if too long
        if len(formatted) > max_length:
            formatted = formatted[:max_length-3] + "..."
        
        return formatted
    
    def get_summary(self, search_response: SearchResponse) -> str:
        """
        Get a brief summary of search results
        
        Args:
            search_response: SearchResponse object
            
        Returns:
            Brief summary string
        """
        if not search_response.success:
            return f"Search failed: {search_response.error_message}"
        
        if not search_response.results:
            return f"No results found for '{search_response.query}'"
        
        # Create a brief summary
        summary_parts = []
        summary_parts.append(f"Found {search_response.total_results} results for '{search_response.query}'")
        
        # Add top result info
        if search_response.results:
            top_result = search_response.results[0]
            summary_parts.append(f"Top result: {top_result.title}")
            if top_result.description:
                desc = top_result.description[:100] + "..." if len(top_result.description) > 100 else top_result.description
                summary_parts.append(f"Summary: {desc}")
        
        return " | ".join(summary_parts)

# Global instance for easy access
web_search = None

def initialize_web_search(api_key: Optional[str] = None) -> BraveWebSearch:
    """Initialize the global web search instance"""
    global web_search
    web_search = BraveWebSearch(api_key)
    return web_search

def get_web_search() -> Optional[BraveWebSearch]:
    """Get the global web search instance"""
    return web_search

# Example usage and testing
if __name__ == "__main__":
    # Test the web search functionality
    print("Testing Brave Web Search...")
    
    # Initialize with API key from config (if available)
    searcher = initialize_web_search()
    
    # Test search
    test_query = "latest AI developments 2024"
    print(f"\nSearching for: {test_query}")
    
    result = searcher.search_web(test_query, count=3)
    
    if result.success:
        print(f"✅ Found {len(result.results)} results")
        print(searcher.format_results(result))
    else:
        print(f"❌ Search failed: {result.error_message}")
    
    # Test news search
    print(f"\nSearching news for: {test_query}")
    news_result = searcher.search_news(test_query, count=2)
    
    if news_result.success:
        print(f"✅ Found {len(news_result.results)} news results")
        print(searcher.format_results(news_result))
    else:
        print(f"❌ News search failed: {news_result.error_message}")

