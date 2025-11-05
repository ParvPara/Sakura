#!/usr/bin/env python3
"""
Simple client for the Tool Router Service
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any

class RouterClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip('/')
    
    async def route_message(
        self, 
        text: str, 
        channel_type: str = "public", 
        mentions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Route a message to determine the appropriate action"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": text,
                "channel_type": channel_type
            }
            if mentions:
                payload["mentions"] = mentions
            
            async with session.post(
                f"{self.base_url}/route",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Router error {response.status}: {error_text}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check router health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Health check failed: {response.status}")

async def _test_router():
    """Test the router with various message types"""
    client = RouterClient()
    
    # Test cases
    test_cases = [
        {
            "text": "What's the weather like today?",
            "expected": "SEARCH",
            "description": "Search-like question"
        },
        {
            "text": "@mod can you help me with something private?",
            "expected": "DM",
            "description": "DM request with mention"
        },
        {
            "text": "Hello! How are you doing?",
            "expected": "NONE",
            "description": "Simple greeting"
        },
        {
            "text": "Search for the latest news about AI",
            "expected": "SEARCH",
            "description": "Explicit search request"
        },
        {
            "text": "Can you DM me the details?",
            "expected": "DM",
            "description": "Direct DM request"
        }
    ]
    
    print("Testing Tool Router...")
    print("=" * 50)
    
    # Health check
    try:
        health = await client.health_check()
        print(f"Health: {health}")
        if not health.get('ok'):
            print("Router is not healthy!")
            return
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Run test cases
    for i, test in enumerate(test_cases, 1):
        try:
            mentions = None
            if "@mod" in test["text"]:
                mentions = "@mod"
            
            result = await client.route_message(
                text=test["text"],
                mentions=mentions
            )
            
            status = "✓" if result["action"] == test["expected"] else "✗"
            print(f"{i}. {status} {test['description']}")
            print(f"   Input: {test['text']}")
            print(f"   Expected: {test['expected']}, Got: {result['action']}")
            print(f"   Latency: {result['latency_ms']}ms")
            print(f"   Raw: '{result['raw']}'")
            print()
            
        except Exception as e:
            print(f"{i}. ✗ {test['description']} - Error: {e}")
            print()

async def benchmark_router():
    """Benchmark router latency"""
    client = RouterClient()
    
    test_message = "What's the weather like today?"
    num_requests = 20
    
    print(f"Benchmarking router with {num_requests} requests...")
    print("=" * 50)
    
    latencies = []
    
    for i in range(num_requests):
        try:
            start_time = asyncio.get_event_loop().time()
            result = await client.route_message(test_message)
            end_time = asyncio.get_event_loop().time()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            
            print(f"Request {i+1}: {latency:.1f}ms")
            
        except Exception as e:
            print(f"Request {i+1}: Error - {e}")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Calculate 95th percentile
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else max_latency
        
        print("\nResults:")
        print(f"Average: {avg_latency:.1f}ms")
        print(f"Min: {min_latency:.1f}ms")
        print(f"Max: {max_latency:.1f}ms")
        print(f"95th percentile: {p95_latency:.1f}ms")
        
        if p95_latency <= 300:
            print("✓ Meets latency target (≤300ms)")
        else:
            print("✗ Exceeds latency target (>300ms)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        asyncio.run(benchmark_router())
    else:
        asyncio.run(test_router())
