#!/usr/bin/env python3
"""
Advanced Memory System for Sakura Bot
Handles people recognition, context management, and memory organization
"""

import json
import time
import os
import threading
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class Person:
    """Represents a person in the memory system"""
    name: str
    discord_username: Optional[str] = None
    aliases: List[str] = None
    traits: List[str] = None
    interests: List[str] = None
    relationship: str = "friend"
    first_seen: str = None
    last_seen: str = None
    interaction_count: int = 0
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.traits is None:
            self.traits = []
        if self.interests is None:
            self.interests = []
        if self.first_seen is None:
            self.first_seen = datetime.now().isoformat()
        if self.last_seen is None:
            self.last_seen = datetime.now().isoformat()

@dataclass
class Memory:
    """Represents a memory entry"""
    content: str
    memory_type: str  # "person", "fact", "conversation", "event"
    people_involved: List[str] = None
    importance: int = 5  # 1-10 scale
    timestamp: str = None
    context: str = ""
    
    def __post_init__(self):
        if self.people_involved is None:
            self.people_involved = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class AdvancedMemorySystem:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.memories_file = os.path.join(data_dir, "advanced_memories.json")
        self.people_file = os.path.join(data_dir, "people.json")
        self.lock = threading.RLock()
        
        # Load existing data
        self.people: Dict[str, Person] = {}
        self.short_term_memories: List[Memory] = []
        self.long_term_memories: List[Memory] = []
        
        self.load_data()
        
        # Configuration
        self.max_short_term = 20
        self.max_long_term = 100
        self.short_term_expiry_hours = 24
        
    def load_data(self):
        """Load people and memories from files"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load people
        if os.path.exists(self.people_file):
            try:
                with open(self.people_file, 'r') as f:
                    people_data = json.load(f)
                    for name, data in people_data.items():
                        self.people[name] = Person(**data)
            except Exception as e:
                print(f"Error loading people data: {e}")
        
        # Load memories
        if os.path.exists(self.memories_file):
            try:
                with open(self.memories_file, 'r') as f:
                    memories_data = json.load(f)
                    self.short_term_memories = [Memory(**m) for m in memories_data.get("short_term", [])]
                    self.long_term_memories = [Memory(**m) for m in memories_data.get("long_term", [])]
            except Exception as e:
                print(f"Error loading memories data: {e}")
    
    def save_data(self):
        """Save people and memories to files"""
        with self.lock:
            # Save people
            people_data = {name: asdict(person) for name, person in self.people.items()}
            with open(self.people_file, 'w') as f:
                json.dump(people_data, f, indent=2)
            
            # Save memories
            memories_data = {
                "short_term": [asdict(m) for m in self.short_term_memories],
                "long_term": [asdict(m) for m in self.long_term_memories]
            }
            with open(self.memories_file, 'w') as f:
                json.dump(memories_data, f, indent=2)
    
    def add_person(self, name: str, discord_username: str = None, **kwargs) -> Person:
        """Add or update a person in the system"""
        with self.lock:
            if name not in self.people:
                person = Person(name=name, discord_username=discord_username, **kwargs)
                self.people[name] = person
                print(f"[MEMORY] Added new person: {name}")
            else:
                person = self.people[name]
                person.last_seen = datetime.now().isoformat()
                person.interaction_count += 1
                if discord_username and discord_username != person.discord_username:
                    person.discord_username = discord_username
                print(f"[MEMORY] Updated person: {name}")
            
            self.save_data()
            return person
    
    def get_person(self, name_or_username: str) -> Optional[Person]:
        """Get a person by name or Discord username"""
        with self.lock:
            # First check Discord username match (prioritize this for Discord usernames)
            for person in self.people.values():
                if person.discord_username and person.discord_username.lower() == name_or_username.lower():
                    return person
                if name_or_username.lower() in [alias.lower() for alias in person.aliases]:
                    return person
            
            # Then check direct name match
            if name_or_username in self.people:
                return self.people[name_or_username]
            
            return None
    
    def extract_people_from_text(self, text: str) -> List[str]:
        """Extract potential people names from text"""
        # Look for capitalized words that might be names
        words = text.split()
        potential_names = []
        
        for word in words:
            # Skip common words and short words
            if (len(word) > 2 and 
                word[0].isupper() and 
                word.lower() not in ['the', 'and', 'but', 'for', 'with', 'this', 'that', 'they', 'their', 'have', 'been', 'from', 'were', 'said', 'each', 'which', 'she', 'will', 'more', 'when', 'there', 'can', 'an', 'its', 'it\'s', 'out', 'use', 'word', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'one', 'about', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'has', 'look', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'over', 'new', 'work', 'first', 'well', 'way', 'even', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'here', 'just', 'know', 'take', 'every', 'good', 'through', 'very', 'think', 'that', 'look', 'back', 'after', 'between', 'never', 'become', 'another', 'might', 'seem', 'should', 'home', 'big', 'give', 'air', 'line', 'set', 'own', 'under', 'read', 'last', 'never', 'us', 'left', 'end', 'along', 'while', 'might', 'next', 'sound', 'below', 'saw', 'something', 'thought', 'both', 'few', 'those', 'always', 'show', 'large', 'often', 'together', 'asked', 'don\'t', 'world', 'going', 'want', 'school', 'important', 'until', 'form', 'food', 'keep', 'children', 'feet', 'land', 'side', 'without', 'boy', 'once', 'animal', 'life', 'enough', 'took', 'sometimes', 'four', 'head', 'above', 'kind', 'began', 'almost', 'live', 'page', 'got', 'earth', 'need', 'far', 'hand', 'high', 'year', 'mother', 'light', 'country', 'father', 'let', 'night', 'picture', 'being', 'study', 'second', 'soon', 'story', 'since', 'white', 'ever', 'paper', 'hard', 'near', 'sentence', 'better', 'best', 'across', 'during', 'today', 'however', 'sure', 'knew', 'it\'s', 'try', 'told', 'young', 'sun', 'thing', 'whole', 'hear', 'example', 'heard', 'several', 'change', 'answer', 'room', 'sea', 'against', 'top', 'turned', 'learn', 'point', 'city', 'play', 'toward', 'five', 'himself', 'usually', 'money', 'seen', 'didn\'t', 'car', 'morning', 'I\'m', 'body', 'upon', 'family', 'music', 'bring', 'color', 'stand', 'sun', 'questions', 'fish', 'area', 'mark', 'dog', 'horse', 'birds', 'problem', 'complete', 'room', 'knew', 'since', 'ever', 'piece', 'told', 'usually', 'didn\'t', 'friends', 'easy', 'heard', 'order', 'red', 'door', 'sure', 'become', 'top', 'ship', 'across', 'today', 'during', 'short', 'better', 'best', 'however', 'low', 'hours', 'black', 'products', 'happened', 'whole', 'measure', 'remember', 'early', 'waves', 'reached', 'listen', 'wind', 'rock', 'space', 'covered', 'fast', 'several', 'hold', 'himself', 'toward', 'five', 'step', 'morning', 'passed', 'vowel', 'true', 'hundred', 'against', 'pattern', 'numeral', 'table', 'north', 'slowly', 'money', 'map', 'farm', 'pulled', 'draw', 'voice', 'seen', 'cold', 'cried', 'plan', 'notice', 'south', 'sing', 'war', 'ground', 'fall', 'king', 'town', 'I\'ll', 'unit', 'figure', 'certain', 'field', 'travel', 'wood', 'fire', 'upon']):
                potential_names.append(word)
        
        return potential_names
    
    def add_short_term_memory(self, content: str, people_involved: List[str] = None, context: str = ""):
        """Add a short-term memory"""
        with self.lock:
            if people_involved is None:
                people_involved = self.extract_people_from_text(content)
            
            memory = Memory(
                content=content,
                memory_type="conversation",
                people_involved=people_involved,
                importance=5,
                context=context
            )
            
            self.short_term_memories.append(memory)
            
            # Trim to max size
            if len(self.short_term_memories) > self.max_short_term:
                self.short_term_memories = self.short_term_memories[-self.max_short_term:]
            
            # Update people interaction counts
            for person_name in people_involved:
                person = self.get_person(person_name)
                if person:
                    person.interaction_count += 1
                    person.last_seen = datetime.now().isoformat()
            
            self.save_data()
    
    def add_long_term_memory(self, content: str, memory_type: str = "fact", people_involved: List[str] = None, importance: int = 5, context: str = ""):
        """Add a long-term memory"""
        with self.lock:
            if people_involved is None:
                people_involved = self.extract_people_from_text(content)
            
            memory = Memory(
                content=content,
                memory_type=memory_type,
                people_involved=people_involved,
                importance=importance,
                context=context
            )
            
            # Avoid duplicates
            for existing in self.long_term_memories:
                if existing.content == content:
                    return
            
            self.long_term_memories.append(memory)
            
            # Sort by importance and trim
            self.long_term_memories.sort(key=lambda x: x.importance, reverse=True)
            if len(self.long_term_memories) > self.max_long_term:
                self.long_term_memories = self.long_term_memories[:self.max_long_term]
            
            self.save_data()
    
    def get_relevant_memories(self, current_context: str, people_mentioned: List[str] = None) -> Dict[str, List[str]]:
        """Get memories relevant to the current context"""
        with self.lock:
            if people_mentioned is None:
                people_mentioned = self.extract_people_from_text(current_context)
            
            relevant_short_term = []
            relevant_long_term = []
            
            # Get short-term memories (recent conversations)
            for memory in self.short_term_memories[-6:]:  # Last 6
                relevant_short_term.append(memory.content)
            
            # Get long-term memories about mentioned people
            for memory in self.long_term_memories:
                # Check if any mentioned people are involved
                if any(person in memory.people_involved for person in people_mentioned):
                    relevant_long_term.append(memory.content)
                    if len(relevant_long_term) >= 5:  # Limit to 5 most relevant
                        break
            
            return {
                "short_term": relevant_short_term,
                "long_term": relevant_long_term
            }
    
    def get_person_context(self, person_name: str) -> Dict[str, Any]:
        """Get comprehensive context about a person"""
        with self.lock:
            person = self.get_person(person_name)
            if not person:
                return {}
            
            # Get memories about this person
            person_memories = []
            for memory in self.long_term_memories:
                if person_name in memory.people_involved:
                    person_memories.append(memory.content)
            
            return {
                "name": person.name,
                "discord_username": person.discord_username,
                "aliases": person.aliases,
                "traits": person.traits,
                "interests": person.interests,
                "relationship": person.relationship,
                "interaction_count": person.interaction_count,
                "memories": person_memories[-5:]  # Last 5 memories
            }
    
    def cleanup_expired_memories(self):
        """Remove expired short-term memories"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=self.short_term_expiry_hours)
            self.short_term_memories = [
                m for m in self.short_term_memories 
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            self.save_data()
    
    def get_memory_summary(self) -> str:
        """Get a summary of the memory system"""
        with self.lock:
            return f"People: {len(self.people)}, Short-term: {len(self.short_term_memories)}, Long-term: {len(self.long_term_memories)}"
