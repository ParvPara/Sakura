import re
import json
import os
from typing import List, Set

class FilterHandler:
    def __init__(self, filter_file: str = "data/filtered_words.json"):
        self.filter_file = filter_file
        self.filtered_words: Set[str] = set()
        self.enabled = True
        self.load_filtered_words()
    
    def load_filtered_words(self):
        """Load filtered words from JSON file"""
        try:
            if os.path.exists(self.filter_file):
                with open(self.filter_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Support both old and new JSON structure
                    if 'filtered_words' in data:
                        words_list = data.get('filtered_words', [])
                    elif 'words' in data:
                        words_list = data.get('words', [])
                    else:
                        words_list = []
                    
                    # Filter out empty strings and whitespace-only strings
                    self.filtered_words = set(word.strip() for word in words_list if word and word.strip())
                
                # Load enabled state
                self.enabled = data.get('enabled', True)
                print(f"[FILTER] Loaded {len(self.filtered_words)} filtered words (enabled: {self.enabled})")
            else:
                # Create default filter file with some example words
                self.filtered_words = set()
                self.save_filtered_words()
                print(f"[FILTER] Created new filter file at {self.filter_file}")
        except Exception as e:
            print(f"[FILTER] Error loading filter file: {e}")
            self.filtered_words = set()
    
    def save_filtered_words(self):
        """Save filtered words to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.filter_file), exist_ok=True)
            with open(self.filter_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'enabled': self.enabled,
                    'words': list(self.filtered_words)
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[FILTER] Error saving filter file: {e}")
    
    def add_filtered_word(self, word: str):
        """Add a word to the filter list"""
        word = word.lower().strip()
        if word:
            self.filtered_words.add(word)
            self.save_filtered_words()
            print(f"[FILTER] Added '{word}' to filter list")
    
    def remove_filtered_word(self, word: str):
        """Remove a word from the filter list"""
        word = word.lower().strip()
        if word in self.filtered_words:
            self.filtered_words.remove(word)
            self.save_filtered_words()
            print(f"[FILTER] Removed '{word}' from filter list")
    
    def get_filtered_words(self) -> List[str]:
        """Get list of all filtered words"""
        return sorted(list(self.filtered_words))
    
    def clear_filtered_words(self):
        """Clear all filtered words"""
        self.filtered_words.clear()
        self.save_filtered_words()
        print("[FILTER] Cleared all filtered words")
    
    def filter_text(self, text: str) -> str:
        """Filter text and replace filtered words with 'Filtered'"""
        if not text or not self.filtered_words or not self.enabled:
            return text
        
        # Convert text to lowercase for comparison
        text_lower = text.lower()
        original_text = text
        
        # Check each filtered word
        for word in self.filtered_words:
            if word.lower() in text_lower:
                # Use regex to replace the word while preserving case
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                text = pattern.sub("Filtered", text)
                print(f"[FILTER] Replaced '{word}' with 'Filtered' in response")
        
        return text
    
    def is_filtered(self, text: str) -> bool:
        """Check if text contains any filtered words"""
        if not text or not self.filtered_words:
            return False
        
        text_lower = text.lower()
        for word in self.filtered_words:
            if word.lower() in text_lower:
                return True
        return False
    
    def filter_message_completely(self, text: str) -> str:
        """Filter text and replace entire message with 'Filtered.' if any filtered words found"""
        if not text or not self.filtered_words or not self.enabled:
            return text
        
        # Check if any filtered words exist
        text_lower = text.lower()
        for word in self.filtered_words:
            if word.lower() in text_lower:
                print(f"[FILTER] Message contains filtered word '{word}', replacing entire message with 'Filtered.'")
                return "Filtered."
        
        return text
    
    def enable_filter(self):
        """Enable the word filter"""
        self.enabled = True
        self.save_filtered_words()
        print("[FILTER] Word filter enabled")
    
    def disable_filter(self):
        """Disable the word filter"""
        self.enabled = False
        self.save_filtered_words()
        print("[FILTER] Word filter disabled")
    
    def is_enabled(self) -> bool:
        """Check if the filter is enabled"""
        return self.enabled
    
    def toggle_filter(self) -> bool:
        """Toggle the filter on/off and return new state"""
        self.enabled = not self.enabled
        self.save_filtered_words()
        print(f"[FILTER] Word filter {'enabled' if self.enabled else 'disabled'}")
        return self.enabled
