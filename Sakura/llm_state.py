class LLMState:
    def __init__(self):
        self.enabled = True
        self.next_cancelled = False
        
        # Store filtered and unfiltered responses for control panel
        self.last_unfiltered_response = ""
        self.last_filtered_response = ""
