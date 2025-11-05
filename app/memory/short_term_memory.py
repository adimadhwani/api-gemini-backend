from collections import deque

class ShortTermMemory:
    def __init__(self, max_size: int = 5):
        self.memory = deque(maxlen=max_size)
    
    def add_query(self, query: str):
        """Add a query to short-term memory"""
        self.memory.append(query)
    
    def get_recent_queries(self) -> list:
        """Get recent queries from memory"""
        return list(self.memory)
    
    def clear_memory(self):
        """Clear the memory"""
        self.memory.clear()