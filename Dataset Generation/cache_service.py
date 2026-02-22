"""
LRU Cache Service for managing model downloads
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional


class Node:
    """Node for doubly-linked list used in LRU cache"""
    def __init__(self, filename: str, size_mb: float) -> None:
        self.filename: str           = filename
        self.size_mb:  float         = size_mb
        self.next:     Optional[Node] = None
        self.prev:     Optional[Node] = None


class LRUCacheService:
    """LRU Cache service for managing model storage with automatic eviction"""
    
    def __init__(self, cache_dir: Path, capacity_gb: float = 25):
        """
        Initialize LRU Cache Service
        
        Args:
            cache_dir: Directory where cached models are stored
            capacity_gb: Maximum cache size in GB (default: 25)
        """
        self.cache_dir = cache_dir
        self.capacity_gb = capacity_gb
        self.capacity_mb = capacity_gb * 1024
        
        # Doubly-linked list with dummy head and tail
        self.head = Node("HEAD", 0)
        self.tail = Node("TAIL", 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Map: filename -> Node
        self.cache_map: Dict[str, Node] = {}
    
    def add(self, filename: str, size_mb: float) -> None:
        """
        Add a file to the cache (marks as most recently used)
        
        Args:
            filename: Name of the file
            size_mb: Size of the file in MB
        """
        if filename in self.cache_map:
            # Already exists, move to front
            self._move_to_front(filename)
            return
        
        # Create new node and add to front
        node = Node(filename, size_mb)
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.cache_map[filename] = node
    
    def access(self, filename: str) -> None:
        """
        Mark a file as recently accessed (moves to front)
        
        Args:
            filename: Name of the file
        """
        if filename in self.cache_map:
            self._move_to_front(filename)
    
    def remove(self, filename: str) -> bool:
        """
        Remove a file from the cache
        
        Args:
            filename: Name of the file to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        if filename not in self.cache_map:
            return False
        
        node = self.cache_map[filename]
        self._remove_node(node)
        del self.cache_map[filename]
        return True
    
    def get_total_size_mb(self) -> float:
        """
        Calculate total size of all cached files
        
        Returns:
            Total size in MB
        """
        total = 0.0
        for node in self.cache_map.values():
            total += node.size_mb
        return total
    
    def get_total_size_gb(self) -> float:
        """
        Calculate total size of all cached files
        
        Returns:
            Total size in GB
        """
        return self.get_total_size_mb() / 1024
    
    def has_space_for(self, size_mb: float) -> bool:
        """
        Check if there's enough space for a new file
        
        Args:
            size_mb: Size of the file to check
            
        Returns:
            True if there's space, False otherwise
        """
        current_size = self.get_total_size_mb()
        return current_size + size_mb <= self.capacity_mb
    
    def ensure_space_for(self, size_mb: float, verbose: bool = True) -> bool:
        """
        Ensure there's enough space by evicting LRU files if necessary
        
        Args:
            size_mb: Size needed in MB
            verbose: Whether to print eviction messages
            
        Returns:
            True if space was ensured, False if couldn't free enough space
        """
        current_size = self.get_total_size_mb()
        
        if verbose:
            print(f"\nCache status: {current_size:.2f} MB / {self.capacity_mb:.2f} MB ({self.capacity_gb} GB)")
            print(f"Estimated download size: {size_mb:.2f} MB")
        
        # Evict until we have enough space
        while current_size + size_mb > self.capacity_mb:
            evicted = self._evict_lru(verbose)
            if not evicted:
                if verbose:
                    print("   Warning: Could not free enough space")
                return False
            current_size = self.get_total_size_mb()
            if verbose:
                print(f"   New cache size: {current_size:.2f} MB")
        
        return True
    
    def _move_to_front(self, filename: str) -> None:
        """Move an existing node to the front (mark as recently used)"""
        if filename not in self.cache_map:
            return
        
        node = self.cache_map[filename]
        self._remove_node(node)
        
        # Add to front
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove a node from the doubly-linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _evict_lru(self, verbose: bool = True) -> bool:
        """
        Evict the least recently used file
        
        Args:
            verbose: Whether to print eviction message
            
        Returns:
            True if evicted successfully, False if cache is empty
        """
        if self.tail.prev == self.head:
            # Cache is empty
            return False
        
        # Get LRU node (before tail)
        lru_node = self.tail.prev
        filename = lru_node.filename
        
        # Remove from disk
        file_path = self.cache_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                if verbose:
                    print(f"   Evicted LRU model: {filename} ({lru_node.size_mb:.2f} MB)")
            except Exception as e:
                if verbose:
                    print(f"   Error evicting {filename}: {e}")
                return False
        
        # Remove from cache
        self._remove_node(lru_node)
        del self.cache_map[filename]
        return True
    
    def initialize_from_disk(self) -> None:
        """Load all existing .gguf files from disk into the cache"""
        for file_path in self.cache_dir.glob('*.gguf'):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            self.add(file_path.name, size_mb)
    
    def get_stats(self) -> Dict[str, float | int]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache stats
        """
        return {
            'total_files': len(self.cache_map),
            'total_size_mb': self.get_total_size_mb(),
            'total_size_gb': self.get_total_size_gb(),
            'capacity_gb': self.capacity_gb,
            'usage_percent': (self.get_total_size_mb() / self.capacity_mb) * 100 if self.capacity_mb > 0 else 0
        }
