"""
ModelDownloader - Simple interface for downloading GGUF models from HuggingFace
"""
from __future__ import annotations
from huggingface_hub import hf_hub_download, list_repo_files
import os
from pathlib import Path
from typing import List, Optional, Dict

# Import cache service
from cache_service import LRUCacheService

# Import global model configuration
try:
    from model_config import MODEL_REPOS, QUANT, DEFAULT_MODEL, DEFAULT_CACHE_DIR, get_model_repo
except ImportError:
    # Fallback if model_config not available
    MODEL_REPOS = {}
    QUANT = {}
    DEFAULT_MODEL = "TheBloke/Llama-2-7B-Chat-GGUF"
    DEFAULT_CACHE_DIR = "../models"
    get_model_repo = lambda x: x

    
class ModelDownloader:
    """Simple manager for downloading GGUF models from HuggingFace"""
    
    # Default cache capacity in GB
    CACHE_CAPACITY_GB = 25
    
    
    def __init__(self, repo_id: str | None = None, cache_dir: str | None = None, cache_capacity_gb: float | None = None) -> None:
        """
        Initialize ModelDownloader
        
        Args:
            repo_id: HuggingFace repository ID, or shortcut from model_config.MODEL_REPOS
                     Examples: "TheBloke/Llama-2-7B-Chat-GGUF" or "LLAMA2_7B"
                     If None, uses DEFAULT_MODEL from config
            cache_dir: Where to save models (default: from DEFAULT_CACHE_DIR)
            cache_capacity_gb: Cache capacity in GB (default: 25)
        """
        # Use default if not provided
        if repo_id is None:
            repo_id = DEFAULT_MODEL
        
        # Resolve shortcuts from model_config.py
        self.repo_id = get_model_repo(repo_id)
        
        # Use default cache dir if not provided
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect available files
        self._available_files = None
        
        # Initialize LRU cache service
        capacity = cache_capacity_gb if cache_capacity_gb is not None else self.CACHE_CAPACITY_GB
        self.cache_service = LRUCacheService(self.cache_dir, capacity)
        self.cache_service.initialize_from_disk()
    
    
    def _get_available_files(self) -> List[str]:
        """Fetch all .gguf files from the HuggingFace repo"""
        if self._available_files is None:
            try:
                print(f"Checking available files in {self.repo_id}...")
                all_files = list_repo_files(self.repo_id)
                self._available_files = [f for f in all_files if f.endswith('.gguf')]
                print(f"   Found {len(self._available_files)} GGUF files")
            except Exception as e:
                print(f"Error listing files: {e}")
                self._available_files = []
        return self._available_files
    
    def list_available(self) -> List[str]:
        """
        List all available .gguf files in the repository
        
        Returns:
            List of filenames
        """
        return self._get_available_files()
    
    def download(self, filename_or_quantization: str) -> Optional[str]:
        """
        Download a model file - SIMPLE VERSION
        
        Args:
            filename_or_quantization: Either a full filename (e.g., "model.Q4_K_M.gguf") 
                                     or just a quantization level (e.g., "Q4_K_M")
            
        Returns:
            Path to downloaded model file, or None if failed
        """
        
        # Get available files
        available = self._get_available_files()
        
        # Determine the filename
        filename = None
        
        # Check if it's already a full filename
        if filename_or_quantization.endswith('.gguf'):
            if filename_or_quantization in available:
                filename = filename_or_quantization
            else:
                print(f"Error: File {filename_or_quantization} not found in repo")
                return None
        else:
            # Try to find a file with this quantization level
            quant = filename_or_quantization.upper()
            matching = [f for f in available if f'.{quant}.gguf' in f or f'-{quant}.gguf' in f]
            
            if not matching:
                print(f"Error: No files found with quantization {quant}")
                print(f"   Available files: {available[:5]}...")  # Show first 5
                return None
            
            if len(matching) > 1:
                print(f"Warning: Multiple files found for {quant}:")
                for i, f in enumerate(matching, 1):
                    print(f"   {i}. {f}")
                print(f"   Using: {matching[0]}")
            
            filename = matching[0]
        
        # Check if file already exists in cache
        cached_path = self.cache_dir / filename
        if cached_path.exists():
            size_mb = cached_path.stat().st_size / (1024 * 1024)
            print(f"\nModel already cached: {filename}")
            print(f"   Location: {cached_path}")
            print(f"   Size: {size_mb:.2f} MB")
            print("   Skipping download.")
            # Mark as recently used
            self.cache_service.access(filename)
            return str(cached_path)
        
        # Estimate file size based on quantization (rough estimates in MB)
        estimated_size_mb = 4000  # Default ~4GB
        size_map = {
            "Q2_K": 2500, "Q3_K": 3300, "Q4_K": 4000, "Q4_0": 4000, "Q4_1": 4000,
            "Q5_K": 4800, "Q5_0": 5000, "Q5_1": 5000,
            "Q6_K": 5500, "Q8_0": 7200, "F16": 13000
        }
        for quant, size in size_map.items():
            if quant in filename:
                estimated_size_mb = size
                break
        
        # Check if we need to evict models to make space
        self.cache_service.ensure_space_for(estimated_size_mb)
        
        print(f"\nDownloading: {filename}")
        print("   This may take a while...\n")
        
        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False
            )
            
            # Get actual file size and add to cache
            actual_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            self.cache_service.add(filename, actual_size_mb)
            
            print(f"Downloaded to: {model_path}")
            print(f"Actual size: {actual_size_mb:.2f} MB")
            return model_path
        except Exception as e:
            print(f"Error downloading: {e}")
            return None
    
    def get_cached_models(self) -> List[Dict[str, str]]:
        """
        List all .gguf models currently in the cache directory
        
        Returns:
            List of dicts with 'filename', 'path', and 'size'
        """
        cached = []
        
        for file_path in self.cache_dir.glob('*.gguf'):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            cached.append({
                'filename': file_path.name,
                'path': str(file_path),
                'size': f"{size_mb:.2f} MB"
            })
        
        return sorted(cached, key=lambda x: x['filename'])
    
    def delete_model(self, filename: str) -> bool:
        """
        Delete a cached model file
        
        Args:
            filename: Name of the file to delete (e.g., "model.Q4_K_M.gguf")
            
        Returns:
            True if deleted successfully, False otherwise
        """
        # Handle both full filename and just the quantization level
        if not filename.endswith('.gguf'):
            # Try to find a file with this quantization
            matches = list(self.cache_dir.glob(f'*{filename}*.gguf'))
            if not matches:
                print(f"Error: No cached files found matching: {filename}")
                return False
            if len(matches) > 1:
                print(f"Warning: Multiple files match {filename}:")
                for m in matches:
                    print(f"   - {m.name}")
                print("   Please specify the full filename")
                return False
            file_path = matches[0]
        else:
            file_path = self.cache_dir / filename
        
        if not file_path.exists():
            print(f"Warning: File not found: {filename}")
            return False
        
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            actual_filename = file_path.name
            file_path.unlink()
            
            # Remove from cache service
            self.cache_service.remove(actual_filename)
            
            print(f"Deleted {actual_filename} - freed {size_mb:.2f} MB")
            return True
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            return False
    
    def print_available_files(self) -> None:
        """Print all available GGUF files in the HuggingFace repo"""
        files = self._get_available_files()
        
        if not files:
            print("\nNo .gguf files found in repository\n")
            return
        
        print(f"\nAvailable Models in {self.repo_id} ({len(files)} files):\n")
        
        for f in files:
            print(f"   {f}")
        
        print()
    
    def print_cached_models(self) -> None:
        """Print all models currently in cache"""
        cached = self.get_cached_models()
        
        if not cached:
            print("\nNo models cached yet.\n")
            return
        
        print(f"\nCached Models ({len(cached)}):\n")
        for model in cached:
            print(f"   {model['filename']} - {model['size']}")
        
        # Get cache stats from service
        stats = self.cache_service.get_stats()
        print(f"\n   Total: {stats['total_size_mb']:.2f} MB ({stats['total_size_gb']:.2f} GB)")
        print(f"   Capacity: {stats['capacity_gb']} GB")
        print(f"   Usage: {stats['usage_percent']:.1f}%")
        print()


# ============================================================================
# CLI Interface - SIMPLE VERSION
# ============================================================================

def main():
    """Simple command-line interface for downloading GGUF models"""
    import sys
    
    if len(sys.argv) < 2:
        print("\nSimple GGUF Model Downloader\n")
        print("Usage:")
        print("  python model_manager.py <model> <quantization>")
        print("  python model_manager.py <model> list              - List available files")
        print("  python model_manager.py cached                    - Show cached models")
        print("  python model_manager.py delete <filename>         - Delete a cached model")
        print("\nModel shortcuts (from model_config.py):")
        if MODEL_REPOS:
            for name in list(MODEL_REPOS.keys())[:5]:  # Show first 5
                print(f"  {name:20s} -> {MODEL_REPOS[name]}")
            print("  ... (see model_config.py for all shortcuts)")
        print("\nQuantization shortcuts:")
        if QUANT:
            for name, q in QUANT.items():
                print(f"  {name:20s} -> {q}")
        print("\nExamples:")
        print("  python model_manager.py LLAMA2_7B RECOMMENDED")
        print("  python model_manager.py LLAMA2_7B Q4_K_M")
        print("  python model_manager.py LLAMA2_7B list")
        print("  python model_manager.py cached")
        print("  python model_manager.py delete model.Q4_K_M.gguf")
        return
    
    # Handle commands that don't need a repo
    command = sys.argv[1].upper()
    
    if command == "CACHED":
        downloader = ModelDownloader()
        downloader.print_cached_models()
        return
    
    if command == "DELETE":
        if len(sys.argv) < 3:
            print("Error: Please specify which file to delete")
            print("Example: python model_manager.py delete model.Q4_K_M.gguf")
            return
        downloader = ModelDownloader()
        for filename in sys.argv[2:]:
            downloader.delete_model(filename)
        return
    
    # First arg is the model (repo or shortcut)
    repo_or_shortcut = sys.argv[1]
    
    if len(sys.argv) < 3:
        print("Error: Please specify what to download or 'list' to see available files")
        print(f"Example: python model_manager.py {repo_or_shortcut} list")
        return
    
    downloader = ModelDownloader(repo_or_shortcut)
    
    # Handle 'list' command
    if sys.argv[2].lower() == "list":
        downloader.print_available_files()
        return
    
    # Download models - resolve quantization shortcuts
    targets = sys.argv[2:]
    print(f"\nDownloading from {downloader.repo_id}")
    
    for target in targets:
        # Check if it's a quantization shortcut
        resolved_target = QUANT.get(target.upper(), target)
        if resolved_target != target:
            print(f"   Using {target} -> {resolved_target}")
        downloader.download(resolved_target)
    
    print("\nDone!")
    downloader.print_cached_models()


if __name__ == "__main__":
    main()
