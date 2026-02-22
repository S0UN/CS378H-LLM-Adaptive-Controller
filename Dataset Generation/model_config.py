"""
Global Model Configuration
Import this anywhere in your project to access model repos and settings
"""
import os

# MODEL REPOSITORY SHORTCUTS
MODEL_REPOS: dict[str, str] = {
    # Llama models
    "LLAMA2_7B": "TheBloke/Llama-2-7B-Chat-GGUF",
    "LLAMA2_13B": "TheBloke/Llama-2-13B-Chat-GGUF",
    "LLAMA3_8B": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    
    # Mistral models
    "MISTRAL_7B": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "MIXTRAL_8X7B": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    
    # Code models
    "CODELLAMA_7B": "TheBloke/CodeLlama-7B-Instruct-GGUF",
    "CODELLAMA_13B": "TheBloke/CodeLlama-13B-Instruct-GGUF",
    
    # Other popular models
    "PHI2": "TheBloke/phi-2-GGUF",
    "NEURAL_CHAT_7B": "TheBloke/neural-chat-7B-v3-1-GGUF",
}

# QUANTIZATION LEVELS
QUANT: dict[str, str] = {
    "TINY": "Q2_K",
    "SMALL": "Q3_K_M",
    "RECOMMENDED": "Q4_K_M",
    "HIGH": "Q5_K_M",
    "VERY_HIGH": "Q6_K",
    "EXCELLENT": "Q8_0",
    "FULL": "F16",
}

DEFAULT_MODEL = os.getenv("MODEL_REPO", MODEL_REPOS["LLAMA2_7B"])
DEFAULT_QUANT = os.getenv("MODEL_QUANT", QUANT["RECOMMENDED"])
DEFAULT_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "../models")

# HELPER FUNCTIONS
def get_model_repo(model_name: str) -> str:
    """
    Get HuggingFace repo ID from shortcut name
    
    Args:
        model_name: Either a shortcut (e.g., "LLAMA2_7B") or full repo ID
    
    Returns:
        Full HuggingFace repository ID
    """
    return MODEL_REPOS.get(model_name.upper(), model_name)


def get_quantization(quant_name: str) -> str:
    """
    Get quantization level from shortcut name
    
    Args:
        quant_name: Either a shortcut (e.g., "RECOMMENDED") or quantization level
    
    Returns:
        Quantization level (e.g., "Q4_K_M")
    """
    return QUANT.get(quant_name.upper(), quant_name)


# USAGE EXAMPLES (for documentation)
if __name__ == "__main__":
    print("Available Model Shortcuts:")
    for name, repo in MODEL_REPOS.items():
        print(f"   {name:20s} -> {repo}")
    
    print("\nAvailable Quantization Shortcuts:")
    for name, quant in QUANT.items():
        print(f"   {name:20s} -> {quant}")
    
    print("\nDefault Settings:")
    print(f"   MODEL_REPO:       {DEFAULT_MODEL}")
    print(f"   MODEL_QUANT:      {DEFAULT_QUANT}")
    print(f"   MODEL_CACHE_DIR:  {DEFAULT_CACHE_DIR}")
    
    print("\nUsage in your code:")
    print("   from model_config import MODEL_REPOS, QUANT")
    print("   from model_manager import ModelDownloader")
    print("   downloader = ModelDownloader(MODEL_REPOS['LLAMA2_7B'])")
    print("   downloader.download(QUANT['RECOMMENDED'])")
