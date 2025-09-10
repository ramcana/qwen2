#!/bin/bash

# Setup script for HuggingFace transfer accelerator
# Enables faster model downloads using Rust-based backend

echo "🚀 Setting up HuggingFace transfer accelerator..."

# Detect OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "🔧 Windows detected"
    echo "📦 Installing huggingface_hub and hf_transfer..."
    pip install -U huggingface_hub hf_transfer
    echo "📝 Setting environment variable..."
    setx HF_HUB_ENABLE_HF_TRANSFER 1
    echo "✅ Rust accelerator enabled! Restart your terminal to apply changes."
elif [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    # Linux or macOS
    echo "🔧 Linux/macOS detected"
    echo "📦 Installing huggingface_hub and hf_transfer..."
    pip install -U huggingface_hub hf_transfer
    echo "📝 Adding environment variable to shell profile..."
    
    # Add to shell profile
    if [ -f "$HOME/.bashrc" ]; then
        echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> "$HOME/.bashrc"
        echo "✅ Added to .bashrc"
    fi
    
    if [ -f "$HOME/.zshrc" ]; then
        echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> "$HOME/.zshrc"
        echo "✅ Added to .zshrc"
    fi
    
    # Export for current session
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "✅ Rust accelerator enabled for current session!"
    echo "💡 Run 'source ~/.bashrc' (or ~/.zshrc) to enable in new terminals"
else
    echo "⚠️ Unknown OS type: $OSTYPE"
    echo "📦 Installing huggingface_hub and hf_transfer..."
    pip install -U huggingface_hub hf_transfer
    echo "📝 Please set HF_HUB_ENABLE_HF_TRANSFER=1 in your environment"
fi

echo "🎉 Setup complete! Use the robust_download.py script for faster downloads."