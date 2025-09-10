# WSL2 Browser Integration Guide

## Option A: Install wslu (WSL Utilities)

```bash
sudo apt update
sudo apt install wslu
```

Then you can use `wslview http://localhost:7860` to open URLs in Windows browser.

## Option B: Create Browser Alias

Add to your ~/.bashrc:

```bash
alias chrome="/mnt/c/Program\ Files/Google/Chrome/Application/chrome.exe"
alias edge="/mnt/c/Program\ Files\ \(x86\)/Microsoft/Edge/Application/msedge.exe"
```

## Option C: Set Default Browser

```bash
export BROWSER="/mnt/c/Program\ Files/Google/Chrome/Application/chrome.exe"
```

## Current Solution

The UI now runs with `inbrowser=False` to avoid permission errors.
Simply open <http://localhost:7860> in your Windows browser manually.

## Usage Commands

- Launch UI: `./launch_ui.sh`
- Direct launch: `source venv/bin/activate && python src/qwen_image_ui.py`
- Access URL: <http://localhost:7860> (in Windows browser)
