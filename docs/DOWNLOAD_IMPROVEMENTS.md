# Qwen Model Download Improvements

This document describes the improvements made to the model download process for better reliability and performance.

## Fast Wins Implemented

### 1. Rust Accelerator Support

We've added support for the HuggingFace Rust-based transfer accelerator (`hf_transfer`) which provides dramatically more robust downloads for large files.

**Setup:**
```bash
# Automatic setup
./scripts/setup_hf_transfer.sh

# Manual setup
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1  # Linux/macOS
# or
setx HF_HUB_ENABLE_HF_TRANSFER 1    # Windows
```

### 2. Resume Download Capability

All download scripts now support resuming interrupted downloads automatically:

```bash
# Using the enhanced downloader
python tools/download_qwen_edit_hub.py --resume

# Using the robust downloader
python tools/robust_download.py Qwen/Qwen2.5-VL-7B-Instruct --out-dir ./models/qwen-vl
```

### 3. Optimized Cache Directory

Cache directories can now be configured for faster SSD usage:

```bash
# Set cache directory to fast SSD
export HF_HOME=/path/to/fast/ssd/.cache/huggingface  # Linux/macOS
# or
setx HF_HOME E:\HFCache  # Windows
```

### 4. Cleanup of Incomplete Downloads

Automatic cleanup of incomplete downloads prevents corruption:

```bash
# Manual cleanup
python tools/download_qwen_edit_hub.py --cleanup
```

### 5. Improved Concurrency Settings

Download scripts now use optimized concurrency settings:
- 8 workers for standard downloads
- 4 workers for memory-constrained environments
- Automatic reduction on retry attempts

### 6. Windows Compatibility

Added `local_dir_use_symlinks=False` for better Windows compatibility.

### 7. Retry Mechanism with Backoff

All download scripts now include automatic retry with exponential backoff:
- Default: 3 retry attempts
- Configurable with `--retries` parameter
- Automatic cleanup between retries

## Usage Examples

### Robust Download Script

```bash
# Download with all improvements enabled
python tools/robust_download.py Qwen/Qwen2.5-VL-7B-Instruct \
  --out-dir ./models/qwen-vl \
  --retries 3 \
  --workers 8
```

### Enhanced Hub Downloader

```bash
# Download with resume capability and optimized settings
python tools/download_qwen_edit_hub.py \
  --cache-dir ./models/qwen-image-edit \
  --resume \
  --retries 3 \
  --max-workers 8
```

### Memory-Optimized Downloader

```bash
# Download with memory constraints and retry capability
python tools/download_qwen_edit_memory_safe.py \
  --retries 3
```

## Performance Tips

1. **Use Fast SSD**: Set `HF_HOME` to a fast NVMe SSD location
2. **Exclude from Antivirus**: Add your model directories to Windows Defender exclusions
3. **Avoid WSL/NTFS Boundary**: In WSL, download to Linux ext4 filesystem, not `/mnt/c/`
4. **Clean Up Partial Downloads**: Use `--cleanup` flag if downloads stall
5. **Lower Concurrency for Unstable Connections**: Use `--max-workers 2` for unstable connections
6. **Git LFS Alternative**: For problematic regions, try:
   ```bash
   git lfs install
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
   ```

## Troubleshooting

### If Downloads Stall at 93-96%
1. Find and delete the incomplete shard file
2. Resume the download

### If Symlinks Fail on Windows
Ensure `local_dir_use_symlinks=False` is set (already configured in our scripts)

### If Memory Issues Occur
Use the memory-safe downloader with reduced concurrency

### If Network is Unstable
Reduce worker count and increase retry attempts
