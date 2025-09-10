# Network Issue Resolution for Model Downloads

## Problem
DNS resolution failures when downloading models from HuggingFace, causing the download process to hang or fail.

## Error Details
```
WARN Reqwest(reqwest::Error { kind: Request, url: "https://transfer.xethub.hf.co/...", source: hyper_util::client::legacy::Error(Connect, ConnectError("dns error", Custom { kind: Uncategorized, error: "failed to lookup address information: Temporary failure in name resolution" })) })
```

## Root Causes
1. DNS server issues
2. Network connectivity problems
3. Firewall restrictions
4. Temporary HuggingFace service issues
5. WSL2 network configuration issues

## Solutions

### 1. Immediate Fixes
```bash
# Clear DNS cache
sudo systemd-resolve --flush-caches

# Or restart network services
sudo systemctl restart systemd-resolved

# Test DNS resolution
nslookup transfer.xethub.hf.co
```

### 2. WSL2 Network Configuration
If using WSL2, network issues are common. Try:

```bash
# Restart WSL2 networking
sudo /etc/init.d/ssh restart
sudo /etc/init.d/systemd-resolved restart

# Or restart WSL2 completely
wsl --shutdown
```

### 3. Use Alternative DNS Servers
Edit `/etc/resolv.conf`:
```
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 1.1.1.1
```

### 4. Environment Variables for Better Connectivity
```bash
export HTTP_PROXY=""
export HTTPS_PROXY=""
export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_OFFLINE=0
```

### 5. Use hf_transfer for Better Downloads
Install the Rust-based downloader for improved reliability:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### 6. Manual Model Download
If automated downloads continue to fail:
1. Download models manually from https://huggingface.co/Qwen/Qwen-Image
2. Place them in the appropriate cache directory
3. Set `local_files_only=True` in the model loading configuration

### 7. Retry Logic Enhancement
The system already has retry logic, but you can increase the retry count and delay:
```python
# In model loading configuration
max_retries = 5
retry_delay = 2  # seconds
```

## Testing Connectivity
```bash
# Test basic connectivity
ping -c 4 huggingface.co

# Test HTTPS connectivity
curl -I https://huggingface.co

# Test specific endpoint
curl -I https://transfer.xethub.hf.co
```

## Prevention
1. Set up proper DNS configuration
2. Use hf_transfer for faster, more reliable downloads
3. Monitor network connectivity before starting large downloads
4. Implement better error handling and user feedback
5. Consider downloading during off-peak hours when network congestion is lower
