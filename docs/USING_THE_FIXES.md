# Using the Network and Caching Fixes

## Overview
This document explains how to use the fixes that have been implemented to resolve model caching and network connectivity issues in the Qwen-Image system.

## Model Caching Fix

### Problem Solved
The system was repeatedly downloading models every time it started, even when the models were already present on the system.

### Solution
The [launch_complete_system.sh](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/scripts/launch_complete_system.sh) script now automatically sets the `QWEN_HOME` environment variable to ensure consistent model caching.

### How to Use
Simply run the launch script as usual:
```bash
./scripts/launch_complete_system.sh
```

The script will automatically:
1. Set `QWEN_HOME` to your project directory
2. Activate the virtual environment
3. Start both the backend API and React frontend

### Benefits
- Models load from cache instead of being redownloaded
- Significantly faster startup times after the initial download
- Reduced bandwidth usage
- Better disk space utilization

## Network Connectivity Fix

### Problem Solved
DNS resolution failures and connectivity issues when downloading models from HuggingFace.

### Solution
Installed and enabled `hf_transfer`, a Rust-based download tool that provides more reliable and faster downloads.

### How to Use
The network fixes are automatically applied when you run the system. However, if you encounter network issues, you can run the diagnostic tool:

```bash
python3 tools/fix_network_issues.py
```

This tool will:
1. Diagnose network connectivity issues
2. Install `hf_transfer` if not already installed
3. Set appropriate environment variables
4. Provide guidance for manual fixes if needed

### Environment Variables Set
The following environment variables are set automatically:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_OFFLINE=0
```

## Testing the Fixes

### Automated Testing
Run the test script to verify that all fixes are working:
```bash
python3 test_fixes.py
```

This will check:
- QWEN_HOME environment variable
- Model cache directory
- hf_transfer installation and enablement
- Network connectivity to HuggingFace endpoints
- Environment variables

### Manual Verification
1. **First Run**: The system will download models (this may take 10-20 minutes)
2. **Subsequent Runs**: Models should load from cache much faster
3. **Network Status**: Check that there are no DNS resolution errors in the logs

## Troubleshooting

### If Models Still Redownload
1. Verify that `QWEN_HOME` is set in the launch script output
2. Check that the cache directory exists: `$QWEN_HOME/models/qwen-image`
3. Ensure the launch script is being run from the project root

### If Network Issues Persist
1. Run the network diagnostic tool: `python3 tools/fix_network_issues.py`
2. Check your DNS settings
3. Verify firewall settings
4. Try using a different network connection
5. Consider using a VPN if there are regional restrictions

### Manual hf_transfer Installation
If the automatic installation fails:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Files Created

1. **[scripts/launch_complete_system.sh](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/scripts/launch_complete_system.sh)** - Updated launch script with QWEN_HOME setting
2. **[tools/fix_network_issues.py](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/tools/fix_network_issues.py)** - Network diagnostic and fix tool
3. **[test_fixes.py](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/test_fixes.py)** - Automated test script
4. **[MODEL_CACHING_FIX.md](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/MODEL_CACHING_FIX.md)** - Documentation of the caching fix
5. **[NETWORK_ISSUE_RESOLUTION.md](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/NETWORK_ISSUE_RESOLUTION.md)** - Comprehensive network issue resolution guide
6. **[NETWORK_AND_CACHING_FIXES.md](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/NETWORK_AND_CACHING_FIXES.md)** - Summary of all fixes
7. **[USING_THE_FIXES.md](file:///wsl.localhost/Ubuntu/home/ramji_t/projects/Qwen2/USING_THE_FIXES.md)** - This document

## Best Practices

1. **Always use the launch script** rather than starting components manually
2. **Run the test script periodically** to ensure fixes are still working
3. **Monitor network connectivity** if downloads fail
4. **Keep hf_transfer updated** for the best performance
5. **Check disk space** before downloading large models
6. **Use the diagnostic tool** at the first sign of network issues

## Support

If you continue to experience issues:
1. Check the log files in the `logs/` directory
2. Run the diagnostic tools
3. Review the documentation files created
4. Contact the development team with detailed error messages