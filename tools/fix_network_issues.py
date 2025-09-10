#!/usr/bin/env python3
"""
Network Issue Fixer for Qwen-Image Model Downloads
Diagnoses and attempts to fix common network issues that prevent model downloads
"""

import os
import subprocess
import sys
import time
from typing import List, Tuple


def check_dns_resolution(hostname: str) -> bool:
    """Check if a hostname can be resolved via DNS"""
    try:
        result = subprocess.run(
            ["nslookup", hostname], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def check_connectivity(host: str, port: int = 443) -> bool:
    """Check if we can connect to a host on a specific port"""
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def flush_dns_cache() -> bool:
    """Attempt to flush DNS cache"""
    try:
        # Try different methods based on system
        methods = [
            ["systemd-resolve", "--flush-caches"],
            ["systemctl", "restart", "systemd-resolved"],
        ]

        for method in methods:
            try:
                result = subprocess.run(
                    method, capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    print(f"✅ DNS cache flush attempted with: {' '.join(method)}")
                    return True
            except:
                continue

        print("⚠️ Could not flush DNS cache automatically (may need sudo)")
        return False
    except Exception as e:
        print(f"⚠️ Error flushing DNS cache: {e}")
        return False


def set_dns_servers() -> bool:
    """Provide guidance for setting DNS servers"""
    print("💡 To set DNS servers manually:")
    print("   1. Edit /etc/resolv.conf (may need sudo):")
    print("      sudo nano /etc/resolv.conf")
    print("   2. Add these lines:")
    print("      nameserver 8.8.8.8")
    print("      nameserver 8.8.4.4")
    print("      nameserver 1.1.1.1")
    return True


def install_hf_transfer() -> bool:
    """Install hf_transfer for better downloads"""
    try:
        print("📦 Installing hf_transfer for faster downloads...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "hf_transfer"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("✅ hf_transfer installed successfully")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("✅ hf_transfer enabled")
            return True
        else:
            print(f"❌ Failed to install hf_transfer: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing hf_transfer: {e}")
        return False


def test_endpoints() -> List[Tuple[str, bool]]:
    """Test connectivity to key endpoints"""
    endpoints = [
        ("huggingface.co", 443),
        ("transfer.xethub.hf.co", 443),
        ("cdn-lfs.huggingface.co", 443),
        ("github.com", 443),
    ]

    results = []
    for host, port in endpoints:
        is_connected = check_connectivity(host, port)
        status = "✅" if is_connected else "❌"
        print(f"{status} {host}:{port}")
        results.append((host, is_connected))

    return results


def diagnose_network_issues() -> dict:
    """Diagnose common network issues"""
    print("🔍 Diagnosing network issues...")

    issues = {"dns_resolution": {}, "connectivity": {}, "environment": {}}

    # Test DNS resolution
    test_hosts = ["huggingface.co", "transfer.xethub.hf.co", "google.com"]
    for host in test_hosts:
        can_resolve = check_dns_resolution(host)
        issues["dns_resolution"][host] = can_resolve
        status = "✅" if can_resolve else "❌"
        print(f"{status} DNS resolution for {host}")

    # Test connectivity
    endpoints = test_endpoints()
    issues["connectivity"] = dict(endpoints)

    # Check environment variables
    env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "HF_ENDPOINT", "HF_HUB_OFFLINE"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        issues["environment"][var] = value
        print(f"📝 {var}: {value}")

    return issues


def fix_common_issues() -> bool:
    """Attempt to fix common network issues"""
    print("🔧 Attempting to fix common network issues...")

    fixes_applied = []

    # Flush DNS cache
    if flush_dns_cache():
        fixes_applied.append("DNS cache flush attempted")

    # Set DNS servers (guidance only)
    if set_dns_servers():
        fixes_applied.append("DNS server guidance provided")

    # Install hf_transfer
    if install_hf_transfer():
        fixes_applied.append("hf_transfer installed")

    # Set environment variables
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HUB_OFFLINE"] = "0"
    fixes_applied.append("Environment variables set")

    if fixes_applied:
        print("✅ Applied fixes:")
        for fix in fixes_applied:
            print(f"   • {fix}")
        return True
    else:
        print("❌ No fixes could be applied automatically")
        return False


def main():
    """Main function to diagnose and fix network issues"""
    print("🌐 Qwen-Image Network Issue Fixer")
    print("=" * 50)

    # Diagnose issues
    issues = diagnose_network_issues()

    # Check if there are critical issues
    critical_dns_issues = sum(1 for v in issues["dns_resolution"].values() if not v)
    critical_connectivity_issues = sum(
        1 for v in issues["connectivity"].values() if not v
    )

    print(f"\n📊 Diagnosis Summary:")
    print(f"   DNS Issues: {critical_dns_issues}/{len(issues['dns_resolution'])}")
    print(
        f"   Connectivity Issues: {critical_connectivity_issues}/{len(issues['connectivity'])}"
    )

    if critical_dns_issues > 0 or critical_connectivity_issues > 0:
        print("\n🔧 Attempting to fix issues...")
        if fix_common_issues():
            print("\n✅ Fixes applied. Testing connectivity again...")
            time.sleep(5)  # Wait a bit for changes to take effect
            test_endpoints()
        else:
            print("\n❌ Could not apply fixes automatically")
            print("💡 Manual steps you can try:")
            print("   1. Restart your network services")
            print("   2. Check your firewall settings")
            print("   3. Try using a different network")
            print("   4. Manually download models from HuggingFace")
    else:
        print("\n✅ No critical network issues detected")
        print("💡 If you're still having download problems, try:")
        print("   1. Installing hf_transfer for better downloads")
        print("   2. Checking HuggingFace status page")
        print("   3. Using a VPN if there are regional restrictions")


if __name__ == "__main__":
    main()
