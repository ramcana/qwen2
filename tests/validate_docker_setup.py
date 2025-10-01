#!/usr/bin/env python3
"""
Docker Setup Validation Script
Validates Docker environment and prerequisites for running tests.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import shutil

class DockerSetupValidator:
    """Validates Docker setup and environment for testing"""
    
    def __init__(self):
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        print("Docker Setup Validation")
        print("=" * 50)
        
        # Run validation checks
        self.validate_docker_installation()
        self.validate_docker_compose()
        self.validate_system_resources()
        self.validate_project_structure()
        self.validate_docker_files()
        self.validate_network_requirements()
        self.validate_permissions()
        
        # Generate summary
        return self.generate_validation_report()
    
    def validate_docker_installation(self):
        """Validate Docker installation and daemon"""
        print("\n1. Docker Installation")
        print("-" * 30)
        
        # Check Docker command
        docker_version = self._check_command_version("docker", ["--version"])
        if docker_version:
            print(f"✅ Docker installed: {docker_version}")
            self.validation_results['docker_installed'] = True
        else:
            print("❌ Docker not found or not working")
            self.validation_results['docker_installed'] = False
            self.errors.append("Docker is not installed or not in PATH")
            return
        
        # Check Docker daemon
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                self.validation_results['docker_daemon_running'] = True
                
                # Parse Docker info
                info_lines = result.stdout.split('\n')
                for line in info_lines:
                    if 'Server Version:' in line:
                        server_version = line.split(':')[1].strip()
                        print(f"   Server Version: {server_version}")
                    elif 'Total Memory:' in line:
                        memory = line.split(':')[1].strip()
                        print(f"   Total Memory: {memory}")
                    elif 'CPUs:' in line:
                        cpus = line.split(':')[1].strip()
                        print(f"   CPUs: {cpus}")
            else:
                print("❌ Docker daemon not running or not accessible")
                self.validation_results['docker_daemon_running'] = False
                self.errors.append("Docker daemon is not running")
                
        except subprocess.TimeoutExpired:
            print("❌ Docker daemon check timed out")
            self.validation_results['docker_daemon_running'] = False
            self.errors.append("Docker daemon check timed out")
        except Exception as e:
            print(f"❌ Error checking Docker daemon: {e}")
            self.validation_results['docker_daemon_running'] = False
            self.errors.append(f"Error checking Docker daemon: {e}")
    
    def validate_docker_compose(self):
        """Validate Docker Compose installation"""
        print("\n2. Docker Compose")
        print("-" * 30)
        
        # Check Docker Compose command
        compose_version = self._check_command_version("docker-compose", ["--version"])
        if compose_version:
            print(f"✅ Docker Compose installed: {compose_version}")
            self.validation_results['docker_compose_installed'] = True
        else:
            # Try docker compose (newer syntax)
            compose_version = self._check_command_version("docker", ["compose", "version"])
            if compose_version:
                print(f"✅ Docker Compose (plugin) installed: {compose_version}")
                self.validation_results['docker_compose_installed'] = True
            else:
                print("❌ Docker Compose not found")
                self.validation_results['docker_compose_installed'] = False
                self.errors.append("Docker Compose is not installed")
    
    def validate_system_resources(self):
        """Validate system resources"""
        print("\n3. System Resources")
        print("-" * 30)
        
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            print(f"   Total Memory: {memory_gb:.1f} GB")
            
            if memory_gb >= 8:
                print("✅ Sufficient memory (8+ GB)")
                self.validation_results['sufficient_memory'] = True
            elif memory_gb >= 4:
                print("⚠️  Limited memory (4-8 GB) - some tests may be slow")
                self.validation_results['sufficient_memory'] = True
                self.warnings.append("Limited memory - performance tests may be affected")
            else:
                print("❌ Insufficient memory (<4 GB)")
                self.validation_results['sufficient_memory'] = False
                self.errors.append("Insufficient memory for Docker tests")
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            print(f"   CPU Cores: {cpu_count}")
            
            if cpu_count >= 4:
                print("✅ Sufficient CPU cores (4+)")
                self.validation_results['sufficient_cpu'] = True
            elif cpu_count >= 2:
                print("⚠️  Limited CPU cores (2-3) - tests may be slower")
                self.validation_results['sufficient_cpu'] = True
                self.warnings.append("Limited CPU cores - tests may take longer")
            else:
                print("❌ Insufficient CPU cores (<2)")
                self.validation_results['sufficient_cpu'] = False
                self.errors.append("Insufficient CPU cores for Docker tests")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            print(f"   Free Disk Space: {disk_free_gb:.1f} GB")
            
            if disk_free_gb >= 10:
                print("✅ Sufficient disk space (10+ GB)")
                self.validation_results['sufficient_disk'] = True
            elif disk_free_gb >= 5:
                print("⚠️  Limited disk space (5-10 GB)")
                self.validation_results['sufficient_disk'] = True
                self.warnings.append("Limited disk space - may affect image builds")
            else:
                print("❌ Insufficient disk space (<5 GB)")
                self.validation_results['sufficient_disk'] = False
                self.errors.append("Insufficient disk space for Docker tests")
                
        except ImportError:
            print("⚠️  psutil not available - cannot check system resources")
            self.warnings.append("Cannot validate system resources - psutil not installed")
        except Exception as e:
            print(f"❌ Error checking system resources: {e}")
            self.warnings.append(f"Error checking system resources: {e}")
    
    def validate_project_structure(self):
        """Validate project structure and required files"""
        print("\n4. Project Structure")
        print("-" * 30)
        
        project_root = Path(__file__).parent.parent
        
        required_files = [
            "Dockerfile.api",
            "frontend/Dockerfile",
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "docker-compose.prod.yml",
            "src/api_server.py",
            "requirements.txt"
        ]
        
        required_dirs = [
            "src",
            "frontend",
            "tests",
            "config",
            "scripts"
        ]
        
        # Check required files
        missing_files = []
        for file_path in required_files:
            full_path = project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} (missing)")
                missing_files.append(file_path)
        
        # Check required directories
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                print(f"✅ {dir_path}/")
            else:
                print(f"❌ {dir_path}/ (missing)")
                missing_dirs.append(dir_path)
        
        self.validation_results['project_structure_valid'] = len(missing_files) == 0 and len(missing_dirs) == 0
        
        if missing_files:
            self.errors.append(f"Missing required files: {', '.join(missing_files)}")
        if missing_dirs:
            self.errors.append(f"Missing required directories: {', '.join(missing_dirs)}")
    
    def validate_docker_files(self):
        """Validate Docker files syntax and content"""
        print("\n5. Docker Files")
        print("-" * 30)
        
        project_root = Path(__file__).parent.parent
        
        # Validate Dockerfiles
        dockerfiles = [
            ("API Dockerfile", "Dockerfile.api"),
            ("Frontend Dockerfile", "frontend/Dockerfile")
        ]
        
        for name, dockerfile_path in dockerfiles:
            full_path = project_root / dockerfile_path
            
            if not full_path.exists():
                print(f"❌ {name} not found")
                continue
            
            try:
                # Basic syntax check by trying to parse with Docker
                result = subprocess.run(
                    ["docker", "build", "--dry-run", "-f", str(full_path), str(project_root)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"✅ {name} syntax valid")
                else:
                    print(f"❌ {name} syntax error")
                    self.errors.append(f"{name} has syntax errors")
                    
            except subprocess.TimeoutExpired:
                print(f"⚠️  {name} validation timed out")
                self.warnings.append(f"{name} validation timed out")
            except Exception as e:
                print(f"⚠️  Could not validate {name}: {e}")
                self.warnings.append(f"Could not validate {name}: {e}")
        
        # Validate Docker Compose files
        compose_files = [
            ("Main Compose", "docker-compose.yml"),
            ("Dev Compose", "docker-compose.dev.yml"),
            ("Prod Compose", "docker-compose.prod.yml")
        ]
        
        for name, compose_path in compose_files:
            full_path = project_root / compose_path
            
            if not full_path.exists():
                print(f"❌ {name} not found")
                continue
            
            try:
                # Validate compose file syntax
                result = subprocess.run(
                    ["docker-compose", "-f", str(full_path), "config"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=project_root
                )
                
                if result.returncode == 0:
                    print(f"✅ {name} syntax valid")
                else:
                    print(f"❌ {name} syntax error")
                    self.errors.append(f"{name} has syntax errors")
                    
            except subprocess.TimeoutExpired:
                print(f"⚠️  {name} validation timed out")
                self.warnings.append(f"{name} validation timed out")
            except Exception as e:
                print(f"⚠️  Could not validate {name}: {e}")
                self.warnings.append(f"Could not validate {name}: {e}")
    
    def validate_network_requirements(self):
        """Validate network requirements"""
        print("\n6. Network Requirements")
        print("-" * 30)
        
        # Check if required ports are available
        required_ports = [80, 443, 8080, 8000, 3000]
        
        try:
            import socket
            
            for port in required_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result != 0:
                    print(f"✅ Port {port} available")
                else:
                    print(f"⚠️  Port {port} in use")
                    self.warnings.append(f"Port {port} is in use - may cause conflicts")
            
            self.validation_results['network_ports_available'] = True
            
        except ImportError:
            print("⚠️  Cannot check port availability - socket module not available")
            self.warnings.append("Cannot check port availability")
        except Exception as e:
            print(f"❌ Error checking network requirements: {e}")
            self.warnings.append(f"Error checking network requirements: {e}")
        
        # Check external network creation
        try:
            result = subprocess.run(
                ["docker", "network", "create", "--driver", "bridge", "test-network-validation"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✅ Can create Docker networks")
                self.validation_results['can_create_networks'] = True
                
                # Clean up test network
                subprocess.run(
                    ["docker", "network", "rm", "test-network-validation"],
                    capture_output=True,
                    timeout=10
                )
            else:
                print("❌ Cannot create Docker networks")
                self.validation_results['can_create_networks'] = False
                self.errors.append("Cannot create Docker networks")
                
        except Exception as e:
            print(f"❌ Error testing network creation: {e}")
            self.validation_results['can_create_networks'] = False
            self.errors.append(f"Error testing network creation: {e}")
    
    def validate_permissions(self):
        """Validate file permissions and Docker access"""
        print("\n7. Permissions")
        print("-" * 30)
        
        # Check Docker socket access
        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✅ Docker socket accessible")
                self.validation_results['docker_socket_accessible'] = True
            else:
                print("❌ Docker socket not accessible")
                self.validation_results['docker_socket_accessible'] = False
                self.errors.append("Docker socket not accessible - check permissions")
                
        except Exception as e:
            print(f"❌ Error checking Docker socket: {e}")
            self.validation_results['docker_socket_accessible'] = False
            self.errors.append(f"Error checking Docker socket: {e}")
        
        # Check write permissions in project directory
        project_root = Path(__file__).parent.parent
        test_dirs = ['logs', 'cache', 'generated_images', 'uploads']
        
        for test_dir in test_dirs:
            dir_path = project_root / test_dir
            
            try:
                dir_path.mkdir(exist_ok=True)
                test_file = dir_path / 'test_write_permission.tmp'
                test_file.write_text('test')
                test_file.unlink()
                print(f"✅ Write access to {test_dir}/")
            except Exception as e:
                print(f"❌ No write access to {test_dir}/: {e}")
                self.errors.append(f"No write access to {test_dir}/")
    
    def _check_command_version(self, command: str, args: List[str]) -> Optional[str]:
        """Check if command exists and get version"""
        try:
            result = subprocess.run(
                [command] + args,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return None
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        # Count results
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for v in self.validation_results.values() if v)
        failed_checks = total_checks - passed_checks
        
        success_rate = passed_checks / total_checks if total_checks > 0 else 0
        overall_success = len(self.errors) == 0
        
        print(f"\nChecks: {passed_checks}/{total_checks} passed ({success_rate:.1%})")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if overall_success:
            print("\n✅ VALIDATION PASSED - Ready for Docker testing!")
        else:
            print("\n❌ VALIDATION FAILED - Issues must be resolved")
        
        # Print errors
        if self.errors:
            print(f"\nERRORS TO FIX:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        # Print warnings
        if self.warnings:
            print(f"\nWARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return {
            'overall_success': overall_success,
            'success_rate': success_rate,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': recommendations,
            'validation_results': self.validation_results,
            'timestamp': time.time()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not self.validation_results.get('docker_installed', True):
            recommendations.append("Install Docker from https://docs.docker.com/get-docker/")
        
        if not self.validation_results.get('docker_compose_installed', True):
            recommendations.append("Install Docker Compose from https://docs.docker.com/compose/install/")
        
        if not self.validation_results.get('docker_daemon_running', True):
            recommendations.append("Start Docker daemon (sudo systemctl start docker)")
        
        if not self.validation_results.get('sufficient_memory', True):
            recommendations.append("Increase system memory to at least 4GB for basic tests, 8GB+ recommended")
        
        if not self.validation_results.get('sufficient_disk', True):
            recommendations.append("Free up disk space - at least 5GB required, 10GB+ recommended")
        
        if not self.validation_results.get('docker_socket_accessible', True):
            recommendations.append("Add user to docker group: sudo usermod -aG docker $USER")
        
        if len(self.warnings) > 3:
            recommendations.append("Review warnings above - they may affect test performance")
        
        return recommendations


def main():
    """Main entry point"""
    validator = DockerSetupValidator()
    
    try:
        report = validator.validate_all()
        
        # Save report
        report_file = Path(__file__).parent / "docker_setup_validation.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nValidation report saved: {report_file}")
        
        # Exit with appropriate code
        if report['overall_success']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()