#!/usr/bin/env python3
"""
Automated Error Detection System for Qwen-Image Project
Detects common issues, performance problems, and potential bugs
Optimized for RTX 4080 + AMD Threadripper environment
"""

import ast
import importlib.util
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ErrorDetector:
    """Comprehensive error detection for the Qwen-Image project"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.info: List[Dict[str, Any]] = []

    def add_error(
        self, category: str, message: str, file_path: str = "", line: int = 0
    ):
        """Add an error to the detection results"""
        self.errors.append(
            {
                "category": category,
                "message": message,
                "file": file_path,
                "line": line,
                "severity": "error",
            }
        )

    def add_warning(
        self, category: str, message: str, file_path: str = "", line: int = 0
    ):
        """Add a warning to the detection results"""
        self.warnings.append(
            {
                "category": category,
                "message": message,
                "file": file_path,
                "line": line,
                "severity": "warning",
            }
        )

    def add_info(self, category: str, message: str, file_path: str = "", line: int = 0):
        """Add an info message to the detection results"""
        self.info.append(
            {
                "category": category,
                "message": message,
                "file": file_path,
                "line": line,
                "severity": "info",
            }
        )

    def check_python_syntax(self) -> None:
        """Check for Python syntax errors"""
        logger.info("🔍 Checking Python syntax...")

        python_files = list(self.project_root.rglob("*.py"))
        python_files = [
            f
            for f in python_files
            if not any(
                skip in str(f)
                for skip in ["venv", "__pycache__", ".git", "build", "dist"]
            )
        ]

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                self.add_error(
                    "syntax", f"Syntax error: {e.msg}", str(py_file), e.lineno or 0
                )
            except Exception as e:
                self.add_error("syntax", f"Parse error: {str(e)}", str(py_file))

    def check_imports(self) -> None:
        """Check for import issues"""
        logger.info("📦 Checking imports...")

        # Add project root to path for import testing
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Critical imports to test
        critical_imports = [
            ("torch", "PyTorch not available - required for model inference"),
            ("gradio", "Gradio not available - required for web interface"),
            ("diffusers", "Diffusers not available - required for model loading"),
            ("transformers", "Transformers not available - required for tokenization"),
            ("PIL", "Pillow not available - required for image processing"),
        ]

        for module, error_msg in critical_imports:
            try:
                importlib.import_module(module)
                self.add_info("imports", f"✅ {module} available")
            except ImportError:
                self.add_error("imports", error_msg)

        # Project-specific imports
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from qwen_generator import QwenImageGenerator
            from qwen_image_config import MODEL_CONFIG

            self.add_info("imports", "✅ Core project modules importable")
        except ImportError as e:
            self.add_error("imports", f"Core module import failed: {str(e)}")

    def check_hardware_requirements(self) -> None:
        """Check hardware and CUDA availability"""
        logger.info("🖥️ Checking hardware requirements...")

        try:
            import torch

            # CUDA availability
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

                self.add_info("hardware", f"✅ CUDA available - {gpu_name}")
                self.add_info("hardware", f"✅ VRAM: {vram_gb:.1f}GB")

                # Check if RTX 4080 optimizations can be used
                if "4080" in gpu_name or vram_gb >= 15:
                    self.add_info("hardware", "✅ Hardware optimal for Qwen-Image")
                elif vram_gb < 12:
                    self.add_warning("hardware", "⚠️ Low VRAM - may need CPU offload")

                # Check CUDA version compatibility
                cuda_version = torch.version.cuda
                if cuda_version:
                    self.add_info("hardware", f"✅ CUDA version: {cuda_version}")
                else:
                    self.add_warning("hardware", "⚠️ CUDA version not detected")

            else:
                self.add_warning(
                    "hardware", "⚠️ CUDA not available - will use CPU (slow)"
                )

        except ImportError:
            self.add_error(
                "hardware", "❌ PyTorch not available - cannot check hardware"
            )

    def check_file_structure(self) -> None:
        """Validate project file structure"""
        logger.info("📁 Checking file structure...")

        required_files = [
            "src/qwen_image_ui.py",
            "src/qwen_generator.py",
            "src/qwen_image_config.py",
            "requirements.txt",
            "README.md",
            "launch.py",
        ]

        required_dirs = ["src", "generated_images", "configs", "scripts", "docs"]

        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                self.add_error("structure", f"Missing required file: {file_path}")
            else:
                self.add_info("structure", f"✅ Found: {file_path}")

        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                self.add_error("structure", f"Missing required directory: {dir_path}")
            else:
                self.add_info("structure", f"✅ Found: {dir_path}/")

    def check_configuration_files(self) -> None:
        """Validate configuration files"""
        logger.info("⚙️ Checking configuration files...")

        config_files = [
            ("pyproject.toml", "TOML configuration"),
            (".flake8", "Flake8 linting config"),
            ("mypy.ini", "MyPy type checking config"),
        ]

        for config_file, description in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                self.add_info("config", f"✅ {description} found")
                try:
                    # Basic validation - try to read the file
                    with open(config_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if len(content.strip()) == 0:
                            self.add_warning("config", f"⚠️ {config_file} is empty")
                except Exception as e:
                    self.add_error("config", f"❌ Cannot read {config_file}: {str(e)}")
            else:
                self.add_warning("config", f"⚠️ {description} missing: {config_file}")

    def check_dependencies(self) -> None:
        """Check if all dependencies are properly installed"""
        logger.info("📋 Checking dependencies...")

        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.add_error("dependencies", "requirements.txt not found")
            return

        try:
            with open(requirements_file, "r", encoding="utf-8") as f:
                requirements = f.read().splitlines()

            # Filter out comments and empty lines
            packages = [
                line.strip()
                for line in requirements
                if line.strip() and not line.strip().startswith("#")
            ]

            missing_packages = []
            for package in packages:
                # Extract package name (handle version specifiers)
                pkg_name = package.split(">=")[0].split("==")[0].split("[")[0].strip()
                if pkg_name.startswith("git+"):
                    continue  # Skip git packages for now

                try:
                    importlib.import_module(pkg_name.replace("-", "_"))
                except ImportError:
                    missing_packages.append(pkg_name)

            if missing_packages:
                self.add_warning(
                    "dependencies",
                    f"⚠️ Missing packages: {', '.join(missing_packages)}",
                )
            else:
                self.add_info(
                    "dependencies", "✅ All core dependencies appear to be installed"
                )

        except Exception as e:
            self.add_error("dependencies", f"Error checking requirements.txt: {str(e)}")

    def check_security_issues(self) -> None:
        """Basic security checks"""
        logger.info("🔒 Checking for security issues...")

        python_files = list(self.project_root.rglob("*.py"))
        python_files = [
            f
            for f in python_files
            if not any(skip in str(f) for skip in ["venv", "__pycache__", ".git"])
        ]

        security_patterns = [
            ("eval(", "Use of eval() can be dangerous"),
            ("exec(", "Use of exec() can be dangerous"),
            ("subprocess.call(", "Use subprocess.run() instead of subprocess.call()"),
            ("shell=True", "Avoid shell=True in subprocess calls"),
            ("pickle.load", "Pickle can execute arbitrary code"),
            ("yaml.load(", "Use yaml.safe_load() instead of yaml.load()"),
        ]

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                for pattern, message in security_patterns:
                    if pattern in content:
                        line_num = content[: content.find(pattern)].count("\n") + 1
                        self.add_warning("security", message, str(py_file), line_num)

            except Exception as e:
                self.add_warning("security", f"Could not scan {py_file}: {str(e)}")

    def check_model_files(self) -> None:
        """Check for potential issues with model files"""
        logger.info("🤖 Checking model-related files...")

        model_extensions = [".pt", ".pth", ".bin", ".safetensors"]
        large_files = []

        for ext in model_extensions:
            for model_file in self.project_root.rglob(f"*{ext}"):
                try:
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    if size_mb > 100:  # Files larger than 100MB
                        large_files.append((str(model_file), size_mb))
                except Exception:
                    continue

        if large_files:
            for file_path, size_mb in large_files:
                self.add_warning(
                    "models",
                    f"Large model file detected: {file_path} ({size_mb:.1f}MB)",
                )

        # Check if models directory exists and has proper structure
        models_dir = self.project_root / "models"
        if models_dir.exists():
            self.add_info("models", "✅ Models directory exists")
        else:
            self.add_info(
                "models", "ℹ️ Models directory not found (will be created on first run)"
            )

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all error detection checks"""
        logger.info("🚀 Starting comprehensive error detection...")

        # Run all checks
        self.check_python_syntax()
        self.check_imports()
        self.check_hardware_requirements()
        self.check_file_structure()
        self.check_configuration_files()
        self.check_dependencies()
        self.check_security_issues()
        self.check_model_files()

        # Compile results
        results = {
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "total_info": len(self.info),
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }

        return results

    def generate_report(
        self, results: Dict[str, Any], output_file: str = "error_detection_report.json"
    ) -> None:
        """Generate a detailed report"""

        # Save JSON report
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary to console
        print("\n" + "=" * 60)
        print("🔍 ERROR DETECTION REPORT")
        print("=" * 60)

        summary = results["summary"]
        print(f"📊 Summary:")
        print(f"   • Errors: {summary['total_errors']}")
        print(f"   • Warnings: {summary['total_warnings']}")
        print(f"   • Info: {summary['total_info']}")
        print()

        # Print errors
        if results["errors"]:
            print("❌ ERRORS:")
            for error in results["errors"]:
                file_info = (
                    f" ({error['file']}:{error['line']})" if error["file"] else ""
                )
                print(f"   • [{error['category']}] {error['message']}{file_info}")
            print()

        # Print warnings
        if results["warnings"]:
            print("⚠️  WARNINGS:")
            for warning in results["warnings"]:
                file_info = (
                    f" ({warning['file']}:{warning['line']})" if warning["file"] else ""
                )
                print(f"   • [{warning['category']}] {warning['message']}{file_info}")
            print()

        # Overall status
        if summary["total_errors"] == 0:
            print("✅ No critical errors found!")
            if summary["total_warnings"] == 0:
                print("🎉 Project appears to be in excellent condition!")
            else:
                print(f"⚠️  {summary['total_warnings']} warning(s) to review")
        else:
            print(f"❌ {summary['total_errors']} error(s) need immediate attention")

        print(f"\n📄 Detailed report saved to: {output_file}")
        print("=" * 60)


def main():
    """Main function"""
    detector = ErrorDetector()
    results = detector.run_all_checks()
    detector.generate_report(results)

    # Exit with error code if critical issues found
    return 1 if results["summary"]["total_errors"] > 0 else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
