"""
Unit tests for ModelDownloadManager with Qwen2-VL support
Tests download scenarios including interruption and resume capabilities
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import threading
import time

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_download_manager import (
    ModelDownloadManager,
    ModelDownloadConfig,
    DownloadProgress,
    download_qwen_image,
    download_qwen2_vl,
    check_model_status
)


class TestModelDownloadManager(unittest.TestCase):
    """Test cases for ModelDownloadManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelDownloadManager(cache_dir=self.temp_dir)
        
        # Mock HuggingFace API responses
        self.mock_repo_info = Mock()
        self.mock_repo_info.siblings = [
            Mock(rfilename="config.json", size=1024),
            Mock(rfilename="model.safetensors", size=1024*1024*1024),  # 1GB
            Mock(rfilename="tokenizer.json", size=2048),
            Mock(rfilename="README.md", size=512)
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelDownloadManager initialization"""
        manager = ModelDownloadManager()
        
        # Check default cache directory
        self.assertTrue(manager.cache_dir.endswith("huggingface/hub"))
        
        # Check supported models are loaded
        self.assertIn("Qwen/Qwen-Image", manager.supported_models)
        self.assertIn("Qwen/Qwen2-VL-7B-Instruct", manager.supported_models)
        self.assertIn("Qwen/Qwen2-VL-2B-Instruct", manager.supported_models)
        
        # Check model configurations
        qwen_image = manager.supported_models["Qwen/Qwen-Image"]
        self.assertEqual(qwen_image["type"], "text-to-image")
        self.assertEqual(qwen_image["architecture"], "MMDiT")
        
        qwen2_vl = manager.supported_models["Qwen/Qwen2-VL-7B-Instruct"]
        self.assertEqual(qwen2_vl["type"], "multimodal-language")
        self.assertIn("text_understanding", qwen2_vl["capabilities"])
    
    @patch('model_download_manager.repo_info')
    def test_check_model_availability_remote_only(self, mock_repo_info):
        """Test checking model availability when only remote exists"""
        mock_repo_info.return_value = self.mock_repo_info
        
        status = self.manager.check_model_availability("Qwen/Qwen-Image")
        
        self.assertTrue(status["remote_available"])
        self.assertFalse(status["local_available"])
        self.assertFalse(status["local_complete"])
        self.assertGreater(status["remote_size_bytes"], 0)
        self.assertEqual(status["local_size_bytes"], 0)
    
    @patch('model_download_manager.repo_info')
    def test_check_model_availability_local_complete(self, mock_repo_info):
        """Test checking model availability when local model is complete"""
        mock_repo_info.return_value = self.mock_repo_info
        
        # Create mock local model directory
        model_dir = Path(self.temp_dir) / "models" / "Qwen-Image"
        model_dir.mkdir(parents=True)
        
        # Create mock files
        for sibling in self.mock_repo_info.siblings:
            file_path = model_dir / sibling.rfilename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"x" * sibling.size)
        
        # Mock the _get_local_model_path method
        with patch.object(self.manager, '_get_local_model_path', return_value=model_dir):
            status = self.manager.check_model_availability("Qwen/Qwen-Image")
        
        self.assertTrue(status["remote_available"])
        self.assertTrue(status["local_available"])
        self.assertTrue(status["local_complete"])
        self.assertEqual(len(status["missing_files"]), 0)
    
    @patch('model_download_manager.repo_info')
    def test_check_model_availability_local_incomplete(self, mock_repo_info):
        """Test checking model availability when local model is incomplete"""
        mock_repo_info.return_value = self.mock_repo_info
        
        # Create mock local model directory with missing files
        model_dir = Path(self.temp_dir) / "models" / "Qwen-Image"
        model_dir.mkdir(parents=True)
        
        # Create only some files (missing model.safetensors)
        for sibling in self.mock_repo_info.siblings[:2]:  # Skip model.safetensors
            file_path = model_dir / sibling.rfilename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"x" * sibling.size)
        
        with patch.object(self.manager, '_get_local_model_path', return_value=model_dir):
            status = self.manager.check_model_availability("Qwen/Qwen-Image")
        
        self.assertTrue(status["remote_available"])
        self.assertTrue(status["local_available"])
        self.assertFalse(status["local_complete"])
        self.assertGreater(len(status["missing_files"]), 0)
    
    def test_model_download_config_defaults(self):
        """Test ModelDownloadConfig default values"""
        config = ModelDownloadConfig(model_name="test-model")
        
        self.assertEqual(config.model_name, "test-model")
        self.assertTrue(config.resume_download)
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.retry_attempts, 3)
        self.assertTrue(config.verify_integrity)
        self.assertTrue(config.cleanup_on_failure)
    
    @patch('model_download_manager.snapshot_download')
    @patch('model_download_manager.repo_info')
    def test_download_diffusion_model_success(self, mock_repo_info, mock_snapshot):
        """Test successful download of diffusion model"""
        mock_repo_info.return_value = self.mock_repo_info
        mock_snapshot.return_value = str(Path(self.temp_dir) / "Qwen-Image")
        
        # Mock check_model_availability to return incomplete status
        with patch.object(self.manager, 'check_model_availability') as mock_check:
            mock_check.return_value = {"local_complete": False, "error": None}
            
            with patch.object(self.manager, 'verify_model_integrity', return_value=True):
                config = ModelDownloadConfig(
                    model_name="Qwen/Qwen-Image",
                    verify_integrity=True
                )
                
                result = self.manager.download_model("Qwen/Qwen-Image", config)
        
        self.assertTrue(result)
        mock_snapshot.assert_called_once()
    
    @patch('model_download_manager.snapshot_download')
    @patch('model_download_manager.repo_info')
    def test_download_qwen2_vl_model_success(self, mock_repo_info, mock_snapshot):
        """Test successful download of Qwen2-VL model"""
        mock_repo_info.return_value = self.mock_repo_info
        mock_snapshot.return_value = str(Path(self.temp_dir) / "Qwen2-VL-7B-Instruct")
        
        with patch.object(self.manager, 'check_model_availability') as mock_check:
            mock_check.return_value = {"local_complete": False, "error": None}
            
            with patch.object(self.manager, 'verify_model_integrity', return_value=True):
                config = ModelDownloadConfig(
                    model_name="Qwen/Qwen2-VL-7B-Instruct",
                    verify_integrity=True
                )
                
                result = self.manager.download_model("Qwen/Qwen2-VL-7B-Instruct", config)
        
        self.assertTrue(result)
        mock_snapshot.assert_called_once()
    
    def test_download_unsupported_model(self):
        """Test download of unsupported model fails gracefully"""
        result = self.manager.download_model("unsupported/model")
        self.assertFalse(result)
    
    @patch('model_download_manager.snapshot_download')
    def test_download_with_interruption(self, mock_snapshot):
        """Test download interruption handling"""
        # Simulate interruption
        self.manager.download_interrupted = True
        
        with patch.object(self.manager, 'check_model_availability') as mock_check:
            mock_check.return_value = {"local_complete": False, "error": None}
            
            result = self.manager.download_model("Qwen/Qwen-Image")
        
        self.assertFalse(result)
    
    @patch('model_download_manager.snapshot_download')
    def test_download_with_network_error(self, mock_snapshot):
        """Test download with network error"""
        mock_snapshot.side_effect = Exception("Network error")
        
        with patch.object(self.manager, 'check_model_availability') as mock_check:
            mock_check.return_value = {"local_complete": False, "error": None}
            
            config = ModelDownloadConfig(
                model_name="Qwen/Qwen-Image",
                cleanup_on_failure=True
            )
            
            with patch.object(self.manager, '_cleanup_partial_download') as mock_cleanup:
                result = self.manager.download_model("Qwen/Qwen-Image", config)
        
        self.assertFalse(result)
        mock_cleanup.assert_called_once()
    
    def test_verify_model_integrity_diffusion_model(self):
        """Test model integrity verification for diffusion models"""
        # Create mock model directory structure
        model_dir = Path(self.temp_dir) / "Qwen-Image"
        model_dir.mkdir(parents=True)
        
        # Create essential files for diffusion model
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "unet/config.json",
            "vae/config.json"
        ]
        
        for file_path in essential_files:
            full_path = model_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("{}")
        
        # Create weight files
        (model_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        
        result = self.manager.verify_model_integrity("Qwen/Qwen-Image", str(model_dir))
        self.assertTrue(result)
    
    def test_verify_model_integrity_multimodal_model(self):
        """Test model integrity verification for multimodal models"""
        # Create mock model directory structure
        model_dir = Path(self.temp_dir) / "Qwen2-VL-7B-Instruct"
        model_dir.mkdir(parents=True)
        
        # Create essential files for multimodal model
        essential_files = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        for file_path in essential_files:
            full_path = model_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("{}")
        
        # Create weight files
        (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")
        
        result = self.manager.verify_model_integrity("Qwen/Qwen2-VL-7B-Instruct", str(model_dir))
        self.assertTrue(result)
    
    def test_verify_model_integrity_missing_files(self):
        """Test model integrity verification with missing files"""
        # Create mock model directory with missing files
        model_dir = Path(self.temp_dir) / "Qwen-Image"
        model_dir.mkdir(parents=True)
        
        # Create only some essential files (missing others)
        (model_dir / "model_index.json").write_text("{}")
        
        result = self.manager.verify_model_integrity("Qwen/Qwen-Image", str(model_dir))
        self.assertFalse(result)
    
    def test_download_qwen_image_convenience_function(self):
        """Test convenience function for downloading Qwen-Image"""
        with patch.object(self.manager, 'download_qwen_image', return_value=True) as mock_download:
            with patch('model_download_manager.ModelDownloadManager', return_value=self.manager):
                result = download_qwen_image()
        
        self.assertTrue(result)
    
    def test_download_qwen2_vl_convenience_function(self):
        """Test convenience function for downloading Qwen2-VL"""
        with patch.object(self.manager, 'download_qwen2_vl', return_value=True) as mock_download:
            with patch('model_download_manager.ModelDownloadManager', return_value=self.manager):
                result = download_qwen2_vl("7B")
        
        self.assertTrue(result)
    
    def test_download_qwen2_vl_invalid_size(self):
        """Test Qwen2-VL download with invalid model size"""
        result = self.manager.download_qwen2_vl("invalid_size")
        self.assertFalse(result)
    
    def test_cleanup_old_models(self):
        """Test cleanup of old models"""
        # Create mock model directories
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir(parents=True)
        
        # Create directories for different models
        old_model_dir = models_dir / "old-model"
        old_model_dir.mkdir()
        (old_model_dir / "large_file.bin").write_bytes(b"x" * 1024 * 1024)  # 1MB
        
        keep_model_dir = models_dir / "Qwen-Image"
        keep_model_dir.mkdir()
        (keep_model_dir / "keep_file.bin").write_bytes(b"x" * 1024)  # 1KB
        
        # Mock the models directory path
        with patch('model_download_manager.Path') as mock_path:
            mock_path.return_value = models_dir
            mock_path.side_effect = lambda x: Path(x) if isinstance(x, str) else x
            
            result = self.manager.cleanup_old_models(["Qwen/Qwen-Image"])
        
        # Check that old model was removed but kept model remains
        self.assertFalse(old_model_dir.exists())
        self.assertTrue(keep_model_dir.exists())
    
    def test_list_available_models(self):
        """Test listing available models with status"""
        with patch.object(self.manager, 'check_model_availability') as mock_check:
            mock_check.return_value = {
                "local_complete": True,
                "remote_available": True,
                "local_size_bytes": 1024*1024*1024
            }
            
            models = self.manager.list_available_models()
        
        self.assertIn("Qwen/Qwen-Image", models)
        self.assertIn("Qwen/Qwen2-VL-7B-Instruct", models)
        
        # Check that model info includes both config and status
        qwen_image = models["Qwen/Qwen-Image"]
        self.assertEqual(qwen_image["type"], "text-to-image")
        self.assertTrue(qwen_image["local_complete"])
    
    def test_progress_callback_system(self):
        """Test progress callback notification system"""
        callback_calls = []
        
        def test_callback(progress):
            callback_calls.append(progress)
        
        self.manager.add_progress_callback(test_callback)
        
        # Create test progress
        progress = DownloadProgress(total_files=10, completed_files=5)
        self.manager._notify_progress(progress)
        
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0].total_files, 10)
        self.assertEqual(callback_calls[0].completed_files, 5)
    
    def test_progress_callback_error_handling(self):
        """Test progress callback error handling"""
        def failing_callback(progress):
            raise Exception("Callback error")
        
        def working_callback(progress):
            self.callback_worked = True
        
        self.callback_worked = False
        self.manager.add_progress_callback(failing_callback)
        self.manager.add_progress_callback(working_callback)
        
        progress = DownloadProgress()
        self.manager._notify_progress(progress)
        
        # Working callback should still be called despite failing callback
        self.assertTrue(self.callback_worked)
    
    def test_format_size_utility(self):
        """Test size formatting utility function"""
        self.assertEqual(self.manager._format_size(0), "0 B")
        self.assertEqual(self.manager._format_size(1024), "1.0 KB")
        self.assertEqual(self.manager._format_size(1024*1024), "1.0 MB")
        self.assertEqual(self.manager._format_size(1024*1024*1024), "1.0 GB")
    
    @patch('model_download_manager.shutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk_usage):
        """Test disk space check with sufficient space"""
        # Mock 100GB free space
        mock_disk_usage.return_value = Mock(free=100 * 1024**3)
        
        result = self.manager.check_disk_space(50)  # Require 50GB
        self.assertTrue(result)
    
    @patch('model_download_manager.shutil.disk_usage')
    def test_check_disk_space_insufficient(self, mock_disk_usage):
        """Test disk space check with insufficient space"""
        # Mock 10GB free space
        mock_disk_usage.return_value = Mock(free=10 * 1024**3)
        
        result = self.manager.check_disk_space(50)  # Require 50GB
        self.assertFalse(result)
    
    def test_estimate_download_time(self):
        """Test download time estimation"""
        # Test with known model
        time_str = self.manager.estimate_download_time("Qwen/Qwen-Image", 100)  # 100 Mbps
        self.assertIn("minutes", time_str.lower())
        
        # Test with unknown model
        time_str = self.manager.estimate_download_time("unknown/model")
        self.assertEqual(time_str, "Unknown")
    
    def test_get_local_model_path(self):
        """Test getting local model path"""
        # Test with non-existent model
        path = self.manager._get_local_model_path("nonexistent/model")
        self.assertIsNone(path)
        
        # Test with existing model directory
        model_dir = Path(self.temp_dir) / "models" / "Qwen-Image"
        model_dir.mkdir(parents=True)
        
        with patch('model_download_manager.Path') as mock_path_class:
            mock_path_class.side_effect = lambda x: Path(x)
            # Mock the specific path check
            with patch.object(Path, 'exists', return_value=True):
                path = self.manager._get_local_model_path("Qwen/Qwen-Image")
                self.assertIsNotNone(path)


class TestDownloadInterruption(unittest.TestCase):
    """Test cases for download interruption and resume scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelDownloadManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_signal_handler_interruption(self):
        """Test signal handler for graceful interruption"""
        # Test SIGINT handling
        self.manager._signal_handler(2, None)  # SIGINT
        self.assertTrue(self.manager.download_interrupted)
        
        # Reset and test SIGTERM
        self.manager.download_interrupted = False
        self.manager._signal_handler(15, None)  # SIGTERM
        self.assertTrue(self.manager.download_interrupted)
    
    @patch('model_download_manager.snapshot_download')
    def test_resume_download_capability(self, mock_snapshot):
        """Test download resume capability"""
        # Configure to use resume
        config = ModelDownloadConfig(
            model_name="Qwen/Qwen-Image",
            resume_download=True
        )
        
        with patch.object(self.manager, 'check_model_availability') as mock_check:
            mock_check.return_value = {"local_complete": False, "error": None}
            
            with patch.object(self.manager, 'verify_model_integrity', return_value=True):
                self.manager.download_model("Qwen/Qwen-Image", config)
        
        # Verify resume_download was passed to snapshot_download
        call_args = mock_snapshot.call_args
        self.assertTrue(call_args.kwargs.get('resume_download', False))
    
    def test_cleanup_partial_download(self):
        """Test cleanup of partial downloads"""
        # Create mock partial download
        model_dir = Path(self.temp_dir) / "models" / "Qwen-Image"
        model_dir.mkdir(parents=True)
        (model_dir / "partial_file.bin").write_bytes(b"partial")
        
        with patch.object(self.manager, '_get_local_model_path', return_value=model_dir):
            self.manager._cleanup_partial_download("Qwen/Qwen-Image")
        
        self.assertFalse(model_dir.exists())


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelDownloadManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('model_download_manager.repo_info')
    def test_network_error_handling(self, mock_repo_info):
        """Test handling of network errors"""
        from huggingface_hub.utils import HfHubHTTPError
        mock_repo_info.side_effect = HfHubHTTPError("Network error")
        
        status = self.manager.check_model_availability("Qwen/Qwen-Image")
        
        self.assertFalse(status["remote_available"])
        self.assertIn("Remote check failed", status["error"])
    
    @patch('model_download_manager.repo_info')
    def test_repository_not_found_error(self, mock_repo_info):
        """Test handling of repository not found errors"""
        from huggingface_hub.utils import RepositoryNotFoundError
        mock_repo_info.side_effect = RepositoryNotFoundError("Repository not found")
        
        status = self.manager.check_model_availability("nonexistent/model")
        
        self.assertFalse(status["remote_available"])
        self.assertIn("Repository not found", status["error"])
    
    def test_permission_error_handling(self):
        """Test handling of permission errors during cleanup"""
        # Create a directory we can't delete (simulate permission error)
        model_dir = Path(self.temp_dir) / "models" / "protected-model"
        model_dir.mkdir(parents=True)
        
        with patch.object(self.manager, '_get_local_model_path', return_value=model_dir):
            with patch('model_download_manager.shutil.rmtree', side_effect=PermissionError("Permission denied")):
                # Should not raise exception, just log warning
                self.manager._cleanup_partial_download("protected/model")
    
    def test_disk_space_check_error_handling(self):
        """Test handling of disk space check errors"""
        with patch('model_download_manager.shutil.disk_usage', side_effect=OSError("Disk error")):
            # Should return True (assume OK) when check fails
            result = self.manager.check_disk_space(50)
            self.assertTrue(result)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)