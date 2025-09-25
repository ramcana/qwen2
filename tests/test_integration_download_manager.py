"""
Integration tests for ModelDownloadManager
Tests the complete download workflow with mocked HuggingFace Hub
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_download_manager import (
    ModelDownloadManager,
    ModelDownloadConfig,
    download_qwen_image,
    download_qwen2_vl
)


class TestDownloadManagerIntegration(unittest.TestCase):
    """Integration tests for the complete download workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelDownloadManager(cache_dir=self.temp_dir)
        
        # Mock repository info
        self.mock_repo_info = Mock()
        self.mock_repo_info.siblings = [
            Mock(rfilename="config.json", size=1024),
            Mock(rfilename="model.safetensors", size=1024*1024*100),  # 100MB
            Mock(rfilename="tokenizer.json", size=2048),
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('model_download_manager.snapshot_download')
    @patch('model_download_manager.repo_info')
    def test_complete_qwen_image_download_workflow(self, mock_repo_info, mock_snapshot):
        """Test complete Qwen-Image download workflow"""
        # Setup mocks
        mock_repo_info.return_value = self.mock_repo_info
        download_path = Path(self.temp_dir) / "Qwen-Image"
        download_path.mkdir(parents=True)
        mock_snapshot.return_value = str(download_path)
        
        # Create mock downloaded files for integrity verification
        (download_path / "model_index.json").write_text("{}")
        (download_path / "scheduler").mkdir()
        (download_path / "scheduler" / "scheduler_config.json").write_text("{}")
        (download_path / "text_encoder").mkdir()
        (download_path / "text_encoder" / "config.json").write_text("{}")
        (download_path / "unet").mkdir()
        (download_path / "unet" / "config.json").write_text("{}")
        (download_path / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        (download_path / "vae").mkdir()
        (download_path / "vae" / "config.json").write_text("{}")
        
        # Mock the _get_local_model_path to return our test path for integrity verification
        with patch.object(self.manager, '_get_local_model_path', return_value=download_path):
            with patch.object(self.manager, 'verify_model_integrity', return_value=True):
                # Test the download
                result = self.manager.download_qwen_image()
        
        # Verify results
        self.assertTrue(result)
        mock_snapshot.assert_called_once()
        
        # Verify snapshot_download was called with correct parameters
        call_args = mock_snapshot.call_args
        self.assertEqual(call_args.kwargs['repo_id'], "Qwen/Qwen-Image")
        self.assertTrue(call_args.kwargs['resume_download'])
    
    @patch('model_download_manager.snapshot_download')
    @patch('model_download_manager.repo_info')
    def test_complete_qwen2_vl_download_workflow(self, mock_repo_info, mock_snapshot):
        """Test complete Qwen2-VL download workflow"""
        # Setup mocks
        mock_repo_info.return_value = self.mock_repo_info
        download_path = Path(self.temp_dir) / "Qwen2-VL-7B-Instruct"
        download_path.mkdir(parents=True)
        mock_snapshot.return_value = str(download_path)
        
        # Create mock downloaded files for integrity verification
        (download_path / "config.json").write_text("{}")
        (download_path / "generation_config.json").write_text("{}")
        (download_path / "model.safetensors.index.json").write_text("{}")
        (download_path / "tokenizer.json").write_text("{}")
        (download_path / "tokenizer_config.json").write_text("{}")
        (download_path / "model-00001-of-00002.safetensors").write_bytes(b"weights")
        
        # Mock the _get_local_model_path to return our test path for integrity verification
        with patch.object(self.manager, '_get_local_model_path', return_value=download_path):
            with patch.object(self.manager, 'verify_model_integrity', return_value=True):
                # Test the download
                result = self.manager.download_qwen2_vl("7B")
        
        # Verify results
        self.assertTrue(result)
        mock_snapshot.assert_called_once()
        
        # Verify snapshot_download was called with correct parameters
        call_args = mock_snapshot.call_args
        self.assertEqual(call_args.kwargs['repo_id'], "Qwen/Qwen2-VL-7B-Instruct")
        self.assertTrue(call_args.kwargs['resume_download'])
        self.assertEqual(call_args.kwargs['max_workers'], 2)  # Limited for large models
    
    @patch('model_download_manager.snapshot_download')
    @patch('model_download_manager.repo_info')
    def test_download_with_progress_tracking(self, mock_repo_info, mock_snapshot):
        """Test download with progress tracking"""
        # Setup mocks
        mock_repo_info.return_value = self.mock_repo_info
        download_path = Path(self.temp_dir) / "Qwen2-VL-7B-Instruct"
        download_path.mkdir(parents=True)
        mock_snapshot.return_value = str(download_path)
        
        # Create minimal files for integrity check
        (download_path / "config.json").write_text("{}")
        (download_path / "generation_config.json").write_text("{}")
        (download_path / "model.safetensors.index.json").write_text("{}")
        (download_path / "tokenizer.json").write_text("{}")
        (download_path / "tokenizer_config.json").write_text("{}")
        (download_path / "model.safetensors").write_bytes(b"weights")
        
        # Setup progress tracking
        progress_updates = []
        def progress_callback(progress):
            progress_updates.append(progress)
        
        self.manager.add_progress_callback(progress_callback)
        
        # Mock the _get_local_model_path to return our test path for integrity verification
        with patch.object(self.manager, '_get_local_model_path', return_value=download_path):
            with patch.object(self.manager, 'verify_model_integrity', return_value=True):
                # Test download
                config = ModelDownloadConfig(
                    model_name="Qwen/Qwen2-VL-7B-Instruct",
                    verify_integrity=True
                )
                result = self.manager.download_model("Qwen/Qwen2-VL-7B-Instruct", config)
        
        # Verify results
        self.assertTrue(result)
        # Progress callbacks should have been called
        self.assertGreater(len(progress_updates), 0)
    
    def test_convenience_functions(self):
        """Test convenience functions work correctly"""
        with patch.object(ModelDownloadManager, 'download_qwen_image', return_value=True):
            result = download_qwen_image()
            self.assertTrue(result)
        
        with patch.object(ModelDownloadManager, 'download_qwen2_vl', return_value=True):
            result = download_qwen2_vl("7B")
            self.assertTrue(result)
    
    @patch('model_download_manager.repo_info')
    def test_model_availability_check_integration(self, mock_repo_info):
        """Test model availability checking integration"""
        mock_repo_info.return_value = self.mock_repo_info
        
        # Mock _get_local_model_path to return None (no local model)
        with patch.object(self.manager, '_get_local_model_path', return_value=None):
            # Test remote-only availability
            status = self.manager.check_model_availability("Qwen/Qwen-Image")
        
        self.assertTrue(status["remote_available"])
        self.assertFalse(status["local_available"])
        self.assertFalse(status["local_complete"])
        self.assertEqual(len(status["missing_files"]), 0)  # No local files to compare
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios"""
        # Test unsupported model
        result = self.manager.download_model("unsupported/model")
        self.assertFalse(result)
        
        # Test invalid Qwen2-VL size
        result = self.manager.download_qwen2_vl("invalid")
        self.assertFalse(result)
    
    def test_disk_space_checking(self):
        """Test disk space checking functionality"""
        # Test with sufficient space (mock large free space)
        with patch('model_download_manager.shutil.disk_usage') as mock_usage:
            mock_usage.return_value = Mock(free=100 * 1024**3)  # 100GB free
            result = self.manager.check_disk_space(50)  # Need 50GB
            self.assertTrue(result)
        
        # Test with insufficient space
        with patch('model_download_manager.shutil.disk_usage') as mock_usage:
            mock_usage.return_value = Mock(free=10 * 1024**3)  # 10GB free
            result = self.manager.check_disk_space(50)  # Need 50GB
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)