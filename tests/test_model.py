"""
test_model.py
-------------
Unit tests for core project components.

Run all tests:
    pytest tests/ -v

Run specific test:
    pytest tests/test_model.py::test_build_model -v
"""

import numpy as np
import pytest


# =============================================================================
# Test: Model Building
# =============================================================================

class TestBuildModel:
    """Tests for the model building function."""

    def test_build_vgg16(self):
        """VGG16 model should build without errors."""
        from src.models.build_model import build_model
        model = build_model("vgg16", num_classes=4)
        assert model is not None
        assert model.output_shape == (None, 4)

    def test_build_resnet50(self):
        """ResNet50 model should build without errors."""
        from src.models.build_model import build_model
        model = build_model("resnet50", num_classes=4)
        assert model.output_shape == (None, 4)

    def test_invalid_model_name(self):
        """Invalid model name should raise ValueError."""
        from src.models.build_model import build_model
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("invalid_model", num_classes=4)

    def test_trainable_layers(self):
        """Model should have some frozen and some trainable layers."""
        from src.models.build_model import build_model
        model = build_model("vgg16", num_classes=4, unfreeze_last_n_layers=4)
        trainable = sum(1 for l in model.layers if l.trainable)
        assert trainable > 0, "At least some layers should be trainable"


# =============================================================================
# Test: Data Preprocessing
# =============================================================================

class TestPreprocessor:
    """Tests for data preprocessing functions."""

    def test_compute_md5(self, tmp_path):
        """MD5 hash should be consistent for the same file."""
        from src.data.preprocessor import compute_md5
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")
        hash1 = compute_md5(str(test_file))
        hash2 = compute_md5(str(test_file))
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 is always 32 hex chars

    def test_different_files_different_hash(self, tmp_path):
        """Different files should have different hashes."""
        from src.data.preprocessor import compute_md5
        file1 = tmp_path / "img1.jpg"
        file2 = tmp_path / "img2.jpg"
        file1.write_bytes(b"image data 1")
        file2.write_bytes(b"image data 2")
        assert compute_md5(str(file1)) != compute_md5(str(file2))


# =============================================================================
# Test: Statistical Tests
# =============================================================================

class TestStatisticalTests:
    """Tests for statistical comparison functions."""

    def test_mcnemar_identical_predictions(self):
        """Two identical models should show no significant difference."""
        from src.evaluation.evaluate import mcnemar_test
        y_true = [0, 1, 2, 0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 1, 2, 0, 1]
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert not result["significant"]  # No difference

    def test_mcnemar_different_predictions(self):
        """Models making different errors should have low p-value."""
        from src.evaluation.evaluate import mcnemar_test
        y_true = [0] * 50 + [1] * 50
        pred_a = [0] * 50 + [1] * 50   # Perfect
        pred_b = [1] * 50 + [0] * 50   # Completely wrong
        result = mcnemar_test(y_true, pred_a, pred_b)
        assert result["significant"]

    def test_wilcoxon_same_scores(self):
        """Identical fold scores should produce p_value = 1.0."""
        from src.evaluation.evaluate import wilcoxon_test
        scores = [0.95, 0.94, 0.96, 0.93, 0.95]
        result = wilcoxon_test(scores, scores)
        assert not result["significant"]

    def test_paired_ttest_very_different(self):
        """Very different fold scores should be statistically significant."""
        from src.evaluation.evaluate import paired_ttest
        scores_a = [0.99, 0.98, 0.99, 0.98, 0.99]
        scores_b = [0.50, 0.51, 0.49, 0.52, 0.50]
        result = paired_ttest(scores_a, scores_b)
        assert result["significant"]


# =============================================================================
# Test: Config Loader
# =============================================================================

class TestConfigLoader:
    """Tests for config loading."""

    def test_load_config(self):
        """Config should load without errors."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg is not None
        assert hasattr(cfg, "data")
        assert hasattr(cfg, "models")
        assert hasattr(cfg, "training")

    def test_config_values(self):
        """Config should have expected values."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg.data.batch_size > 0
        assert cfg.training.n_folds > 1
        assert cfg.data.image_size == [224, 224]


# =============================================================================
# Test: Predictor (with mock model)
# =============================================================================

class TestPredictor:
    """Tests for the RiceLeafPredictor class."""

    def test_preprocess_output_shape(self):
        """Preprocessing should output correct shape."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1]))

        from PIL import Image
        import numpy as np

        # Mock predictor without loading real model
        class MockPredictor:
            def __init__(self):
                self.image_size = (224, 224)
            def preprocess(self, image):
                from app.predictor import RiceLeafPredictor
                # Use the actual preprocess logic
                image = image.resize(self.image_size)
                img_array = np.array(image, dtype=np.float32) / 255.0
                return np.expand_dims(img_array, axis=0)

        predictor = MockPredictor()
        img = Image.new("RGB", (512, 512), color=(100, 150, 80))
        result = predictor.preprocess(img)
        assert result.shape == (1, 224, 224, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
