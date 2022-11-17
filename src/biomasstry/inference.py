"""Inference API."""

import pytest

from utils import get_random_image, make_output_tensor


SAMPLE_OUTPUT_IMAGE = "output_image.tiff"

def load_model():
    return make_output_tensor


class BioMassPredictor:
    """Class for above ground biomass (AGBM) prediction from satellite images."""
    def __init__(self, transforms=None):
        self.model = load_model()
        self.transforms = transforms

    def predict(self, image):
        """Return the AGBM for each pixel."""
        if self.transforms is not None:
            image_tensor = self.transforms(image)
            pred = self.model(image_tensor)

        return SAMPLE_OUTPUT_IMAGE

class TestBioMassPredictor:
    """Class for running unit tests on inference."""
    def test_instance(self):
        predictor = BioMassPredictor()
        assert isinstance(predictor, BioMassPredictor)

    def test_predict(self):
        predictor = BioMassPredictor()
        image = get_random_image()
        pred = predictor.predict(image)
        assert pred == SAMPLE_OUTPUT_IMAGE
