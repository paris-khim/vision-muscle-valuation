import cv2
import numpy as np

class MedicalAugmentor:
    """Clinical-grade medical data augmentation pipeline for clinical imaging."""
    def add_rician_noise(self, image, sigma=0.05):
        """Standard noise model for ultrasound/MRI imaging."""
        noise1 = np.random.normal(0, sigma, image.shape)
        noise2 = np.random.normal(0, sigma, image.shape)
        return np.sqrt((image + noise1)**2 + noise2**2)

    def elastic_deformation(self, image, alpha=10, sigma=2):
        """Simulate tissue deformation common in muscle scans."""
        from scipy.ndimage import gaussian_filter, map_coordinates
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        return map_coordinates(image, indices, order=1).reshape(shape)
