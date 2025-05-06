"""
Transforms for spectrogram data.
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List, Union, Callable


class Transform:
    """Base class for all transforms."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the transform to an image.
        
        Args:
            image (np.ndarray): Image to transform
            
        Returns:
            np.ndarray: Transformed image
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """String representation of the transform."""
        return self.__class__.__name__ + '()'


class Compose:
    """
    Compose multiple transforms together.
    
    Args:
        transforms (List[Transform]): List of transforms to compose
    """
    
    def __init__(self, transforms: List[Transform]):
        """Initialize the compose transform."""
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all transforms to an image.
        
        Args:
            image (np.ndarray): Image to transform
            
        Returns:
            np.ndarray: Transformed image
        """
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def __repr__(self) -> str:
        """String representation of the compose transform."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize:
    """
    Normalize an image by mean and standard deviation.
    
    Args:
        mean (float): Mean value
        std (float): Standard deviation
    """
    
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        """Initialize the normalize transform."""
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image.
        
        Args:
            image (np.ndarray): Image to normalize
            
        Returns:
            np.ndarray: Normalized image
        """
        return (image - self.mean) / self.std
    
    def __repr__(self) -> str:
        """String representation of the normalize transform."""
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomNoise:
    """
    Add random noise to an image.
    
    Args:
        mean (float): Mean of the noise
        std (float): Standard deviation of the noise
        prob (float): Probability of applying the transform
    """
    
    def __init__(self, mean: float = 0.0, std: float = 0.1, prob: float = 0.5):
        """Initialize the random noise transform."""
        self.mean = mean
        self.std = std
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Add random noise to an image.
        
        Args:
            image (np.ndarray): Image to transform
            
        Returns:
            np.ndarray: Transformed image
        """
        if np.random.random() < self.prob:
            noise = np.random.normal(self.mean, self.std, image.shape)
            noisy_image = image + noise
            # Clip to [0, 1] range
            return np.clip(noisy_image, 0, 1)
        return image
    
    def __repr__(self) -> str:
        """String representation of the random noise transform."""
        return self.__class__.__name__ + '(mean={0}, std={1}, prob={2})'.format(
            self.mean, self.std, self.prob
        )


class RandomHorizontalFlip:
    """
    Randomly flip an image horizontally.
    
    Args:
        prob (float): Probability of applying the transform
    """
    
    def __init__(self, prob: float = 0.5):
        """Initialize the random horizontal flip transform."""
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly flip an image horizontally.
        
        Args:
            image (np.ndarray): Image to transform
            
        Returns:
            np.ndarray: Transformed image
        """
        if np.random.random() < self.prob:
            return np.fliplr(image)
        return image
    
    def __repr__(self) -> str:
        """String representation of the random horizontal flip transform."""
        return self.__class__.__name__ + '(prob={0})'.format(self.prob)


class RandomTimeShift:
    """
    Randomly shift an image in the time dimension.
    
    Args:
        max_shift (int): Maximum number of pixels to shift
        prob (float): Probability of applying the transform
    """
    
    def __init__(self, max_shift: int = 5, prob: float = 0.5):
        """Initialize the random time shift transform."""
        self.max_shift = max_shift
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly shift an image in the time dimension.
        
        Args:
            image (np.ndarray): Image to transform (width, height, channels)
            
        Returns:
            np.ndarray: Transformed image
        """
        if np.random.random() < self.prob:
            # Get image dimensions
            width, height, channels = image.shape
            
            # Generate random shift amount
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            
            # Create shifted image
            shifted_image = np.zeros_like(image)
            
            # Apply shift (positive shift = move right, negative shift = move left)
            if shift > 0:
                shifted_image[:, shift:, :] = image[:, :(height - shift), :]
            elif shift < 0:
                shifted_image[:, :height + shift, :] = image[:, -shift:, :]
            else:
                return image
            
            return shifted_image
        
        return image
    
    def __repr__(self) -> str:
        """String representation of the random time shift transform."""
        return self.__class__.__name__ + '(max_shift={0}, prob={1})'.format(
            self.max_shift, self.prob
        )


class RandomFrequencyShift:
    """
    Randomly shift an image in the frequency dimension.
    
    Args:
        max_shift (int): Maximum number of pixels to shift
        prob (float): Probability of applying the transform
    """
    
    def __init__(self, max_shift: int = 5, prob: float = 0.5):
        """Initialize the random frequency shift transform."""
        self.max_shift = max_shift
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly shift an image in the frequency dimension.
        
        Args:
            image (np.ndarray): Image to transform (width, height, channels)
            
        Returns:
            np.ndarray: Transformed image
        """
        if np.random.random() < self.prob:
            # Get image dimensions
            width, height, channels = image.shape
            
            # Generate random shift amount
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            
            # Create shifted image
            shifted_image = np.zeros_like(image)
            
            # Apply shift (positive shift = move down, negative shift = move up)
            if shift > 0:
                shifted_image[shift:, :, :] = image[:(width - shift), :, :]
            elif shift < 0:
                shifted_image[:width + shift, :, :] = image[-shift:, :, :]
            else:
                return image
            
            return shifted_image
        
        return image
    
    def __repr__(self) -> str:
        """String representation of the random frequency shift transform."""
        return self.__class__.__name__ + '(max_shift={0}, prob={1})'.format(
            self.max_shift, self.prob
        )


class RandomMasking:
    """
    Randomly mask parts of an image.
    
    Args:
        max_masks (int): Maximum number of masks to apply
        max_width (int): Maximum width of each mask
        max_height (int): Maximum height of each mask
        prob (float): Probability of applying the transform
    """
    
    def __init__(self, max_masks: int = 3, max_width: int = 5, max_height: int = 10, prob: float = 0.5):
        """Initialize the random masking transform."""
        self.max_masks = max_masks
        self.max_width = max_width
        self.max_height = max_height
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly mask parts of an image.
        
        Args:
            image (np.ndarray): Image to transform
            
        Returns:
            np.ndarray: Transformed image
        """
        if np.random.random() < self.prob:
            # Get image dimensions
            width, height, channels = image.shape
            
            # Create a copy of the image
            masked_image = image.copy()
            
            # Generate random number of masks
            n_masks = np.random.randint(1, self.max_masks + 1)
            
            for _ in range(n_masks):
                # Generate random mask size
                mask_width = np.random.randint(1, self.max_width + 1)
                mask_height = np.random.randint(1, self.max_height + 1)
                
                # Generate random mask position
                start_x = np.random.randint(0, width - mask_width + 1)
                start_y = np.random.randint(0, height - mask_height + 1)
                
                # Apply mask (set values to 0)
                masked_image[start_x:start_x + mask_width, start_y:start_y + mask_height, :] = 0
            
            return masked_image
        
        return image
    
    def __repr__(self) -> str:
        """String representation of the random masking transform."""
        return self.__class__.__name__ + '(max_masks={0}, max_width={1}, max_height={2}, prob={3})'.format(
            self.max_masks, self.max_width, self.max_height, self.prob
        )
