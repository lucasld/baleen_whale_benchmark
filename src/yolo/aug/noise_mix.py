"""Albumentations transform for blending noise-only spectrograms into labeled samples."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import albumentations as A


class NoiseMix(A.ImageOnlyTransform):
    """Blend a randomly sampled noise spectrogram into the current spectrogram."""

    def __init__(
        self,
        noise_paths: Iterable[Path],
        alpha: float = 0.25,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        self.noise_paths: List[str] = [str(Path(p)) for p in noise_paths]
        if not self.noise_paths:
            raise ValueError("NoiseMix requires at least one noise image path.")
        self.alpha = float(alpha)
        # `always_apply` is unused here because the base class does not support it in this Albumentations version.
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:  # type: ignore[override]
        noise_path = random.choice(self.noise_paths)
        noise_img = cv2.imread(noise_path, cv2.IMREAD_COLOR)
        if noise_img is None:
            raise FileNotFoundError(f"NoiseMix could not read noise image: {noise_path}")
        noise_img = cv2.resize(noise_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        blended = cv2.addWeighted(img, 1.0 - self.alpha, noise_img, self.alpha, 0.0)
        return blended.astype(img.dtype, copy=False)
