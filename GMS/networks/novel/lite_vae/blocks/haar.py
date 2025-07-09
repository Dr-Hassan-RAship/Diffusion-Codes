import torch
import pywt
import numpy as np
from torch import nn, Tensor
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import re


class HaarTransform(nn.Module):
    """
    Recursive 2D Haar Wavelet Transform (DWT + iDWT) with optional visualization.

    Supports PyTorch (B, C, H, W) tensors, and can output stacked or nested coeffs.
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels

    # ------------------------------------------------------------------
    def dwt_recursive(self, x: np.ndarray) -> List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Perform recursive Haar DWT and return list of [(LL, (LH, HL, HH))] from shallow to deep.
        """
        coeffs_all = []
        current = x
        for _ in range(self.levels):
            coeffs2 = pywt.dwt2(current, "haar")
            LL, (LH, HL, HH) = coeffs2
            coeffs_all.append((LL, (LH, HL, HH)))
            current = LL
        return coeffs_all

    # ------------------------------------------------------------------
    def reconstruct_from_coeffs(
        self, coeffs_all: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ) -> np.ndarray:
        """
        Reconstruct full image using pywt.waverec2 from coeffs list.
        """
        LL_final, _ = coeffs_all[-1]
        detail_list = [t[1] for t in reversed(coeffs_all)]  # deep → shallow
        coeffs = [LL_final, *detail_list]
        return pywt.waverec2(coeffs, "haar")

    # ------------------------------------------------------------------
    def plot_coeffs(self, coeffs_all: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> None:
        """
        Visualize approximation + detail bands for each level.
        """
        levels = len(coeffs_all)
        titles = ["Approximation", "Horizontal", "Vertical", "Diagonal"]
        fig, axes = plt.subplots(levels, 4, figsize=(14, 4 * levels))

        if levels == 1:
            axes = axes.reshape(1, -1)

        for lvl, (LL, (LH, HL, HH)) in enumerate(coeffs_all, 1):
            for col, band in enumerate([LL, LH, HL, HH]):
                ax = axes[lvl - 1, col]
                ax.imshow(band, cmap="gray", interpolation="nearest")
                ax.set_title(f"Level {lvl} – {titles[col]}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def group_by_level_strict(self, 
        coeff_dict: Dict[str, np.ndarray],
        *,
        batch_size: int,
        num_levels: int,
        return_dict = True
    ) -> List[torch.Tensor] | Dict[str, torch.Tensor]:
        """
        Convert the dict from HaarTransform(stacked_version=True) into a list of
        tensors [level0, level1, level2] with shape (B, 4, H, W).

        Guarantees:
        • Each (batch_i, level_j) appears exactly once.
        • Tensors are ordered so batch dimension is aligned across levels.
        • Raises ValueError if any coefficient is missing or duplicated.
        """
        
        _key_re = re.compile(r"batch_(\d+)_level_(\d+)")
         
        # Allocate empty grid: grid[level][batch] = array
        grid: List[List[np.ndarray | None]] = [
            [None] * batch_size for _ in range(num_levels)
        ]

        # Fill the grid
        for key, arr in coeff_dict.items():
            m = _key_re.fullmatch(key)
            if m is None:
                raise ValueError(f"Key '{key}' does not match 'batch_i_level_j'")
            b_idx, l_idx = map(int, m.groups())
            if not (0 <= b_idx < batch_size) or not (0 <= l_idx < num_levels):
                raise ValueError(f"Key '{key}' index out of range")
            if grid[l_idx][b_idx] is not None:
                raise ValueError(f"Duplicate entry for batch {b_idx} level {l_idx}")
            grid[l_idx][b_idx] = arr

        # Ensure no missing entries
        for l, row in enumerate(grid):
            for b, arr in enumerate(row):
                if arr is None:
                    raise ValueError(f"Missing coeffs for batch {b} level {l}")

        # Stack per level into tensors
        level_tensors: List[torch.Tensor] = []
        for row in grid:                                # iterate over levels
            stacked = torch.stack([torch.from_numpy(arr) for arr in row], dim=0)
            level_tensors.append(stacked)

        if return_dict:
            return level_tensors, {f"level_{idx}": tensor for idx, tensor in enumerate(level_tensors)}
        else:
            return level_tensors

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor | list,
        *,
        inverse: bool = False,
        nchw: bool = True,
        stacked_version: bool = False
    ):
        """
        Perform Haar DWT or inverse DWT.
        """

        if inverse:
            coeffs_all = x  # type: ignore
            recon = self.reconstruct_from_coeffs(coeffs_all)
            return recon, coeffs_all

        if not nchw:
            raise NotImplementedError("Only NCHW tensors are supported.")

        b, c, h, w = x.shape
        coeffs_batch = []

        # Convert to numpy outside the loop to prevent slow memory migration
        x_np = x[:, 0].detach().cpu().numpy().astype(np.float32)

        for i in range(b):
            coeffs_i = self.dwt_recursive(x_np[i])  # grayscale input
            coeffs_batch.append(coeffs_i)

        if stacked_version:
            # Build dict in one go using nested comprehension
            dict_coeffs = {
                f"batch_{bi}_level_{li}": np.stack([LL, LH, HL, HH], axis=0)
                for bi, coeffs_i in enumerate(coeffs_batch)
                for li, (LL, (LH, HL, HH)) in enumerate(coeffs_i)
            }
            return self.group_by_level_strict(dict_coeffs, batch_size=b, num_levels=self.levels)

        return coeffs_batch

