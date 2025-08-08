# ------------------------------------------------------------------------------#
#
# File name                 : image_dataset.py
# Purpose                   : PyTorch `Dataset` for paired image / mask loading
#                             with flexible extensions and optional pickle split.
# Usage                     : from data import ImageDataset
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 21, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import torch, os, pickle, random, logging

import numpy            as np

from PIL                        import Image
from torch.utils.data           import Dataset
from pathlib                    import Path
from typing                     import List, Tuple
from utils.tools                import get_precision_dtypes
from .transforms                import build_transforms
from configs.config             import *

# --------------------------- Pickle generator ---------------------------------#
def make_split_pickle(
    root_dir: str,
    split: float = 0.9,
    pickle_name: str = "_train_test_names.pkl",
) -> Path:
    """
    Scans `root_dir/images` & `root_dir/masks`, pairs files by basename,
    then creates a pickled dict with 90/10 (default) train/val split.

    Returns:
        Path to the generated pickle file.
    """
    img_dir  = Path(root_dir) / "images"
    mask_dir = Path(root_dir) / "masks"

    # Collect filenames with desired extensions
    img_files  = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_FORMAT)]
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(IMG_FORMAT)]

    img_bases  = {Path(f).stem for f in img_files}
    mask_bases = {Path(f).stem for f in mask_files}

    common_bases = sorted(img_bases & mask_bases)
    random.seed(SEED)
    random.shuffle(common_bases)

    train_len  = int(split * len(common_bases))
    train_list = common_bases[:train_len]
    val_list   = common_bases[train_len:]

    pickle_dict = {
        "train": {"name_list": train_list},
        "val": {"name_list": val_list},
    }

    pickle_path = Path(root_dir) / pickle_name
    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_dict, f)

    logging.info(
        f"Pickle saved to {pickle_path} | "
        f"Train: {len(train_list)} | Val: {len(val_list)}"
    )
    return pickle_path


# --------------------------- Main Dataset class -------------------------------#
class ImageDataset(Dataset):
    """
    Dataset for paired image / mask segmentation tasks.

    Args:
        pickle_file     : Path to train/val pickle (creates one if None).
        root_dir        : Root containing `images/` & `masks/` (used if pickle is None).
        stage           : "train", "val", or "test".
        img_size        : Final resize size.
        file_exts       : Tuple of allowed extensions.
    """

    def __init__(
        self,
        pickle_file : str    = None,
        root_dir    : str    = None,
        precision   : str    = "float16",
        stage       : str    = "train",
        img_size    : int    = 224,
    ) -> None:

        super().__init__()

        # ------ Get appropriate ext. for dtype -----#
        self.np_dtype, self.torch_dtype = get_precision_dtypes(precision)

        # ---------- Load / create pickle ---------- #
        if pickle_file is None:
            if root_dir is None:
                raise ValueError("Either `pickle_file` or `root_dir` must be provided.")
            pickle_file = make_split_pickle(root_dir, split=0.9)

        with open(pickle_file, "rb") as f:
            split_dict = pickle.load(f)

        self.names:    List[str] = split_dict[stage]["name_list"]
        self.stage:    str       = stage
        
        self.img_dir:  Path   = Path(pickle_file).parent / "images"
        self.mask_dir: Path   = Path(pickle_file).parent / "masks"
        
        self.transforms       = build_transforms(stage, img_size)

        logging.info(f"{stage.capitalize()} set size: {len(self.names)}")

    # ------------------------ __getitem__ -------------------------------------#
    def __getitem__(self, idx):
        name = self.names[idx]

        img_base  = self.img_dir / name
        mask_base = self.mask_dir / name

        img_path  = img_base.with_suffix(IMG_FORMAT)
        mask_path = mask_base.with_suffix(IMG_FORMAT)

        # Load RGB
        img_np  = np.array(Image.open(img_path).convert("RGB"), dtype = np.float32)
        mask_np = np.array(Image.open(mask_path).convert("RGB"), dtype = np.float32)

        augmented = self.transforms(image = img_np, mask = mask_np)

        img = augmented["image"].to(self.torch_dtype)
        seg = augmented["mask"].to(self.torch_dtype)
        return {
            "name": name,
            "img": img,
            "seg": seg,
        }

    # ------------------------ __len__ ----------------------------------------#
    def __len__(self):
        return len(self.names)

# --------------------------------- End -----------------------------------------#
