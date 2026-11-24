import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from einops import rearrange
from utils import random_click  # your custom click function

class BratsSAM(Dataset):
    """
    PyTorch Dataset for BraTS MRI data adapted for SAM training.
    Returns:
        dict with keys: 'image', 'mask', 'p_label', 'image_meta_dict'
    """

    def __init__(self, data_path, args, mode='Training', prompt='click', plane=False):
        """
        Args:
            data_path (str): Root path of dataset.
            args: Args object with at least: dataset, four_chan, mri, image_size, out_size.
            mode (str): 'Training' or 'Validation'.
            prompt (str): Type of prompt for SAM ('click').
            plane (bool): Option for slice selection.
        """
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.args = args
        self.four_chan = getattr(args, 'four_chan', True)
        self.mri = getattr(args, 'mri', None)
        self.img_size = getattr(args, 'image_size', 240)
        self.mask_size = getattr(args, 'out_size', 240)

        # Determine folder based on dataset
        self.folder = {
            'brats_ssa': 'brats_ssa',
            'brats_2020': 'brats_2020',
            'brats_men': 'brats_men',
            'brats_ped': 'brats_ped'
        }.get(args.dataset, 'brats')

        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, self.folder)) if f.is_dir()]

    def __len__(self):
        return len(self.subfolders)

    def load_modalities(self, subfolder):
        """
        Load MRI modalities and segmentation mask as numpy arrays.
        """
        base_name = os.path.basename(subfolder)

        if self.folder == 'brats_2020':
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}_t1.nii.gz")))
            t1c = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}_t1ce.nii.gz")))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}_t2.nii.gz")))
            t2f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}_flair.nii.gz")))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}_seg.nii.gz")))
        else:
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}-t1n.nii.gz")))
            t1c = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}-t1c.nii.gz")))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}-t2w.nii.gz")))
            t2f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}-t2f.nii.gz")))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subfolder, f"{base_name}-seg.nii.gz")))

        return t1, t1c, t2, t2f, mask

    def __getitem__(self, index):
        subfolder = self.subfolders[index]
        t1, t1c, t2, t2f, mask = self.load_modalities(subfolder)

        # Convert mask to whole tumor: 1 if any label present
        mask = (mask != 0).astype(np.uint8)

        # Generate a random point for SAM prompt if needed
        point_label = 1
        pt = None
        if self.prompt == 'click':
            point_label, pt = random_click(mask / 255, point_label)

        # Stack channels for SAM
        if self.four_chan:
            img = np.stack([t1, t1c, t2, t2f])  # (4, D, H, W)
        else:
            # Select MRI modality
            if self.mri in ['t1', 't1c', 't2', 't2f']:
                m = {'t1': t1, 't1c': t1c, 't2': t2, 't2f': t2f}[self.mri]
                img = np.stack([m, m, m])  # (3, D, H, W)
            else:
                img = np.stack([t1, t2, t2f])  # default 3-channel fallback

        # Rearrange to (C, H, W, D) for SAM
        img = rearrange(img, 'c d h w -> c h w d')
        mask = np.expand_dims(mask, axis=0)
        mask = rearrange(mask, 'c d h w -> c h w d')

        # Metadata
        image_meta_dict = {'filename_or_obj': subfolder}

        data = {
            'image': img.astype(np.float32),
            'mask': mask.astype(np.float32),
            'p_label': point_label,
            'image_meta_dict': image_meta_dict
        }

        return data
