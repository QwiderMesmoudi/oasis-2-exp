import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import cv2
import timm
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, df, img_size=224, num_slices=50):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.num_slices = num_slices

    def __len__(self):
        return len(self.df)

    def load_nifti(self, path):
        img = nib.load(path).get_fdata()
        img = (img - img.mean()) / (img.std() + 1e-5)
        return img

    def get_slices(self, volume):
        z = volume.shape[2]
        center = z // 2
        half = self.num_slices // 2
        # Ensure we don't go out of bounds
        start, end = max(0, center-half), min(z, center+half)
        slices = volume[:, :, start:end]
        return slices

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume = self.load_nifti(row['path'])
        slices = self.get_slices(volume)

        images = []
        for i in range(slices.shape[2]):
            img = slices[:, :, i]
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = np.stack([img]*3, axis=0) 
            images.append(img)

        images = np.stack(images) 
        label = row['label']
        return torch.tensor(images, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class SliceModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=1)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        out = self.backbone(x)
        out = out.view(B, S)
        out = out.mean(dim=1) 
        return out