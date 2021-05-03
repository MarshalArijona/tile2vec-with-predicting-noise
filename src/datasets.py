from torch.utils.data import Dataset

from torch.utils.data import DataLoader as TorchDataLoader
from src.dataloader import *

from torchvision import transforms

import torch
import glob
import os
import numpy as np

from src.data_utils import *
from src.interface import *

class TileTripletsDataset(Dataset):
    def __init__(self, tile_dir, transform=None, n_triplets=None,
        pairs_only=True):
        self.tile_dir = tile_dir
        self.tile_files = glob.glob(os.path.join(self.tile_dir, '*'))
        self.transform = transform
        self.n_triplets = n_triplets
        self.pairs_only = pairs_only

    def __len__(self):
        if self.n_triplets: return self.n_triplets
        else: return len(self.tile_files) // 3

    def __getitem__(self, idx):
        a = np.load(os.path.join(self.tile_dir, '{}anchor.npy'.format(idx)))
        n = np.load(os.path.join(self.tile_dir, '{}neighbor.npy'.format(idx)))
        
        if self.pairs_only:
            name = np.random.choice(['anchor', 'neighbor', 'distant'])
            d_idx = np.random.randint(0, self.n_triplets)
            d = np.load(os.path.join(self.tile_dir, '{}{}.npy'.format(d_idx, name)))
        else:
            d = np.load(os.path.join(self.tile_dir, '{}distant.npy'.format(idx)))
        
        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class TileTripletsDatasetNoise(TileTripletsDataset):
    def __init__(self,
                 tile_dir,
                 transform=None,
                 n_triplets=None,
                 pairs_only=True,
                 z_dims=512,
                 sampling=False,
                 **kwargs):
    
        super().__init__(tile_dir, transform, n_triplets, pairs_only, **kwargs)
        if not sampling:
            anchor_noise = np.expand_dims(generate_random_targets(n_triplets, z_dims), axis=1)
            neighbor_noise = np.expand_dims(generate_random_targets(n_triplets, z_dims), axis=1)
            distant_noise = np.expand_dims(generate_random_targets(n_triplets, z_dims), axis=1)

            noise = np.concatenate([anchor_noise, neighbor_noise, distant_noise], 1)
            
            self.nat = noise 
        else:
            noise = np.expand_dims(generate_random_targets(n_triplets, z_dims), axis=1)
            self.nat = noise 



    def __getitem__(self, index):
        sample = super().__getitem__(index)

        return sample, self.nat[index, :, :]

    def update_targets(self, indexes, new_targets, sampling=False):
        if not sampling:
            size = new_targets.shape[0] // 3
            
            new_anchor_targets = new_targets[:size, :]
            new_neighbor_targets = new_targets[size:2*size, :]
            new_distant_targets = new_targets[2*size:, :]
        
            self.nat[indexes, 0, :] = new_anchor_targets
            self.nat[indexes, 1, :] = new_neighbor_targets
            self.nat[indexes, 2, :] = new_distant_targets
        
        else:
            self.nat[indexes, 0, :] = new_targets


# ## TRANSFORMS ###

class GetBands(object):
    """
    Gets the first X bands of the tile triplet.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Tiles are already in [c, w, h] order
        a, n, d = (a[:self.bands,:,:], n[:self.bands,:,:], d[:self.bands,:,:])
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Randomly horizontal flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=2).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=2).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=1).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=1).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: a = np.rot90(a, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: n = np.rot90(n, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: d = np.rot90(d, k=rotations, axes=(1,2)).copy()
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        a, n, d = (clip_and_scale_image(sample['anchor'], self.img_type),
                   clip_and_scale_image(sample['neighbor'], self.img_type),
                   clip_and_scale_image(sample['distant'], self.img_type))
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, d = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            torch.from_numpy(sample['distant']).float())
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

# ## TRANSFORMS ###


def triplet_dataloader(img_type, tile_dir, bands=4, augment=True,
    batch_size=4, shuffle=True, num_workers=4, n_triplets=None,
    pairs_only=True):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    assert img_type in ['landsat', 'rgb', 'naip']
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(tile_dir, transform=transform,
        n_triplets=n_triplets, pairs_only=pairs_only)
    
    dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    
    return dataloader

def triplet_noise_dataloader(img_type, tile_dir, bands=4, augment=True,
                            batch_size=4, shuffle=True, num_workers=4, n_triplets=None,
                            pairs_only=True):
    assert img_type in ['landsat', 'rgb', 'naip']
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    
    dataset = TileTripletsDatasetNoise(tile_dir, transform=transform,
        n_triplets=n_triplets, pairs_only=pairs_only)
   
    cuda = torch.cuda.is_available()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                             num_workers=num_workers)
    return dataloader, dataset
