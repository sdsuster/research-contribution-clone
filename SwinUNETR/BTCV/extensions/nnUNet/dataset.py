import os
import warnings
from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import Dataset

import json
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles
# from extensions.nnUNet.utils import unpack_dataset
import os

import numpy as np
import torch
from monai import data, transforms
import math


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class nnUNetBaseDataset(Dataset):
    """
    Defines the interface
    """
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__()
        # print('loading dataset')
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()

        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = identifiers
        
    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, id):
        return self.load_case(self.identifiers[id])

    @abstractmethod
    def load_case(self, identifier):
        pass

    @staticmethod
    @abstractmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
            ):
        pass

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str) -> List[str]:
        pass

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = 4,
                       verify: bool = True):
        pass

from monai.data.meta_tensor import MetaTensor
class nnUNetDatasetNumpy(nnUNetBaseDataset):
    def load_case(self, identifier):
        data_npy_file = join(self.source_folder, identifier + '.npy')
        if not isfile(data_npy_file):
            data = np.load(join(self.source_folder, identifier + '.npz'))['data']
        else:
            data = np.load(data_npy_file)

        seg_npy_file = join(self.source_folder, identifier + '_seg.npy')
        if not isfile(seg_npy_file):
            seg = np.load(join(self.source_folder, identifier + '.npz'))['seg']
        else:
            seg = np.load(seg_npy_file)

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_npy_file = join(self.folder_with_segs_from_previous_stage, identifier + '.npy')
            if isfile(prev_seg_npy_file):
                seg_prev = np.load(prev_seg_npy_file)
            else:
                seg_prev = np.load(join(self.folder_with_segs_from_previous_stage, identifier + '.npz'))['seg']
        else:
            seg_prev = None

        # properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        # print(properties['nibabel_stuff']['reoriented_affine'])
        return {"image": data, 'label': seg, 'label_prev': seg_prev} if seg_prev else  {"image": data, 'label': seg}

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = 4,
                       verify: bool = True):
        return unpack_dataset(folder, True, overwrite_existing, num_processes, verify)
    
def do_split(base_dir, sub_folder, k = 5):
    from sklearn.model_selection import KFold
    ids = nnUNetDatasetNumpy.get_identifiers(join(base_dir, sub_folder))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Generate nnU-Net-style splits
    splits = []
    for train_idx, val_idx in kf.split(ids):
        train_ids = [ids[i] for i in train_idx]
        val_ids = [ids[i] for i in val_idx]
        splits.append({
            "train": train_ids,
            "val": val_ids
        })

    # Save to JSON file
    with open(join(base_dir,"splits_final.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print("Saved 5-fold nnU-Net splits to splits_final.json")

def load_fold_split(json_path, fold_id):
    """
    Load nnU-Net-style cross-validation split JSON and return train/val list for a specific fold.
    
    Args:
        json_path (str): Path to splits_final.json
        fold_id (int): Which fold to load (0-based index)
    
    Returns:
        (train_ids, val_ids): Lists of IDs
    """
    with open(json_path, "r") as f:
        splits = json.load(f)
    
    if fold_id >= len(splits):
        raise ValueError(f"Requested fold_id {fold_id} exceeds available folds ({len(splits)})")

    fold = splits[fold_id]
    return fold["train"], fold["val"]
def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            # transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.EnsureTyped(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                constant_values=0
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            # transforms.EnsureChannelFirstd(keys=["image", "label"]),
            
            # transforms.EnsureTyped(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            # transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            
            # transforms.EnsureTyped(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    infer_transform = transforms.Compose(
        [
            # transforms.EnsureChannelFirstd(keys=["image"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            # transforms.ToTensord(keys=["image"]),
            
            # transforms.EnsureTyped(keys=["image"]),
            transforms.Spacingd(
                keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear")
            ),
            # transforms.EnsureTyped(keys=["image"]),  # Preserves metadata
            transforms.ToTensord(keys=["image"]),
        ]
    )
    if args.test_mode:
        if args.test_set:
            test_ds = data.Dataset(data=test_files, transform=infer_transform)
            test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = test_loader
        else:
            test_ds = data.Dataset(data=test_files, transform=test_transform)
            test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = test_loader

    else:
        
        train_ids, val_ids = load_fold_split(datalist_json, fold_id=0)
        datalist = nnUNetDatasetNumpy(folder=join(args.data_dir,'nnUNetPlans_3d_fullres'), identifiers=train_ids)
        # for x in datalist:
        #     print(x)

        # datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        
        
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        # batch = next(iter(train_loader))
        # label = batch["label"]  # shape: (B, 1, D, H, W) or (B, 1, H, W)
        # print("Label shape:", label.shape)
        # batch = next(iter(train_loader))
        # label = batch["label"]  # shape: (B, 1, D, H, W) or (B, 1, H, W)
        # print("Label shape:", label.shape)
        # exit()
        # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_files = nnUNetDatasetNumpy(folder=join(args.data_dir,'nnUNetPlans_3d_fullres'), identifiers=val_ids)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
