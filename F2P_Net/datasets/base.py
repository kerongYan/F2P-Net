import random
from typing import Dict, List, Union

import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from F2P_Net.datasets.transforms import Compose

from F2P_Net.datasets.misc import generate_prompts_from_mask


class BaseSegDataset(Dataset):

    def __init__(
            self,
            dataset_config: Union[Dict, List[Dict]],
            label_threshold: Union[int, None] = 128,
            transforms: List = None
    ):
        self.label_threshold = label_threshold
        self.transforms = Compose(transforms) if transforms else None

        if isinstance(dataset_config, Dict):
            dataset_config_list = [dataset_config]
        elif isinstance(dataset_config, List):
            dataset_config_list = dataset_config
        else:
            raise RuntimeError(
                    f"Your given dataset_config should be either a dict or a list of dicts, "
                    f"but got {type(dataset_config)}!"
            )
        self.idx2img_gt_path = {}
        for config_dict in dataset_config_list:
            self.idx2img_gt_path.update(config_dict)
        self.idx_list = list(self.idx2img_gt_path.keys())


    def __len__(self):
        return len(self.idx_list)


    def __getitem__(self, index):
        index_name = self.idx_list[index]
        image = Image.open(self.idx2img_gt_path[index_name]['image_path']).convert('RGB')
        # Determine whether a mask exists; if not, generate a mask of the same dimensions as the input image, filled entirely with zeros.s
        if 'mask_path' not in self.idx2img_gt_path[index_name].keys():
            gt_mask = np.zeros_like(np.array(image))
        else:
            gt_mask = Image.open(self.idx2img_gt_path[index_name]['mask_path']).convert('L')
        image, gt_mask = np.array(image, dtype=np.float32), np.array(gt_mask, dtype=np.float32)
        # For the 0-255 mask, discretize its values into 0.0 or 1.0
        if self.label_threshold is not None :
            if gt_mask.max() > 0:
                gt_mask = (gt_mask / gt_mask.max()) * 255  # Extend the maximum mask value to 255
            gt_mask = np.where(gt_mask > self.label_threshold, 1.0, 0.0)

        # Ensure the image has three channels. If the image is single-channel (grayscale), expand it to three channels using np.repeat
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = np.repeat(image, repeats=3, axis=-1)
        elif len(image.shape) != 3 and image.shape[0] != 3:
            raise RuntimeError(f'Wrong image shape: {image.shape}. It should be either [H, W] or [H, W, 1] or [H, W, 3]!')

        # For the mask, ensure it is 2D ([H, W]). If the mask has multiple channels (e.g., RGB mask), take the first channel.
        if len(gt_mask.shape) == 3:
            if gt_mask.shape[2] == 3:
                gt_mask = gt_mask[:, :, 0]  # Keep only the first channel
            elif gt_mask.shape[2] == 1:
                gt_mask = gt_mask[:, :, 0]  # Keep the single channel case
            else:
                raise RuntimeError(f'Wrong mask shape: {gt_mask.shape}. It should be [H, W, 1] or [H, W, 3]!')
        elif len(gt_mask.shape) == 2:
            pass  # Already [H, W] shape, use it directly
        else:
            raise RuntimeError(f'Wrong mask shape: {gt_mask.shape}. It should be either [H, W] or [H, W, 1] or [H, W, 3]!')
        # If the image and mask dimensions differ, resize the image to match the mask dimensions.
        if image.shape[:2] != gt_mask.shape[:2]:
            if image.shape[0] == gt_mask.shape[1] and image.shape[1] == gt_mask.shape[0]:
                image = np.transpose(image, (1, 0, 2))
            else:
                image = cv2.resize(image, (gt_mask.shape[1], gt_mask.shape[0]))

        # If data augmentation (transforms) is specified, the transformation operations are applied. Transforms are expected to be a dictionary containing the keys ‘image’ and ‘mask’.
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=gt_mask)
            image, gt_mask = transformed["image"], transformed['mask']
        
        # Retrieve tag information
        label = int(self.idx2img_gt_path[index_name]['label'])  # Convert label to integer type, 0 indicates normal, 1 indicates abnormal, checked and read correctly
        label_name = self.idx2img_gt_path[index_name]['label_name']  # Get label name ("defective" or "good")
        object_name = self.idx2img_gt_path[index_name]['object_name']  # Get object name (e.g., "bottle")

        # Return a dictionary containing the image, ground truth mask, and sample index.
        return dict(
            images=image, gt_masks=gt_mask, index_name=index_name, label=label, label_name=label_name, object_name=object_name
        )


    @classmethod
    def collate_fn(cls, batch):

        batch_dict = dict()
        batch_copy = batch.copy()  # Create a copy of the batch to avoid modifying the original batch

        while len(batch) != 0:
            ele_dict = batch[0]
            if ele_dict is not None:
                for key in ele_dict.keys():
                    if key not in batch_dict.keys():
                        batch_dict[key] = []
                    batch_dict[key].append(ele_dict[key])
            
            batch.remove(ele_dict)

        
        batch_dict['images'] = [torch.from_numpy(item).permute(2, 0, 1) for item in batch_dict['images']]
        
        
        batch_dict['gt_masks'] = [torch.from_numpy(item) for item in batch_dict['gt_masks']]
        
        
        batch_dict['object_masks'] = \
            [torch.from_numpy(item) if item is not None else None for item in batch_dict['object_masks']]

        # Fill the prompt points so that each prompt contains an equal number of points.
        point_coords, point_labels = [], []
        for item in batch_dict['point_coords']:
           
            if item is None:
                point_coords.append(None)
                point_labels.append(None)
            else:
                _point_coords, _point_labels = item, []
                
                max_num_coords = max(len(_p_c) for _p_c in _point_coords)
                for _p_c in _point_coords:
                    
                    _point_labels.append([1 for _ in _p_c])

                    curr_num_coords = len(_p_c)
                    
                    if curr_num_coords < max_num_coords:
                        _p_c.extend([[0, 0] for _ in range(max_num_coords - curr_num_coords)])
                        _point_labels[-1].extend([-1 for _ in range(max_num_coords - curr_num_coords)])

                
                point_coords.append(torch.FloatTensor(_point_coords))
                point_labels.append(torch.LongTensor(_point_labels))
        
        
        batch_dict['point_coords'] = point_coords
        batch_dict['point_labels'] = point_labels

        
        batch_dict['box_coords'] = \
            [torch.FloatTensor(item) if item is not None else None for item in batch_dict['box_coords']]
        
        
        # Add the label (normal or abnormal) for each sample to the batch_dict.
        batch_dict['label'] = [ele_dict['label'] for ele_dict in batch_copy]  
        batch_dict['label_names'] = [ele_dict['label_name'] for ele_dict in batch_copy] 
        batch_dict['object_names'] = [ele_dict['object_name'] for ele_dict in batch_copy]  

        
        return batch_dict




class BinaryF2P_NetDataset(BaseSegDataset):

    def __init__(
            self,
            train_flag: bool,
            offline_prompt_points: Union[str, List[str]] = None,
            prompt_point_num: int = 1,
            max_object_num: int = None,
            object_connectivity: int = 8,
            area_threshold: int = 10,
            relative_threshold: bool = True,
            relative_threshold_ratio: float = 0.001,
            ann_scale_factor: int = 8,
            noisy_mask_threshold: float = 0.5,
            **super_args
    ):
        super(BinaryF2P_NetDataset, self).__init__(**super_args)
        self.train_flag = train_flag
        self.prompt_kwargs = dict(
            object_connectivity=object_connectivity,
            area_threshold=area_threshold,
            relative_threshold=relative_threshold,
            relative_threshold_ratio=relative_threshold_ratio,
            max_object_num=max_object_num,
            prompt_point_num=prompt_point_num,
            ann_scale_factor=ann_scale_factor,
            noisy_mask_threshold=noisy_mask_threshold
        )

        self.offline_prompt_points = None
        if offline_prompt_points is not None:
            self.offline_prompt_points = {}

            if isinstance(offline_prompt_points, str):
                offline_prompt_points = [offline_prompt_points]
            for item in offline_prompt_points:
                self.offline_prompt_points.update(item)


    def __getitem__(self, index):
        ret_dict = super(BinaryF2P_NetDataset, self).__getitem__(index)
        
        
        label = ret_dict['label']  
        label_name = ret_dict['label_name']  
        object_name = ret_dict['object_name']  


        '''
        The following code primarily processes ground truth segmentation masks (gt_mask) and generates three types of training prompts (point, box, mask). Through steps such as extracting object regions, randomly selecting points, computing bounding boxes, and generating noise masks, the model receives rich training input prompts, aiding it in better learning the object segmentation task.

        Specific steps:

        Extract object regions.
        Generate different prompts as required: point, box, and mask.
        Produce noise masks for training to enhance data diversity.
        '''
        point_coords, box_coords, noisy_object_masks, object_masks = generate_prompts_from_mask(
            gt_mask=ret_dict['gt_masks'],
            tgt_prompts=[random.choice(['point', 'box', 'mask'])] if self.train_flag else ['point', 'box'], 
            **self.prompt_kwargs
        )
        # offline random prompt points for evaluation
        if self.offline_prompt_points is not None:
            point_coords = self.offline_prompt_points[ret_dict['index_name']]

        ret_dict.update(
            point_coords=point_coords,
            box_coords=box_coords,
            noisy_object_masks=noisy_object_masks,
            object_masks=object_masks,
            label=label,  
            label_name=label_name,  
            object_name=object_name  
        )
        return ret_dict