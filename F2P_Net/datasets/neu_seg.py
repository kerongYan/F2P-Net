import json
from os.path import join
import os
from F2P_Net.datasets.base import BinaryF2P_NetDataset


few_shot_img_dict = {
    1: ['train/004360.jpg'],
    4: ['train/004360.jpg', 'train/000201.jpg', 'train/002069.jpg', 'train/000626.jpg'],
    16: [
        'train/000223.jpg',
        'train/004360.jpg',
        'train/000201.jpg',
        'train/002069.jpg',
        'train/000626.jpg',
        'train/000289.jpg',
        'train/000515.jpg',
        'train/000532.jpg',
        'train/000609.jpg',
        'train/000619.jpg',
        'train/000639.jpg',
        'train/000668.jpg',
        'train/004031.jpg',
        'train/004066.jpg',
        'train/004075.jpg',
        'train/004159.jpg'
    ],
    32: [
        'train/000225.jpg',
        'train/001010.jpg',
        'train/000248.jpg',
        'train/000258.jpg',
        'train/000265.jpg',
        'train/000269.jpg',
        'train/004360.jpg',
        'train/000521.jpg',
        'train/000529.jpg',
        'train/000532.jpg',
        'train/000591.jpg',
        'train/000598.jpg',
        'train/000657.jpg',
        'train/000615.jpg',
        'train/000622.jpg',
        'train/000626.jpg',
        'train/000630.jpg',
        'train/000641.jpg',
        'train/000646.jpg',
        'train/004031.jpg',
        'train/004039.jpg',
        'train/003752.jpg',
        'train/004059.jpg',
        'train/004062.jpg',
        'train/004066.jpg',
        'train/004075.jpg',
        'train/004160.jpg',
        'train/004182.jpg',
        'train/004193.jpg',
        'train/004204.jpg',
        'train/004296.jpg',
        'train/004380.jpg',
    ],
    64: [
        'train/000214.jpg',
        'train/000215.jpg',
        'train/000223.jpg',
        'train/000233.jpg',
        'train/000248.jpg',
        'train/000258.jpg',
        'train/000265.jpg',
        'train/000269.jpg',
        'train/000271.jpg',
        'train/000276.jpg',
        'train/000283.jpg',
        'train/000289.jpg',
        'train/004360.jpg',
        'train/000513.jpg',
        'train/000514.jpg',
        'train/000515.jpg',
        'train/000521.jpg',
        'train/000527.jpg',
        'train/000529.jpg',
        'train/000532.jpg',
        'train/000534.jpg',
        'train/000545.jpg',
        'train/000549.jpg',
        'train/000556.jpg',
        'train/000557.jpg',
        'train/000561.jpg',
        'train/000578.jpg',
        'train/000585.jpg',
        'train/000591.jpg',
        'train/000598.jpg',
        'train/000603.jpg',
        'train/000606.jpg',
        'train/000610.jpg',
        'train/000611.jpg',
        'train/000615.jpg',
        'train/000619.jpg',
        'train/000626.jpg',
        'train/000627.jpg',
        'train/000629.jpg',
        'train/000633.jpg',
        'train/000639.jpg',
        'train/000651.jpg',
        'train/000658.jpg',
        'train/000660.jpg',
        'train/000664.jpg',
        'train/000668.jpg',
        'train/000686.jpg',
        'train/002069.jpg',
        'train/004031.jpg',
        'train/004039.jpg',
        'train/004058.jpg',
        'train/004059.jpg',
        'train/004062.jpg',
        'train/004065.jpg',
        'train/004066.jpg',
        'train/004070.jpg',
        'train/004075.jpg',
        'train/004159.jpg',
        'train/004160.jpg',
        'train/004182.jpg',
        'train/004193.jpg',
        'train/004204.jpg',
        'train/004296.jpg',
        'train/004380.jpg',
        ]
}

class NEU_SegDataset(BinaryF2P_NetDataset):

    def __init__(
            self,
            data_dir: str,
            train_flag: bool,
            shot_num: int = None,
            **super_args
    ):  
        json_path = join(data_dir, 'train.json' if train_flag else 'test.json') 
        with open(json_path, 'r') as j_f:
            json_config = json.load(j_f)

        for key in json_config.keys():
            json_config[key]['image_path'] = join(data_dir, json_config[key]['image_path'])

            
            for key in json_config:
                if 'mask_path' in json_config[key]:
                    json_config[key]['mask_path'] = join(data_dir, json_config[key]['mask_path'])
            
            json_config[key]['label'] = int(json_config[key]['label'])  
            json_config[key]['label_name'] = json_config[key]['label_name']  
            json_config[key]['object_name'] = json_config[key]['object_name']  
            
        if shot_num is not None:
            assert shot_num in [1, 4, 16, 32, 64], f"Invalid shot_num: {shot_num}! Must be either 1 or 4 or 16 or 32 or 64!"
            json_config = {key: value for key, value in json_config.items() if key in few_shot_img_dict[shot_num]}
            
        super(NEU_SegDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=8,
            area_threshold=20, relative_threshold=True,
            **super_args
        )