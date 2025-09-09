import json
from os.path import join

from F2P_Net.datasets.base import BinaryF2P_NetDataset


few_shot_img_dict = {
    1: ['defective/11350'],
    4: [
        'good/10526',
        'defective/10335',
        'defective/11477',
        'defective/12303'
    ],
    16: [
        'good/10888',
        'good/12190',
        'good/10526',
        'defective/10028',
        'defective/10712',
        'defective/10423',
        'defective/10638',
        'defective/10312',
        'defective/10335',
        'defective/11584',
        'defective/11398',
        'defective/10713',
        'defective/11350',
        'defective/11269',
        'defective/11477',
        'defective/11804',
        'defective/12303'
    ],
    32: [
        'good/10988',
        'good/10888',
        'good/11393',
        'good/12190',
        'good/10526',
        'good/10053',
        'good/11639',
        'good/10639',
        'good/10533',
        'good/12016',
        'defective/10028',
        'defective/10712',
        'defective/10382',
        'defective/10423',
        'defective/10638',
        'defective/10312',
        'defective/10335',
        'defective/11214',
        'defective/10797',
        'defective/11447',
        'defective/11477',
        'defective/12233',
        'defective/12315',
        'defective/11584',
        'defective/11398',
        'defective/10715',
        'defective/11350',
        'defective/11269',
        'defective/11804',
        'defective/12303',
        'defective/12071',
        'defective/12317',
        'defective/10135'
    ],
    64: [
        'good/10988',
        'good/10888',
        'good/11393',
        'good/12190',
        'good/10526',
        'good/10053',
        'good/11639',
        'defective/10028',
        'defective/10712',
        'defective/10382',
        'defective/10423',
        'defective/10638',
        'defective/10312',
        'defective/10335',
        'defective/11214',
        'defective/10797',
        'defective/11447',
        'defective/11477',
        'defective/12233',
        'defective/12315',
        'defective/11584',
        'defective/11398',
        'defective/10715',
        'defective/11350',
        'defective/11269',
        'defective/11804',
        'defective/12303',
        'defective/12071',
        'defective/12317',
        'defective/10135',
        'defective/10309',
        'defective/10602',
        'defective/10893',
        'defective/11458',
        'defective/11500',
        'defective/11952',
        'defective/11190',
        'defective/11404',
        'defective/11574',
        'defective/10426',
        'defective/10729',
        'defective/11025',
        'defective/11932',
        'defective/11927',
        'defective/12109',
        'defective/12160',
        'defective/12201',
        'defective/12216',
        'defective/11573',
        'defective/11506',
        'defective/11002',
        'defective/10951',
        'defective/11088',
        'defective/10637',
        'defective/10849',
        'defective/11101',
        'defective/11643',
        'defective/11884',
        'defective/11139',
        'defective/12070',
        'defective/10774',
        'defective/10756',
        'defective/10618',
        'defective/10514',
        'defective/10421',
        'defective/10414',
        'defective/10516'
    ]
}

class KolektorSDD2_Dataset(BinaryF2P_NetDataset):

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


        if shot_num is not None:
            assert shot_num in [1, 4, 16, 32, 64], f"Invalid shot_num: {shot_num}! Must be either 1 or 4 or 16 or 32 or 64!"
            json_config = {key: value for key, value in json_config.items() if key in few_shot_img_dict[shot_num]}

        super(KolektorSDD2_Dataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=8,
            area_threshold=20, relative_threshold=True,
            **super_args
        )