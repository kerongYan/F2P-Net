import json
from os.path import join

from F2P_Net.datasets.base import BinaryF2P_NetDataset


few_shot_img_dict = {
    1: [ 'Class1/Train/defective/0873'],
    4: [
        'Class1/Train/defective/0873',
        'Class3/Train/defective/0617',
        'Class4/Train/defective/1139',
        'calss8/Train/defective/2097'
    ],
    16: [
        'Class1/Train/defective/0873',
        'Class2/Train/defective/0576', 
        'Class2/Train/defective/0578',
        'Class3/Train/defective/0617',
        'Class3/Train/defective/0637',
        'Class4/Train/defective/1139',
        'Class5/Train/defective/0733',
        'Class5/Train/defective/0923',
        'Class6/Train/defective/0607',
        'Class6/Train/defective/0881',
        'Class7/Train/defective/2284',
        'Class8/Train/defective/2053',
        'Class8/Train/defective/2067',
        'Class9/Train/defective/1845',
        'Class9/Train/defective/1883',
        'Class10/Train/defective/2292'
    ],
    32: [
        'Class1/Train/defective/0881',
        'Class1/Train/defective/1142',
        'Class1/Train/defective/0595',
        'Class2/Train/defective/0576', 
        'Class2/Train/defective/0578',
        'Class2/Train/defective/0649',
        'Class3/Train/defective/0617',
        'Class3/Train/defective/0637',
        'Class3/Train/defective/0856',
        'Class3/Train/defective/0993',
        'Class4/Train/defective/0845',
        'Class4/Train/defective/1139',
        'Class4/Train/defective/1140',
        'Class5/Train/defective/0599',
        'Class5/Train/defective/0733',
        'Class5/Train/defective/0923',
        'Class5/Train/defective/1050',
        'Class6/Train/defective/0607',
        'Class6/Train/defective/0881',
        'Class6/Train/defective/1034',
        'Class6/Train/defective/1071', 
        'Class7/Train/defective/1793',
        'Class7/Train/defective/2284',
        'Class7/Train/defective/2285',
        'Class8/Train/defective/1823',
        'Class8/Train/defective/2053',
        'Class8/Train/defective/2067',
        'Class8/Train/defective/2097',
        'Class9/Train/defective/1660',
        'Class9/Train/defective/1845',
        'Class9/Train/defective/1883',
        'Class9/Train/defective/2230',
        'Class10/Train/defective/2292' 
    ],
    64: [
        'Class1/Train/defective/0881',
        'Class1/Train/defective/1142', 
        'Class1/Train/defective/0595',
        'Class1/Train/defective/0873',
        'Class1/Train/defective/1132',
        'Class1/Train/defective/1141', 
        'Class2/Train/defective/0576', 
        'Class2/Train/defective/0578',
        'Class2/Train/defective/0649',
        'Class2/Train/defective/0907',
        'Class2/Train/defective/0989',
        'Class2/Train/defective/1039',
        'Class3/Train/defective/0578',
        'Class3/Train/defective/0607',
        'Class3/Train/defective/0617',
        'Class3/Train/defective/0637',
        'Class3/Train/defective/0856',
        'Class3/Train/defective/0993',
        'Class4/Train/defective/0576',
        'Class4/Train/defective/0587',
        'Class4/Train/defective/0845',
        'Class4/Train/defective/1139',
        'Class4/Train/defective/1140',
        'Class4/Train/defective/1141',
        'Class5/Train/defective/0589',  
        'Class5/Train/defective/0599',
        'Class5/Train/defective/0733',
        'Class5/Train/defective/0923',
        'Class5/Train/defective/1050',
        'Class5/Train/defective/1052',
        'Class6/Train/defective/0579',  
        'Class6/Train/defective/0581',
        'Class6/Train/defective/0607',
        'Class6/Train/defective/0881',
        'Class6/Train/defective/1034',
        'Class6/Train/defective/1071',
        'Class7/Train/defective/1154',
        'Class7/Train/defective/1171',
        'Class7/Train/defective/1770',
        'Class7/Train/defective/1793',
        'Class7/Train/defective/2284',
        'Class7/Train/defective/2285',
        'Class8/Train/defective/1152',
        'Class8/Train/defective/1154',
        'Class8/Train/defective/1345',
        'Class8/Train/defective/1823',
        'Class8/Train/defective/2053',
        'Class8/Train/defective/2056',
        'Class8/Train/defective/2097',
        'Class9/Train/defective/1153',
        'Class9/Train/defective/1189',
        'Class9/Train/defective/1512',
        'Class9/Train/defective/1660',
        'Class9/Train/defective/1845',
        'Class9/Train/defective/1883',
        'Class9/Train/defective/2230',
        'Class8/Train/defective/2067',
        'Class10/Train/defective/1158',
        'Class10/Train/defective/1182',
        'Class10/Train/defective/1626',
        'Class10/Train/defective/1884',
        'Class10/Train/defective/2265',
        'Class10/Train/defective/2278',
        'Class10/Train/defective/2288',
        'Class10/Train/defective/2292'
    ]
}

class DAGM2007_Dataset(BinaryF2P_NetDataset):

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

        super(DAGM2007_Dataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=8,
            area_threshold=20, relative_threshold=True,
            **super_args
        )