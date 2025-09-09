import json
from os.path import join

from F2P_Net.datasets.base import BinaryF2P_NetDataset


few_shot_img_dict = {
    1: ['defective/exp5_num_20449'],
    4: [
      'defective/exp1_num_249594', 'defective/exp1_num_155300','defective/exp3_num_136351','defective/exp2_num_304305'
    ],
    16: [
        'good/exp3_num_319346',
        'good/exp5_num_37084',
        'defective/exp1_num_249594',
        'defective/exp1_num_155300',
        'defective/exp1_num_4727',
        'defective/exp2_num_109256',
        'defective/exp3_num_116541',
        'defective/exp3_num_20409',
        'defective/exp3_num_33395',
        'defective/exp3_num_136351',
        'defective/exp4_num_284532',
        'defective/exp5_num_317558',
        'defective/exp6_num_116623',
        'defective/exp6_num_258703',
        'defective/exp2_num_36302',
        'defective/exp2_num_304305'
    ],
    32: [
        'good/exp4_num_311355',
        'good/exp6_num_266375',
        'good/exp6_num_11357',
        'good/exp3_num_319346',
        'good/exp5_num_37084',
        'defective/exp1_num_249594',
        'defective/exp1_num_155300',
        'defective/exp1_num_4727',
        'defective/exp2_num_109256',
        'defective/exp3_num_116541',
        'defective/exp3_num_20409',
        'defective/exp3_num_33395',
        'defective/exp3_num_136351',
        'defective/exp4_num_284532',
        'defective/exp5_num_317558',
        'defective/exp6_num_116623',
        'defective/exp6_num_258703',
        'defective/exp2_num_36302',
        'defective/exp2_num_304305',
        'defective/exp2_num_271384',
        'defective/exp2_num_4964',
        'defective/exp2_num_291023',
        'defective/exp4_num_186909',
        'defective/exp4_num_136676',
        'defective/exp4_num_32185',
        'defective/exp5_num_54331',
        'defective/exp4_num_342190',
        'defective/exp6_num_304361',
        'defective/exp6_num_194304',
        'defective/exp6_num_356817',
        'defective/exp6_num_258703',
        'defective/exp5_num_108791',
        'defective/exp5_num_242089'
    ],
    64: [
        'good/exp4_num_311355',
        'good/exp6_num_266375',
        'good/exp6_num_11357',
        'good/exp3_num_319346',
        'good/exp5_num_37084',
        'defective/exp1_num_249594',
        'defective/exp1_num_155300',
        'defective/exp1_num_4727',
        'defective/exp0_num_461',
        'defective/exp1_num_3667',
        'defective/exp1_num_85781',
        'defective/exp1_num_271361',
        'defective/exp1_num_339819',
        'defective/exp1_num_346311',
        'defective/exp1_num_356718',
        'defective/exp2_num_109256',
        'defective/exp2_num_3696',
        'defective/exp2_num_7009',
        'defective/exp2_num_108923',
        'defective/exp2_num_116527',
        'defective/exp2_num_98102',
        'defective/exp2_num_40424',
        'defective/exp2_num_129056',
        'defective/exp2_num_241913',
        'defective/exp3_num_116541',
        'defective/exp3_num_20409',
        'defective/exp3_num_33395',
        'defective/exp3_num_136351',
        'defective/exp3_num_270169',
        'defective/exp3_num_271400',
        'defective/exp3_num_172286',
        'defective/exp3_num_135593',
        'defective/exp3_num_274138',
        'defective/exp4_num_284532',
        'defective/exp5_num_317558',
        'defective/exp6_num_116623',
        'defective/exp6_num_258703',
        'defective/exp2_num_36302',
        'defective/exp2_num_304305',
        'defective/exp2_num_271384',
        'defective/exp2_num_4964',
        'defective/exp2_num_291023',
        'defective/exp4_num_186909',
        'defective/exp4_num_136676',
        'defective/exp4_num_32185',
        'defective/exp4_num_297524',
        'defective/exp4_num_291052',
        'defective/exp4_num_304328',
        'defective/exp4_num_356767',
        'defective/exp4_num_241939',
        'defective/exp4_num_77613',
        'defective/exp5_num_54331',
        'defective/exp5_num_291077',
        'defective/exp5_num_274174',
        'defective/exp5_num_352540',
        'defective/exp4_num_342190',
        'defective/exp6_num_304361',
        'defective/exp6_num_194304',
        'defective/exp6_num_356817',
        'defective/exp6_num_139916',
        'defective/exp6_num_313568',
        'defective/exp6_num_270255',
        'defective/exp5_num_108791',
        'defective/exp5_num_242089'
    ]
}

class MTDataset(BinaryF2P_NetDataset):

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
        super(MTDataset, self).__init__(
            dataset_config=json_config, train_flag=train_flag,
            label_threshold=254, object_connectivity=8,
            area_threshold=20, relative_threshold=True,
            **super_args
        )