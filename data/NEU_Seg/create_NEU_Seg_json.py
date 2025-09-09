import os
import json

def write(lines, path, split=None):
    assert type(lines) == list
    for i in range(len(lines)):
        if type(lines[i]) == list:
            if split is None:
                lines[i] = " ".join([str(now) for now in lines[i]])
            else:
                lines[i] = split.join([str(now) for now in lines[i]])
        else:
            lines[i] = str(lines[i])
    with open(path, "w+", encoding='utf-8') as f:
        f.write('\n'.join(lines))


def convert_json_format(input_file, output_file):

    with open(input_file, 'r') as f:
        lines = f.readlines()


    output_data = {}

    for line in lines:
        record = json.loads(line.strip())


        file_name = "/".join(record["image_path"].split("/")[-2:]).replace(".png", "")

 
        output_data[file_name] = record

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    


if __name__ == '__main__':

    is_train_sample = False  # True indicates a training sample; False indicates a test sample.


    root = '/home/lenovo/yankerong/F2P_Net/data/NEU_Seg' # Replace with the actual file location
    class_name = 'NEU_Seg' # Sample category

    save_root = '/home/lenovo/yankerong/F2P_Net/data/NEU_Seg'# Replace with the actual file location
    os.makedirs(save_root, exist_ok=True)

    

    if is_train_sample:
        save_path = os.path.join(save_root, f"train.json")
    else:
        save_path = os.path.join(save_root, f"test.json")

    if is_train_sample:
        samples = []
        train_root = os.path.join(root,class_name)  
        image_classes = os.listdir(train_root) # ['train', 'test', 'ground_truth']

        for image_class in image_classes:
            if image_class=='train':
                for image in os.listdir(os.path.join(train_root,image_class)):
                    mask_root = os.path.join(class_name, 'ground_truth', image_class, image.replace('.jpg', '.png'))
                    sample = {
                            "image_path": os.path.join(class_name,'train', image),
                            "label": 1,
                            "label_name": "defective",
                            "object_name": class_name,
                            "mask_path": mask_root
                        }

                    samples.append(sample)

            




    else:
        samples = []
        test_root = os.path.join(root, class_name) # /home/lenovo/yankerong/F2P_Net/data/NEU_Seg/NEU_Seg
        image_classes = os.listdir(test_root) # ['train', 'test', 'ground_truth']

        for image_class in image_classes:
            if image_class=='test':
                for image in os.listdir(os.path.join(test_root,image_class)):
                
                    mask_root = os.path.join(class_name, 'ground_truth', image_class, image.replace('.jpg', '.png'))
                    sample = {
                        "image_path": os.path.join(class_name,'test', image),
                        "label": 1,
                        "label_name": "defective",
                        "object_name": class_name,
                        "mask_path": mask_root
                    }
                    samples.append(sample)

    samples=[json.dumps(sample) for sample in samples]
    write(samples,save_path)



    if is_train_sample:
        input_file = "/home/lenovo/yankerong/F2P_Net/data/NEU_Seg/train.json"  # Replace with the actual file location
        output_file = "/home/lenovo/yankerong/F2P_Net/data/NEU_Seg/train.json"  # Replace with the actual file location
    else:
        input_file = "/home/lenovo/yankerong/F2P_Net/data/NEU_Seg/test.json"  # Replace with the actual file location
        output_file = "/home/lenovo/yankerong/F2P_Net/data/NEU_Seg/test.json"  # Replace with the actual file location

    convert_json_format(input_file, output_file)
