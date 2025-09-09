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

 
        file_name = "/".join(record["image_path"].split("/")[-2:]).replace(".jpg", "")


        output_data[file_name] = record


    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    


if __name__ == '__main__':

    is_train_sample = False  # True indicates a training sample; False indicates a test sample.


    root = '/home/lenovo/yankerong/F2P_Net/data/MT' # Replace with the actual file location
    class_name = 'MT' # Sample category

    save_root = '/home/lenovo/yankerong/F2P_Net/data/MT'# Replace with the actual file location
    os.makedirs(save_root, exist_ok=True)

    

    if is_train_sample:
        save_path = os.path.join(save_root, f"train.json")
    else:
        save_path = os.path.join(save_root, f"test.json")

    if is_train_sample:
        samples = []
        train_root = os.path.join(root,class_name,'train')  # /home/lenovo/yankerong/F2P_Net/data/MT/MT/train
        image_classes = os.listdir(train_root) # ['good', 'defective']

        for image_class in image_classes:
            if image_class=='good':
                for image in os.listdir(os.path.join(train_root,image_class)):
                    mask_root = os.path.join(class_name, 'ground_truth',image.replace('.jpg', '.png'))
                    sample = {
                            "image_path": os.path.join(class_name, 'train', 'good', image),
                            "label": 0,
                            "label_name": "good",
                            "object_name": class_name,
                            "mask_path": mask_root
                        }

                    samples.append(sample)
            else:
                for image in os.listdir(os.path.join(train_root, 'defective')):
                    mask_root = os.path.join(class_name, 'ground_truth',image.replace('.jpg', '.png'))

                    sample = {
                                    "image_path": os.path.join(class_name, 'train', 'defective', image),
                                    "label": 1,
                                    "label_name": "defective",
                                    "object_name": class_name,
                                    "mask_path": mask_root
                                }
                    samples.append(sample)

            samples.append(sample)




    else:
        samples = []
        test_root = os.path.join(root, class_name, 'test')
        image_classes = os.listdir(test_root)

        for image_class in image_classes:

            for image in os.listdir(os.path.join(test_root,image_class)):
                if image_class=='good':
                    mask_root = os.path.join(class_name, 'ground_truth', image.replace('.jpg', '.png'))
                    sample = {
                        "image_path": os.path.join(class_name, 'test', 'good', image),
                        "label": 0,
                        "label_name": "good",
                        "object_name": class_name,
                        "mask_path": mask_root
                    }
                else:
                    mask_root = os.path.join(class_name, 'ground_truth', image.replace('.jpg', '.png'))

                    sample = {
                        "image_path": os.path.join(class_name, 'test', image_class, image),
                        "label": 1,
                        "label_name": "defective",
                        "object_name": class_name,
                        "mask_path": mask_root
                    }

                samples.append(sample)

    samples=[json.dumps(sample) for sample in samples]
    write(samples,save_path)


 
    if is_train_sample:
        input_file = "/home/lenovo/yankerong/F2P_Net/data/MT/train.json"  # Replace with the actual file location
        output_file = "/home/lenovo/yankerong/F2P_Net/data/MT/train.json"  # Replace with the actual file location
    else:
        input_file = "/home/lenovo/yankerong/F2P_Net/data/MT/test.json"  # Replace with the actual file location
        output_file = "/home/lenovo/yankerong/F2P_Net/data/MT/test.json"  # Replace with the actual file location

    convert_json_format(input_file, output_file)
