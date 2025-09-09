import os
import json



def write(lines, path):
    assert isinstance(lines, list)
    with open(path, "w+", encoding='utf-8') as f:
        f.write('\n'.join(lines))

def convert_json_format(input_file, output_file):

    with open(input_file, 'r') as f:
        lines = f.readlines()


    output_data = {}

    for line in lines:
        record = json.loads(line.strip())


        file_name = "/".join(record["image_path"].split("/")[-4:]).replace(".PNG", "")

 
        output_data[file_name] = record


    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    is_train_sample = False  # True indicates a training sample; False indicates a test sample.

    root = '/home/lenovo/yankerong/F2P_Net/data/DAGM2007'  # Replace with the actual file location
    save_root = '/home/lenovo/yankerong/F2P_Net/data/DAGM2007'  # Replace with the actual file location
    os.makedirs(save_root, exist_ok=True)

    if is_train_sample:
        save_path = os.path.join(save_root, "train.json")
    else:
        save_path = os.path.join(save_root, "test.json")

    all_samples = []

    for class_index in range(1, 11):  
        class_name = f"Class{class_index}"

        if is_train_sample:
            samples = []
            train_root = os.path.join(root, class_name, 'Train')  # /data/DAGM2007/ClassX/Train
            image_classes = os.listdir(train_root)

            for image_class in image_classes:
                if image_class == 'image':
                    for image in os.listdir(os.path.join(train_root, image_class)):
                        sample = {
                            "image_path": os.path.join(class_name, 'Train', 'image', image),
                            "label": 0,
                            "label_name": "good",
                            "object_name": class_name
                        }
                        samples.append(sample)
                elif image_class == 'defective':
                    for image in os.listdir(os.path.join(train_root, 'defective')):
                        mask_root = os.path.join(class_name, 'Train', 'Label', image.replace('.png', '.png'))
                        sample = {
                            "image_path": os.path.join(class_name, 'Train', 'defective', image),
                            "label": 1,
                            "label_name": "defective",
                            "object_name": class_name,
                            "mask_path": mask_root
                        }
                        samples.append(sample)

        else:
            samples = []
            test_root = os.path.join(root, class_name, 'Test') 
            image_classes = os.listdir(test_root)

            for image_class in image_classes:
                if image_class == 'image':
                    for image in os.listdir(os.path.join(test_root, image_class)):
                        sample = {
                            "image_path": os.path.join(class_name, 'Test', 'image', image),
                            "label": 0,
                            "label_name": "good",
                            "object_name": class_name
                        }
                        samples.append(sample)
                elif image_class == 'defective':
                    for image in os.listdir(os.path.join(test_root, image_class)):
                        mask_root = os.path.join(class_name, 'Test', 'Label', image.replace('.png', '.png'))
                        sample = {
                            "image_path": os.path.join(class_name, 'Test', image_class, image),
                            "label": 1,
                            "label_name": "defective",
                            "object_name": class_name,
                            "mask_path": mask_root
                        }
                        samples.append(sample)

       
        all_samples.extend(samples)

 
    all_samples = [json.dumps(sample) for sample in all_samples]
    write(all_samples, save_path)


    if is_train_sample:
        input_file = "/home/lenovo/yankerong/F2P_Net/data/DAGM2007/train.json"  # Replace with the actual file location
        output_file = "/home/lenovo/yankerong/F2P_Net/data/DAGM2007/train.json"  # Replace with the actual file location
    else:
        input_file = "/home/lenovo/yankerong/F2P_Net/data/DAGM2007/test.json"  # Replace with the actual file location
        output_file = "/home/lenovo/yankerong/F2P_Net/data/DAGM2007/test.json"  # Replace with the actual file location

    convert_json_format(input_file, output_file)

