"""
## Torchvision dataset
data/
│
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── 3/
│
├── val/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── 3/
│
└── test/
    ├── 0/
    ├── 1/
    ├── 2/
    └── 3/

"""

"""
### Yolo dataset
yolo_dataset/
│
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img3.jpg
│   │   ├── img4.jpg
│   │   └── ...
│    └── test/
│   │   ├── img5.jpg
│   │   ├── img6.jpg
│   │   └── ...
│   │
└──labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   ├── val/
│   │   ├── img3.txt
│   │   ├── img4.txt
│   │   └── ...
│    └── test/
│   │   ├── img5.txt
│   │   ├── img6.txt
│   │   └── ...
"""

import os
import cv2
import tqdm

def extract_info(line):
    info = line.split(" ")
    label = int(info[0])
    cx, cy = float(info[1]), float(info[2])
    w, h = float(info[3]), float(info[4][:-1])

    label_info = {
        "label": label,
        "cx": cx,
        "cy": cy,
        "width": w,
        "height": h
    }
    return label_info

def coord_converter(label_info, shape):
    w = label_info["width"]*shape[1]
    h = label_info["height"]*shape[0]
    cx = label_info["cx"]*shape[1]
    cy = label_info["cy"]*shape[0]

    x1, y1 = int(cx) - int(w/2), int(cy) - int(h/2)
    x2, y2 = int(cx) + int(w/2), int(cy) + int(h/2)

    coords = [x1,y1,x2,y2]
    return coords

yolo_dataset_path = ''
output_dataset_path = ''
cnt=0

os.makedirs(output_dataset_path, exist_ok=True)

splits = ['train', 'val', 'test']
for split in splits:
    images_dir = os.path.join(yolo_dataset_path, 'images', split)
    labels_dir = os.path.join(yolo_dataset_path, 'labels', split)
    
    for image_name in tqdm.tqdm(os.listdir(images_dir)):
        if not image_name.endswith('.bmp'):
            continue
        
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name.replace('.bmp', '.txt'))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        height, width, _ = image.shape
        
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
    
                label_info = extract_info(line)
                coords = coord_converter(label_info, image.shape)

                x_min, y_min, x_max, y_max = coords 
                class_id = label_info["label"]
                
                cropped_object = image[y_min:y_max, x_min:x_max]
                
                class_dir = os.path.join(output_dataset_path, split, f'{int(class_id)}')
                os.makedirs(class_dir, exist_ok=True)
                
                object_filename = f'{image_name.replace(".bmp", "")}_obj_{str(cnt)}.bmp'
                cnt+=1
                object_path = os.path.join(class_dir, object_filename)
                try:
                    cv2.imwrite(object_path, cropped_object)
                except:
                    print("Error while saving...")

        print(f'Processed {image_name} for {split} split.')

print("Dataset processing completed.")
