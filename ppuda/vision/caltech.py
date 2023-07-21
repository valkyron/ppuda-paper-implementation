# import os
# import hashlib
# import xml.etree.ElementTree as ET
# from collections import defaultdict
# from PIL import Image
# import torch.utils.data as data


# class Caltech101Dataset(data.Dataset):
#     def __init__(self, root, split='train', transform=None):
#         self.root = root
#         self.split = split
#         self.transform = transform

#         assert self.split in ['train', 'val'], f"Invalid split: {self.split}"

#         self.classes = os.listdir(os.path.join(root, '101_ObjectCategories'))
#         self.classes.sort()

#         self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
#         self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}

#         self.imgs = self._make_dataset()

#     def _make_dataset(self):
#         img_list = []
#         annotation_dir = os.path.join(self.root, 'Annotations')

#         for target_class in self.classes:
#             class_dir = os.path.join(self.root, '101_ObjectCategories', target_class)
#             for filename in os.listdir(class_dir):
#                 if filename.endswith('.jpg') or filename.endswith('.jpeg'):
#                     img_path = os.path.join(class_dir, filename)
#                     annotation_path = os.path.join(annotation_dir, target_class, f"{filename.split('.')[0]}.xml")
#                     img_list.append((img_path, target_class, annotation_path))

#         if self.split == 'train':
#             img_list = [item for item in img_list if 'BACKGROUND_Google' not in item[0]]

#         return img_list

#     def __getitem__(self, index):
#         img_path, target, annotation_path = self.imgs[index]

#         with open(annotation_path, 'r') as f:
#             annotation = f.read()

#         xml = ET.fromstring(annotation)

#         # get bounding box coordinates
#         objects = xml.findall('object')
#         bbox = None
#         for obj in objects:
#             if obj.find('name').text == target:
#                 bbox = obj.find('bndbox')
#                 break
#         assert bbox is not None, f"Could not find bounding box for {img_path}"

#         x1 = int(bbox.find('xmin').text)
#         y1 = int(bbox.find('ymin').text)
#         x2 = int(bbox.find('xmax').text)
#         y2 = int(bbox.find('ymax').text)

#         # crop image using bounding box
#         img = Image.open(img_path).convert('RGB')
#         img = img.crop((x1, y1, x2, y2))

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, self.class_to_idx[target]

#     def __len__(self):
#         return len(self.imgs)



# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------

# import os
# from collections import defaultdict
# import torchvision
# import hashlib
# from torchvision.datasets.vision import VisionDataset
# # from torchvision.datasets import Caltech101


# def split_train_and_val(list_of_tups, num_val_per_class=50):

#     class_dict = defaultdict(list)

#     for item in list_of_tups:
#         class_dict[item[1]].append(item)

#     train_samples, val_samples = [], []

#     # fix the class ordering
#     for k in sorted(class_dict.keys()):
#         v = class_dict[k]

#         # last num_val_per_class will be the val samples
#         train_samples.extend(v[:-num_val_per_class])
#         val_samples.extend(v[-num_val_per_class:])

#     return train_samples, val_samples

# class Caltech101Dataset(Dataset):

#     def __init__(self, root, split, transform=None, target_transform=None):
#         assert split in {"train", "val"}

#         # super().__init__(root, download=True, transform=transform, target_transform=target_transform)
#         # VisionDataset.__init__(self, root, transform=transform, target_transform=target_transform)
#         # Caltech101.__init__(self, root, download=True)

#         self.split = split

#         # if split == "train":
#         #     img_list = self.train_list
#         # elif split == "val":
#         #     img_list = self.test_list

#         # remove images from BACKGROUND_Google category
#         if self.split == 'train':
#             img_list = [item for item in img_list if 'BACKGROUND_Google' not in item[0]]

#         self.samples = [(os.path.join(self.root, img[0]), img[1]) for img in img_list]

#         if split == "train":
#             train_samples, _ = split_train_and_val(self.samples, num_val_per_class=20)
#             self.samples = train_samples
#         elif split == "val":
#             _, val_samples = split_train_and_val(self.samples, num_val_per_class=10)
#             self.samples = val_samples

#         m = hashlib.sha256()
#         m.update(str(self.samples).encode())

#         # convert from hex string to number
#         self.checksum = int(m.hexdigest(), 16)

#     @property
#     def num_examples(self):
#         return len(self)

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------


from torch.utils.data import Dataset
import hashlib
from PIL import Image

class Caltech101Dataset(Dataset):
    def __init__(self, split, images, labels=None, transforms=None):
        # assert split in {"train", "val"}
        self.transforms = transforms
        self.split = split
        # Create a list of tuples, where each tuple contains an image and its corresponding label
        
        self.samples = []
        print(self.split + ":" + str(len(labels)))
        for i, image in enumerate(images):
            label = labels[i] if labels is not None else None
            self.samples.append((image, label))
        m = hashlib.sha256()
        m.update(str(self.samples).encode())

        # convert from hex string to number
        self.checksum = int(m.hexdigest(), 16)

    @property
    def num_examples(self):
        return len(self)

    def __len__(self):
        return (len(self.samples))
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with open(image_path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label