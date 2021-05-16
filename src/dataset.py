"""
@author: Viet Nguyen (nhviet1009@gmail.com)
"""
import torch
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
import os
from torch.utils.data import Dataset
from PIL import Image

def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class CocoDataset(CocoDetection):
    def __init__(self, root, year, mode, transform=None):
        annFile = os.path.join(root, "annotations", "instances_{}{}.json".format(mode, year))
        root = os.path.join(root, "{}{}".format(mode, year))
        super(CocoDataset, self).__init__(root, annFile)
        self._load_categories()
        self.transform = transform

    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
        image, target = super(CocoDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        return image, target[0]["image_id"], (height, width), boxes, labels


class MAVIdataset(Dataset):
    def __init__(self, root, setname='train', transform=None):
        self.root = root 
        self.setname = setname
        with open(os.path.join(self.root, self.setname + '_annotations_roundup.json'), 'r') as f:
            self.data = json.load(f)
        self.images = self.data['images']
        self.annotations = self.data['annotations']

        assert len(self.images) == len(self.annotations)
        super(MAVIdataset,self).__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        self.impath[item] = os.path.join(self.root, self.setname, self.images[item]['filename']) 
        image = Image.open(self.impath[item], mode = 'r')
        width, height = image.size
        boxes = []
        labels = []
        target = self.annotations[item]
        if len(target) == 0:
            return None, None, None, None, None
        for annotation in target: 
            bbox = annotation['bbox'] 
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height]) # normalising the coordinates 
            labels.append(0) # since only one class of polygon(text) so...
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        return image, target[0]["name"], (height, width), boxes, labels