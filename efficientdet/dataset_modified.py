import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from .utils import SegmentationMask
from utils.coco_category import convert_to_coco_category


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.contiguous_category_id_to_json_id = convert_to_coco_category()
        self.json_category_id_to_contiguous_id = {
            v: k for k, v in self.contiguous_category_id_to_json_id.items()
        }
        # self.json_category_id_to_contiguous_id = {
        #     v: i for i, v in enumerate(self.coco.getCatIds())
        # }
        #
        # self.contiguous_category_id_to_json_id = {
        #     v: k for k, v in self.json_category_id_to_contiguous_id.items()
        # }

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        
        img = self.load_image(idx)
        id = self.image_ids[idx]
        annot, mask = self.load_annotations(idx)
        mask = SegmentationMask(mask, img.shape[:2])
        sample = {'img': img, 'annot': annot, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        sample.update({"id": id})
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        masks = []

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations, masks

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

            masks += [a["segmentation"]]

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations, masks


def collater(data):
    data = data
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    mask = [s['mask'] for s in data]
    id = [s['id'] for s in data]


    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)
    num_annots = [ annot.shape[0] for annot in annots]

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)
    # out_sample = {'img': imgs, 'annot': annot_padded, 'scale': scales, "mask": mask, "num": num_annots, "id": id}
    return {'img': imgs, 'annot': annot_padded, 'scale': scales, "mask": mask, "num": num_annots, "id": id}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots, masks = sample['img'], sample['annot'], sample["mask"]
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        if annots.shape[0] == 0:
            return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots),
                'mask': masks, 'scale': scale}

        annots[:, :4] *= scale
        masks = masks.resize((resized_height, resized_width))
        masks = masks.resize_img((self.img_size, self.img_size))

        if masks.size[0] != new_image.shape[0] or masks.size[1] != new_image.shape[1]:
            print("mask:", masks.size)
            print("image:", image.shape[:2])
        # resize_result = {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots),
        #         'mask': masks, 'scale': scale}
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots),
                'mask': masks, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots, masks = sample['img'], sample['annot'], sample['mask']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            if annots.shape[0] == 0:
                sample = {'img': image, 'annot': annots, 'mask': masks}
                return sample

            #x_flip augmenter
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            # masks.polygons = masks.transpose(0)
            for i, polygon in enumerate(masks.polygons):
                masks.polygons[i] = polygon.transpose(0)

            sample = {'img': image, 'annot': annots, 'mask': masks}
        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots, masks = sample['img'], sample['annot'], sample['mask']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, "mask": masks}
