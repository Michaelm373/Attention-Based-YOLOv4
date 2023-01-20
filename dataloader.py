import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, sampler
import os
import torch
from PIL import Image, ImageFile
from utils import (cells_to_bboxes,
    iou_width_height,
    non_max_suppression,
    plot_image)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        # annotiations is either the train or test cvs
        # these are practically a dictionary which shows which image corresponds to which label
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        # a sum of the images with the different scales (self.S)
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        # same threshold as used in the paper
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # images are on the left half and labels are on the right
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        # np.roll takes the final elements and rolls them into the begining, used to make it compatable with albumentations
        # bboxes gets the label (object x y w h), ndmin means ouput must have at least 2 dims
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        # opens image, converts to rgb, makes it numpy array for albumentations
        image = np.array(Image.open(img_path).convert("RGB"))

        # for albumentations
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # [prob_object, x, y, w, h, class] is why we have 6 as the final dimension
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]
        #### left off here ###
        for box in bboxes:
            # calculates iou for width and height with the all anchors
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            # which anchors had the highest iou is th ecorrect one for that box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            # used to make sure each of the 3 scales have an object and bounding box
            has_anchor = [False] * 3  # each scale should have one anchor
            # goes through the anchors
            for anchor_idx in anchor_indices:
                # scale_idx is the scale we want
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')
                # anchor on scale is which anchor on the scale we want
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # gets scals
                S = self.S[scale_idx]
                # gets the x and y of which cell it belongs to
                i, j = int(S * y), int(S * x)
                # from targets, takes the scale and which anchor on that scale and the i, j
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                # if the anchor isn't already taken
                if not anchor_taken and not has_anchor[scale_idx]:
                    # set the target on the particular scale to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # finds x, y, w, h targets
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                # if the anchor box is not taken and its larger than the ignore threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, list(targets)


def get_loaders(train_csv_path, test_csv_path, img_dir, label_dir, anchors, transforms, im_size=416, bs=8, num_work=2,
                valid_size=0.2):
    train_dataset = YOLODataset(
        train_csv_path,
        transform=transforms,
        S=[im_size // 32, im_size // 16, im_size // 8],
        img_dir=img_dir,
        label_dir=label_dir,
        anchors=anchors)

    test_dataset = YOLODataset(
        test_csv_path,
        transform=transforms,
        S=[im_size // 32, im_size // 16, im_size // 8],
        img_dir=img_dir,
        label_dir=label_dir,
        anchors=anchors)

    # splits train and validation data
    num_total = len(train_dataset)
    indices = list(range(num_total))
    np.random.shuffle(indices)
    split = int(np.floor(num_total * valid_size))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=num_work,
        drop_last=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        num_workers=num_work,
        shuffle=False,
        drop_last=False)

    eval_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        sampler=valid_sampler,
        num_workers=num_work,
        drop_last=False)

    return train_loader, test_loader, eval_loader

if __name__ == "__main__":
    classes = ["aeroplane","bicycle","bird","boat","bottle","bus",
    "car","cat", "chair","cow","diningtable","dog","horse","motorbike",
    "person","pottedplant","sheep","sofa","train","tvmonitor"]
    
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]]
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    train_path = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/train.csv"
    test_path = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/test.csv"
    im_dir = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/images"
    label_dir = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/labels"

    dataset = YOLODataset(train_path, test_path, im_dir, label_dir, anchors)

    img_dim = 416
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=img_dim),
            A.PadIfNeeded(
                min_height=img_dim, min_width=img_dim, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    train_loader, test_loader, train_eval_loader = get_loaders(train_path, test_path, im_dir, label_dir, anchors,
                                                               test_transforms, bs=1)

    for x, y in train_loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = non_max_suppression(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes, classes)
