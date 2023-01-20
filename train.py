import torch
from torch import optim
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from model import AS_YOLO
from dataloader import get_loaders
from loss import YoloLoss
from utils import(
    check_class_accuracy,
    plot_examples,
    get_evaluation_bboxes,
    mean_average_precision)

def train_pass(train_loader, model, optimizer, loss_fn, scaled_anchors, device):
    losses = []
    model.train()
    x, y = train_loader[0], train_loader[1]

    with torch.autograd.set_detect_anomaly(True):
        if device == 'cuda':
            x = x.cuda()

        out = model(x)
        y0, y1, y2 = y[2], y[1], y[0]
        if device == 'cuda':
            y0, y1, y2 = y0.cuda(), y1.cuda(), y2.cuda()
            scaled_anchors[0], scaled_anchors[1], scaled_anchors[2] = scaled_anchors[0].cuda(), scaled_anchors[
                1].cuda(), scaled_anchors[2].cuda()

        loss = (
                loss_fn(out[0], y0, scaled_anchors[0])  # .cuda())
                + loss_fn(out[1], y1, scaled_anchors[1])  # .cuda())
                + loss_fn(out[2], y2, scaled_anchors[2])  # .cuda())
        )

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = sum(losses) / len(losses)

        # clears cache
        gc.collect()
        torch.cuda.empty_cache()

    return mean_loss


def validation_pass(val_loader, model, loss_fn, scaled_anchors, device):
    losse = []
    model.eval()
    x, y = val_loader[0], val_loader[1]

    with torch.autograd.set_detect_anomaly(True):
        if device == 'cuda':
            x = x.cuda()

        # with torch.autograd.set_detect_anomaly(True):
        out = model(x)

        y0, y1, y2 = y[2], y[1], y[0]
        if device == 'cuda':
            y0, y1, y2 = y0.cuda(), y1.cuda(), y2.cuda()
            scaled_anchors[0], scaled_anchors[1], scaled_anchors[2] = scaled_anchors[0].cuda(), scaled_anchors[
                1].cuda(), scaled_anchors[2].cuda()

        loss = (
                loss_fn(out[0], y0, scaled_anchors[0])  # .cuda())
                + loss_fn(out[1], y1, scaled_anchors[1])  # .cuda())
                + loss_fn(out[2], y2, scaled_anchors[2])  # .cuda())
        )

        losse.append(loss.item())
        valid_loss = sum(losse) / len(losse)

        # clears cache
        gc.collect()
        torch.cuda.empty_cache()

    return valid_loss


def train(train_loader, eval_loader, model, optimizer, scheduler, loss_fn, anchors, device,
          train_pass, val_pass, epochs, labels, val_loss_min=float('inf'), save_path="/kaggle/working/AS_YOLO_5.pt"):
    losses = []
    val_losses = []

    S = [img_dim // 8, img_dim // 16, img_dim // 32]
    scaled_anchors = torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)

    trainloader = iter(train_loader)
    evalloader = iter(eval_loader)
    testloader = iter(test_loader)

    for epoch in range(epochs):
        try:
            train_data = trainloader.next()
        except:
            trainloader = iter(train_loader)
            train_data = trainloader.next()
        try:
            eval_data = evalloader.next()
        except:
            evalloader = iter(eval_loader)
            eval_data = evalloader.next()

        # runs through training and evaluation and prints results
        mean_loss_train = train_pass(train_data, model, optimizer, loss_fn, scaled_anchors, device)
        with torch.no_grad():
            val_loss = val_pass(eval_data, model, loss_fn, scaled_anchors, device)
        learning_rate = scheduler.get_last_lr()[0]
        print("epoch:", epoch, "\ttrain loss:", mean_loss_train, "\t valid loss:", val_loss, "\tlearning rate:",
              learning_rate)

        # checks classs accuracy and mAP every 5 epochs
        if epoch > 0 and epoch % 10 == 0:
            test_data = testloader.next()
            model.eval()
            class_acc, noobj_acc, obj_acc = check_class_accuracy(model, test_data, threshold=0.05, device=device)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_data,
                model,
                iou_threshold=0.45,
                anchors=anchors,
                threshold=0.05,
                device=device,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=0.5,
                box_format="midpoint",
                num_classes=20,
            )
            print()
            print(f"Class accuracy: {class_acc}")
            print(f"No obj accuracy: {noobj_acc}")
            print(f"Obj accuracy: {obj_acc}")
            print(f"Mean Average Percision: {mapval}")
            print()
            model.train()

        # plots a couple examples from the test loader every 100 epochs
        if epoch > 0 and epoch % 100 == 0:
            plot_examples(model, test_data, thresh=0.4, iou_thresh=0.5, anchors=scaled_anchors, device=device,
                                 labels=labels)

        losses.append(mean_loss_train)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss <= val_loss_min:
            torch.save(model.state_dict(), save_path)
            val_loss_min = val_loss
            print('Saving Model...')
            print()
        else:
            print()

    return losses, val_losses



if __name__ == "__main__":
    # model and loss
    img_dim = 416
    
    # loads AS_YOLO with the pretrained vgg19 weights
    model = AS_YOLO(pretrained=True)
    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if device == 'cuda':
        model.cuda()
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.pannet.parameters(), lr=0.01, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)

    # dataset
    anchors = [
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]
    ]
    S = [img_dim // 8, img_dim // 16, img_dim // 32]
    scaled_anchors = torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)

    train_path = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/train.csv"
    test_path = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/test.csv"
    im_dir = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/images"
    label_dir = "/kaggle/input/pascal-voc-yolo-works-with-albumentations/PASCAL_VOC/labels"

    # data augmentations so the final model will perform better
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(img_dim)),
            A.PadIfNeeded(
                min_height=int(img_dim),
                min_width=int(img_dim),
                border_mode=int(cv2.BORDER_CONSTANT),
            ),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=int(cv2.BORDER_CONSTANT)
                    ),
                    A.Affine(shear=15, p=0.5, mode="constant"),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
    )

    train_loader, test_loader, eval_loader = get_loaders(train_path, test_path, im_dir, label_dir, anchors,
                                                         train_transforms)


    train_losses, val_losses = train(train_loader, eval_loader, test_loader, model, optimizer, scheduler, loss_fn, anchors, device,
                                     train_pass, validation_pass, epochs=1001, labels=classes)

    # plots the loss throughout training
    x_axis = list(range(len(train_losses)))
    ax = plt.gca()
    ax.set_ylim([0, 100])
    ax.set_xlim([50,epochs])
    plt.plot(x_axis, train_losses, label="Training Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.show()
