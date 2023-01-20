import torch
from torch import nn, optim
from torchvision import transforms, datasets
from model import CBL
import math
import time
import gc
import matplotlib.pyplot as plt

  """
    In every paper about a YOLO based algorithm, the backbone (Darknet53) comes pre-trained.
    This file contains all the code used to train Darknet.
    In the final model, I opted for a pre-trained vgg-19 as the backbone because it is more space-efficient.
  """

class CSP(nn.Module):
    def __init__(self, conv_dim, res_dim):
        super(CSP, self).__init__()

        self.cbl1 = CBL(conv_dim1=conv_dim, conv_dim2=conv_dim, kernel_size=1)
        self.cbl2 = CBL(conv_dim1=conv_dim, conv_dim2=conv_dim, kernel_size=1)

        res_layers = []
        for i in range(res_dim):
            res_layers.append(ResBlock(conv_dim=conv_dim))
        self.residual = nn.Sequential(*res_layers)

        self.cbl3 = CBL(conv_dim1=conv_dim, conv_dim2=conv_dim, kernel_size=1)
        self.cbl4 = CBL(conv_dim1=conv_dim * 2, conv_dim2=conv_dim * 2, kernel_size=1)

    def forward(self, x):
        x1 = self.cbl1(x)

        x2 = self.cbl2(x)
        x2 = self.residual(x)
        x2 = self.cbl3(x2)

        x_out = torch.cat([x1, x2], dim=1)

        x_out = self.cbl4(x_out)
        return x_out

class Darknet_train(nn.Module):
    def __init__(self):
        super(Darknet_train, self).__init__()

        self.layers_1 = nn.Sequential(CBL(conv_dim1=3, conv_dim2=16, kernel_size=3),
                                      CBL(conv_dim1=16, conv_dim2=16, kernel_size=3, stride=2),
                                      CSP(conv_dim=16, res_dim=1),
                                      CBL(conv_dim1=32, conv_dim2=32, kernel_size=3, stride=2),
                                      CSP(conv_dim=32, res_dim=2),
                                      CBL(conv_dim1=64, conv_dim2=64, kernel_size=3, stride=2),
                                      CSP(conv_dim=64, res_dim=4))

        self.layers_2 = nn.Sequential(CBL(conv_dim1=128, conv_dim2=128, kernel_size=3, stride=2),
                                      CSP(conv_dim=128, res_dim=4))

        self.layers_3 = nn.Sequential(CBL(conv_dim1=256, conv_dim2=256, kernel_size=3, stride=2),
                                      CSP(conv_dim=256, res_dim=3))

        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(86528, 10)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x0 = x.shape[0]
        x1 = self.layers_1(x)
        x2 = self.layers_2(x1)
        x3 = self.layers_3(x2)

        out = x3.view(x0, -1)
        out = self.softmax(self.dropout(self.linear(out)))

        return [x1, x2, x3], out


def darknetDataset(test_dir, train_dir, valid_dir, transform=None, num_workers=2,
                   batch_size=20, img_dim=416):
    if not transform:
        transform = transforms.Compose([transforms.Resize((img_dim, img_dim)),
                                        transforms.ToTensor()])

    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    images, labels = iter(train_loader).next()
    images = images.numpy()

    assert images.shape == (batch_size, 3, img_dim, img_dim)
    assert len(labels) == batch_size

    return train_loader, valid_loader, test_loader


def trainLoopDarknet(network, optimizer, data_train, data_val, device):  # train_loss, valid_loss,
    # tracking loss
    train_loss = 0.0
    valid_loss = 0.0

    # keep track of time
    start_time = time.time()

    ### Training Loop ###
    # sets to training mode
    network.train()
    # gets data and switches to gpu
    images, labels = data_train[0], data_train[1]
    if device == 'cuda':
        images, labels = images.cuda(), labels.cuda()

    # zeros gradient and runs through model
    optimizer.zero_grad()
    _, out = network(images)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    # updates loss
    train_loss += loss.item() * images.size(0)

    # clears cache
    gc.collect()
    torch.cuda.empty_cache()

    ### Validation Loop ###
    # sets to validation mode
    darknet.eval()
    # gets data
    images, labels = data_val[0], data_val[1]
    # switches to gpu
    if device == 'cuda':
        images, labels = images.cuda(), labels.cuda()

    # runs the validation set
    with torch.no_grad():
        _, x = network(images)
        loss = criterion(x, labels)

    # updates loss
    valid_loss += loss.item() * images.size(0)

    # clears cache
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    epoch_duration = end_time - start_time

    return train_loss, valid_loss, epoch_duration


def DarknetTrain(network, optimizer, scheduler, train_loader, valid_loader, train_loop, device, lr, epochs=10,
                 valid_loss_min=None):
    if not valid_loss_min:
        valid_loss_min = 10000

    # for graphing
    x_axis = list(range(epochs))
    y_train = []
    y_val = []

    # prepares data
    trainloader = iter(train_loader)
    validloader = iter(valid_loader)

    # extracts data
    data_train = trainloader.next()
    data_val = validloader.next()

    buffer = 0
    for e in range(epochs):
         print("EPOCH: ", e+1)

        # training loop
        train_loss, valid_loss, epoch_duration = train_loop(network=network, optimizer=optimizer, data_train=data_train,
                                                            data_val=data_val,
                                                            device=device)  # train_loss=train_loss, valid_loss=valid_loss,

        # updates overall losses
        training_loss = train_loss / (data_train[0].shape[0])
        validation_loss = valid_loss / (data_val[0].shape[0])
        y_train.append(training_loss)
        y_val.append(validation_loss)

        print("Training Loss: {:.4f} \tValidation Loss: {:.4f} \n\t\tEpoch Time: {:.4f}".format(
                training_loss, validation_loss, epoch_duration))

        scheduler.step()

        if valid_loss <= valid_loss_min:
            torch.save(darknet.state_dict(), 'darknet.pt')
            valid_loss_min = valid_loss
            print('Saving Model...')
            print()
        else:
            print()


    return x_axis, y_train, y_val

if __name__ == "__main__":
    # prepares data
    train_loader_dn, valid_loader_dn, test_loader_dn = darknetDataset(
        test_dir="/kaggle/input/fast-food-classification-dataset/Fast Food Classification V2/Test",
        train_dir="/kaggle/input/fast-food-classification-dataset/Fast Food Classification V2/Train",
        valid_dir="/kaggle/input/fast-food-classification-dataset/Fast Food Classification V2/Valid")

    # defines model, loss function, optimizer, and whatever else is needed for training
    darknet = Darknet_train()
    criterion = nn.CrossEntropyLoss()
    optimizer_dark = optim.SGD(darknet.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer_dark, step_size=30, gamma=0.9)

    # clears cache
    torch.cuda.empty_cache()
    # sets processing to gpu
    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if device == 'cuda':
        darknet.cuda()

    x_axis, y_train, y_val = DarknetTrain(darknet, optimizer_dark, scheduler, train_loader_dn, valid_loader_dn,
                                          train_loop=trainLoopDarknet, device=device, lr=0.001, epochs=300)

    plt.plot(x_axis, y_train, label="Train Loss")
    plt.plot(x_axis, y_val, label="Validation Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.show()
