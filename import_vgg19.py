from torchvision import models
import torch
import numpy as np
from model import VGG19_backbone

if __name__ == "__main__":
    # imports pretrained model
    vgg19 = models.vgg19(pretrained=True)
    PATH = "/kaggle/working/backbone.pt"

    #instantiates the version we'll be using
    backbone = VGG19_backbone

    # cuts off the classification layers, turns the the gradient off
    vgg19_conv = vgg19.features
    vgg19_conv.eval()
    for name, param in vgg19_conv.named_parameters():
        param.requires_grad = False

    # converts state dicts so vgg19 can have 3 outputs and will be compatable with YOLO architecture
    state_dict = backbone.state_dct()
    bone = vgg19_conv.state_dict()
    trained_layers = [name for name, param in bone.items()]
    new_layers = [keys for keys, values in state_dict.items()]

    assert len(new_layers) - len(trained_layers) == 0

    # moves weights from pretrained model to the new one
    for i in range(len(new_layers)):
        state_dict[new_layers[i]] = bone[trained_layers[i]]

    # loads the weigths using that state dict
    backbone.load_state_dict(state_dict)

    weights_list = []
    bias_list = []

    # makes sure the weights are the same
    with torch.no_grad():
        for (key, value), (name, param) in zip(vgg19_conv.named_parameters(), backbone.named_parameters()):
            assert (value - param).numpy().all() == np.zeros(param.shape).all()

    # saves state dict so we can use a pretrained vgg19 when training the whole model
    torch.save(backbone.state_dict(), PATH)
