from torchvision.models import resnet18
from torch import nn


def create_model(n_in, out):
    model = resnet18()
    model.conv1 = nn.Conv2d(n_in, 64, kernel_size=(
        7, 7), stride=(1, 1), padding=(3, 3))
    model.fc = nn.Linear(512, out)
    return model
