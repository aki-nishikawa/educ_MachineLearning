from re import A
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
		)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
		)
        self.layer3 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
		)
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
		)
        self.output = nn.Sequential(
            nn.Linear(64, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)
        return x