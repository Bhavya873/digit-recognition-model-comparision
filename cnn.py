import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_ch, out_ch, pool=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 64,  pool=True),
            block(64, 128, pool=True),
            block(128,256, pool=False),
            block(256,256, pool=True),
            block(256,512, pool=False),
            block(512,512, pool=True),
            block(512,512, pool=False),
            block(512,512, pool=True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
