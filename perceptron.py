import torch
import torch.nn as nn

class PerceptronOVR(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_classes, input_dim))
        self.biases  = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x @ self.weights.t() + self.biases

def load_perceptron(path="perceptron_ova_mnist.pth", device="cpu"):
    model = PerceptronOVR().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
