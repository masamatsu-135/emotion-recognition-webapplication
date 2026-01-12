import torch
import torch.nn as nn
from torchvision import models


class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
            except AttributeError:
                backbone = models.resnet18(pretrained=True)

        else:
            try:
                backbone = models.resnet18(weights=None)
            except TypeError:
                backbone = models.resnet18(pretrained=False)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
    

def get_model(
        model_type="resnet", num_classes=7, pretrained=True, device="cpu"
        ):
    
    model_type = model_type.lower()
    if model_type == "resnet":
        model = EmotionResNet18(
            num_classes=num_classes,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}.")
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Test Resnet18 model ===")
    resnet = get_model(model_type="resnet", device=device, pretrained=False)
    resnet.eval()
    dummy_resnet = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out_resnet = resnet(dummy_resnet)
    print("ResNet Output shape:", out_resnet.shape)