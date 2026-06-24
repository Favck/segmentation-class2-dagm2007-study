from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp
import torch

class DiceLoss(nn.Module):
  def __init__(self, smooth=1e-7):
    super().__init__()

    self.smooth = smooth

  def forward(self, output, targets):

    outputs = torch.sigmoid(output)

    outputs = outputs.view(-1)
    targets = targets.view(-1)

    interseption = (outputs * targets).sum()

    dice_score = (2. * interseption + self.smooth)/(outputs.sum() + targets.sum() + self.smooth)

    return 1 - dice_score
      

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
jacard = JaccardIndex(threshold=0.5, task="binary", average="none")
optim = torch.optim.AdamW(model.parameters(), 1e-2)
