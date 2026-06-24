from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp
import torch


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

jacard = JaccardIndex(threshold=0.5, task="binary", average="none")
optim = torch.optim.AdamW(model.parameters(), 1e-2)
