import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class MyDataset(Dataset):
  def __init__(self, image_files, segment_files, PATH="/content/dagm_data/DAGM_KaggleUpload/Class2/Train/", Train=True) -> None:
    super().__init__()

    self.image_files = [PATH + i for i in image_files]
    self.segment_files = [PATH + "Label/" + i for i in segment_files]
    if Train:
      self.transform= v2.Compose(
          [
          v2.Resize((256, 256)),
          v2.RandomHorizontalFlip(p=0.5),
          v2.RandomVerticalFlip(p=0.5),
          v2.ToImage(),
          v2.ToDtype(torch.float32, scale=True),
          v2.Normalize(mean=[0.5], std=[0.5])
          ]
      )
    else:
      self.transform= v2.Compose(
          [
          v2.Resize((256, 256)),
          v2.ToImage(),
          v2.ToDtype(torch.float32, scale=True),
          v2.Normalize(mean=[0.5], std=[0.5])
          ]
      )



  def __getitem__(self, index):

    img = Image.open(self.image_files[index]).convert("L")
    mask = Image.open(self.segment_files[index]).convert("L")


    img, mask = self.transform(img, mask)

    mask = (mask>0.5).float()

    return (img, mask)

  def __len__(self):

    return len(self.image_files)
