from tqdm.notebook import tqdm
from src.dataset import MyDataset
from src.model import optim, DiceLoss, model, jacard, device
import seaborn as sns
import os
import matplotlib.pyplot as plt
import torch




criterion = DiceLoss()
epochs = 40
avg_train_loss = []
avg_val_dice = []
avg_val_iou = []

for epoch in tqdm(range(epochs)):
  model.train()
  train_loss = 0

  for X_batch, Y_batch in tqdm(train_loader):
    X_batch = X_batch.to(device)
    Y_batch = Y_batch.to(device)
    Y_pred = model(X_batch)
    loss = criterion(Y_pred, Y_batch)
    train_loss += loss.item()

    optim.zero_grad()

    loss.backward()
    optim.step()

  avg_train_loss.append(train_loss/len(train_loader))

  model.eval()
  val_loss_dice = 0
  val_loss_iou = 0
  with torch.no_grad():
    for X_val, Y_val in val_loader:
      X_val = X_val.to(device)
      Y_val = Y_val.to(device)

      Y_pred = model(X_val)

      loss = criterion(Y_pred, Y_val)
      val_loss_dice += loss.item()

      val_loss_iou += jacard(torch.sigmoid(Y_pred), Y_val).to(device).item()



  avg_val_dice.append(val_loss_dice/len(val_loader))
  avg_val_iou.append(val_loss_iou/len(val_loader))
  print(f"Epoch:{epoch} | Train: {avg_train_loss[-1] :.6f} | Val: {avg_val_dice[-1]:.6f} | Val: {avg_val_iou[-1]:.6f}")



sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))


sns.lineplot(data=avg_train_loss, label="Train Dice Loss", marker="o")
sns.lineplot(data=avg_val_dice, label="Val Dice Loss", marker="s")
sns.lineplot(data=avg_val_iou, label="Val Jaccard (IoU)")


plt.title("История обучения модели", fontsize=16, fontweight="bold")
plt.xlabel("Эпоха", fontsize=12)
plt.ylabel("Значение", fontsize=12)

plt.legend() 
plt.show()

#PATH = ??
seg_test = sorted(os.listdir(PATH))
img_test = sorted(m for m in os.listdir(PATH) if m.replace(".PNG", "_label.PNG") in seg_test)

data_val = MyDataset(img_test, seg_test, PATH=PATH+"/", Train=False)
test_loader = torch.utils.data.DataLoader(data_val, batch_size=10, shuffle=False)

model.eval()

with torch.no_grad():
  jac = 0
  dice = 0
  for X_batch, Y_batch in test_loader:
    X_batch = X_batch.to(device)
    Y_batch = Y_batch.to(device)
    
    Y_pred = model(X_batch)
    loss_1 = jacard(torch.sigmoid(Y_pred), Y_batch)
    loss_2 = criterion(Y_pred,Y_batch)
    jac += loss_1.item()
    dice += loss_2.item()
  
  print(f"Test Jaccard: {jac/len(test_loader):.4f}\nTest Dice: {dice/len(test_loader):.4f}")

