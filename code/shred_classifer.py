'''
This script contains the script for training a CNN to identify whether the source is a shred or not!

During VI, we classify it as a shred, if the fracflux is indeed coming from the source it is part of/magnitude changes significantly > 0.25!
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ShredClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a ResNet18 but modify first conv layer to accept 6 channels
        self.backbone = resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final fc layer
        self.backbone.fc = nn.Identity()  # We'll handle it ourselves

        # Final classifier that takes (resnet features + 6 metadata) as input
        self.classifier = nn.Sequential(
            nn.Linear(512 + 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, images, metadata):
        x = self.backbone(images)  # (batch_size, 512)
        x = torch.cat([x, metadata], dim=1)  # (batch_size, 512+6)
        x = self.classifier(x)  # (batch_size, 1)
        return x.squeeze(1)  # make output (batch_size,)



from torch.utils.data import DataLoader
import torch.optim as optim

def train(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # binary cross-entropy loss

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, metadata, labels in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            preds = model(images, metadata)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")

def evaluate(model, val_loader, device='cuda'):
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for images, metadata, labels in val_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).float()

            preds = model(images, metadata)
            loss = criterion(preds, labels)

            total_loss += loss.item()

    return total_loss / len(val_loader)


from sklearn.metrics import roc_auc_score, accuracy_score

def final_eval(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).float()

            preds = model(images, metadata)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

    print(f"Test AUC: {auc:.4f} | Test Accuracy: {acc:.4f}")


def get_inputs(file_path, image_path):
    '''
    Given the input file, it reads the relevant files to get the relevant data

    file_path is the path to the galaxy folder to read the tractor model
    image_path is the path to the full fits file image

    the output of this is the 6 channel 64x64 image given to CNN
    '''


    return 

    


if __name__ == '__main__':

    ##load the dataset

    data = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_VI_labelled.fits")

    #get the labelled data set!
    data_label = data[(data["IS_SHRED_VI"] == "good") | (data["IS_SHRED_VI"] == "fragment")]



    tractor_model = np.load(save_path + "/tractor_source_model.npy")

        #we want to plot the original iamge 
        resis = data_arr - tractor_model 

    

