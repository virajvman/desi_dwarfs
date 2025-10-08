'''
This script contains the script for training a CNN to identify whether the source is a shred or not!

During VI, we classify it as a shred, if the fracflux is indeed coming from the source it is part of/magnitude changes significantly > 0.25!
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.model_selection import KFold
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, f1_score
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from astropy.table import Table
import numpy as np
from desi_lowz_funcs import print_maxabs_diff, match_c_to_catalog, sdss_rgb
import os
from astropy.io import fits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import matplotlib.gridspec as gridspec
from torchvision import transforms
from astropy.table import vstack

def find_best_threshold(y_true, y_scores):
    # Sweep thresholds from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, 101)
    
    acc_list = []
    f1_list = []
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        acc_list.append(accuracy_score(y_true, preds))
        f1_list.append(f1_score(y_true, preds))
    
    best_acc_thresh = thresholds[np.argmax(acc_list)]
    best_f1_thresh = thresholds[np.argmax(f1_list)]

    # Youden's J statistic (maximizes TPR - FPR)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    best_j_thresh = roc_thresh[np.argmax(youden_j)]

    print(f"Best threshold by accuracy: {best_acc_thresh:.2f}")
    print(f"Best threshold by F1:       {best_f1_thresh:.2f}")
    print(f"Best threshold by Youden's J (balanced TPR/FPR): {best_j_thresh:.2f}")

    # Calculate accuracy with the new threshold (using best accuracy threshold)
    final_preds_acc = (y_scores >= best_acc_thresh).astype(int)
    final_accuracy = accuracy_score(y_true, final_preds_acc)
    
    # Calculate accuracy with the new threshold (using best F1 threshold)
    final_preds_f1 = (y_scores >= best_f1_thresh).astype(int)
    final_accuracy_f1 = accuracy_score(y_true, final_preds_f1)
    
    # Calculate accuracy with the new threshold (using best Youden's J threshold)
    final_preds_j = (y_scores >= best_j_thresh).astype(int)
    final_accuracy_j = accuracy_score(y_true, final_preds_j)

    print(f"Final accuracy with best accuracy threshold: {final_accuracy:.2f}")
    print(f"Final accuracy with best F1 threshold:       {final_accuracy_f1:.2f}")
    print(f"Final accuracy with best Youden's J threshold: {final_accuracy_j:.2f}")
    
    return
    

def get_preds(model, loader, device='cuda'):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for images, metadata, _ in loader:
            images = images.to(device)
            metadata = metadata.to(device)

            preds = model(images, metadata)
            
            all_scores.extend(preds.cpu().numpy())

    return np.array(all_scores)


def get_pcnn_scores(model, dataset, device='cuda'):
    '''
    Function that computes the probability of being a fragment!
    '''
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)

            preds = model(images, metadata)  # pCNN scores (between 0 and 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_preds, all_labels



class ShredDataset(Dataset):
    def __init__(self, images, metadata_scaled, labels, indices, transform=None):
        self.images = images[indices]
        self.metadata_scaled = metadata_scaled[indices]
        self.labels = labels[indices]
        self.transform = transform  # save transform to apply later

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        metadata = self.metadata_scaled[idx]
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            # torchvision transforms expect (C, H, W) torch tensor
            image_tensor = torch.tensor(image, dtype=torch.float32)
            image = self.transform(image_tensor)  # apply transform
        else:
            image = torch.tensor(image, dtype=torch.float32)

        metadata = torch.tensor(metadata, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
   
        return image, metadata, label


class SmallShredCNN(nn.Module):
    def __init__(self, metadata_dim, img_channels=3, use_meta_fc=True, dropout_rate=0.3, num_cnn_layers=3, base_channels=16):
        '''
        img_channels = the number of filters in the input matrix
        use_mata_fc = should hte metadata be passed through a fully connected layer first?
        base_channels = the number of output channels in the first input layer
        '''
        super(SmallShredCNN, self).__init__()
        self.use_meta_fc = use_meta_fc
        self.num_cnn_layers = num_cnn_layers

        # Build CNN layers dynamically
        cnn_layers = []
        in_channels = img_channels
        for i in range(num_cnn_layers):
            out_channels = base_channels * (2 ** i)  # doubles channels each layer
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)  # downsample by 2
            bn = nn.BatchNorm2d(out_channels)
            cnn_layers += [conv, bn, nn.ReLU(inplace=True)]
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Compute flattened size after CNN dynamically (assuming input image is square and size 64x64)
        self.feature_map_size = 64 // (2 ** num_cnn_layers)  # e.g., 64 -> 32 -> 16 -> 8
        self.flatten_dim = out_channels * self.feature_map_size * self.feature_map_size

        # Optional MLP for metadata
        if self.use_meta_fc:
            self.meta_fc = nn.Linear(metadata_dim, 32)
            combined_dim = self.flatten_dim + 32
        else:
            combined_dim = self.flatten_dim + metadata_dim

        # Combined classifier
        hidden_dim = max(combined_dim // 2, 64)
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images, metadata):
        x = self.cnn(images)
        x = x.view(-1, self.flatten_dim)

        # Metadata branch
        if self.use_meta_fc:
            m = F.relu(self.meta_fc(metadata))
        else:
            m = metadata

        # Combine
        combined = torch.cat([x, m], dim=1)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        out = torch.sigmoid(self.fc2(combined))

        return out.squeeze()




def train(model, train_loader, val_loader, num_epochs=50, patience=5, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # binary cross-entropy loss

    best_val_auc = 0  # Track best AUC too!
    best_val_loss = float('inf')

    epochs_no_improve = 0
    best_model_state=None

    print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'Val Acc':<10}{'Val AUC':<10}")
    print("-" * 50)

    ##this is the training loop!
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

        avg_train_loss = total_loss / len(train_loader)

        #validation loop!
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)

         # Unified progress print:
        print(f"{epoch+1:<8}{avg_train_loss:<12.4f}{val_loss:<12.4f}{val_acc:<10.4f}{val_auc:<10.4f}")

        ##the early stopping check!!

        # Save best model by AUC (better metric than loss)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience {patience} reached)")
                model.load_state_dict(best_model_state)
                print("Best model restored and saved.")
                break
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        #     best_model_state = model.state_dict()  # Save best weights
        #     torch.save(best_model_state, "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_best_model.pth")
            

        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f'Early stopping at epoch {epoch+1}')
        #         model.load_state_dict(best_model_state)  # Restore best weights
        #         torch.save(model.state_dict(), "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_best_model.pth")
        #         print("Model saved.")
        #         break


##note this kfold cross validation is only used for robustness check and not for actually getting the model

def cross_validate_kfold(model_class, train_dataset, n_splits=5, batch_size=64, device='cuda',use_meta_fc=True, dropout_rate=0.3, num_cnn_layers=4,use_channels=6):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_fold_auc = []
    all_fold_acc = []
    all_fold_f1 = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # Split into per-fold training and validation sets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Init fresh model for this fold
        #we would need to feed the parameters to this!
        
        model = model_class(metadata_dim=6, img_channels=use_channels, use_meta_fc=use_meta_fc, dropout_rate=dropout_rate, num_cnn_layers=num_cnn_layers).to(device)

        # Train the model on this fold
        train(model, train_loader, val_loader, num_epochs=50, patience=5, lr=1e-4, device=device)

        # Evaluation on this fold's validation split
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)

                preds = model(images, metadata)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # Metrics
        auc = roc_auc_score(val_labels, val_preds)

        # Using default 0.5 threshold
        preds_binary = (val_preds >= 0.5).astype(int)
        acc = accuracy_score(val_labels, preds_binary)
        f1 = f1_score(val_labels, preds_binary)

        print(f"Fold {fold + 1} - AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

        all_fold_auc.append(auc)
        all_fold_acc.append(acc)
        all_fold_f1.append(f1)

    # Summary after all folds
    print("\n=== Cross-validation summary ===")
    print(f"Mean AUC: {np.mean(all_fold_auc):.4f} ± {np.std(all_fold_auc):.4f}")
    print(f"Mean Acc: {np.mean(all_fold_acc):.4f} ± {np.std(all_fold_acc):.4f}")
    print(f"Mean F1:  {np.mean(all_fold_f1):.4f} ± {np.std(all_fold_f1):.4f}")
                
def evaluate(model, val_loader, device='cuda'):
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in val_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).float()

            preds = model(images, metadata)
            loss = criterion(preds, labels)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute accuracy (threshold at 0.5)
    pred_labels = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)
    #the accuracy score is defined here:
    # https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

    # Compute ROC-AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float('nan')  # in case only one class present in val set

    # print(f"Val Loss: {avg_loss:.4f} | Acc: {acc:.4f} | ROC-AUC: {auc:.4f}")

    return avg_loss, acc, auc  # we still return val loss to keep existing training loop working


def get_inputs(tgid, file_path, image_path, verbose=False):
    '''
    Given the input file, it reads the relevant files to get the relevant data

    file_path is the path to the galaxy folder to read the tractor model
    image_path is the path to the full fits file image

    we are going to be saving 96x96 images, and later cropping it ot 64x64 in the dataset creation state

    the output of this is the 6 channel 64x64 image given to CNN
    '''

    save_path = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_classifier_input_images/image_{tgid}.npy"

    # if os.path.exists( save_path ):
    #     cnn_input = np.load( save_path)
    # else:
    #load the relevant data
    tractor_model = np.load(file_path + "/tractor_source_model.npy")

    img_data = fits.open(image_path)
    data_arr = img_data[0].data

    if np.shape(data_arr) != np.shape(tractor_model):
        print(f"Something wonky happening with the sizes. {tgid}, {np.shape(data_arr)} {np.shape(tractor_model)} ")
        #we will crop the data_arr to the shape 350
        size = 350
        box_size = data_arr.shape[1]
        start = (box_size - size) // 2
        end = start + size
        data_arr = data_arr[:, start:end, start:end]
        print(f"After correcting, shape of data_arr : {np.shape(data_arr)}")

        size = 350
        box_size = tractor_model.shape[1]
        start = (box_size - size) // 2
        end = start + size
        tractor_model = tractor_model[:, start:end, start:end]
        print(f"After correcting, shape of tractor model : {np.shape(tractor_model)}")

        
        
    #we want to plot the original iamge 
    resis = data_arr - tractor_model 

    #now we trim all these objects to be in the central 64x64
    #actually, even though we want 64x64, we are doing some augmentations,and so we will choose a larger one
    size = 96
    box_size = data_arr.shape[1]
    start = (box_size - size) // 2
    end = start + size

    img = data_arr[:, start:end, start:end]
    img_minus_s = resis[:, start:end, start:end]

    #now we stack these two 
    cnn_input = np.vstack((img,img_minus_s))

    if verbose:
        print(f"Input image shape is {cnn_input.shape}")

    ##we need to also get the 6 metadata objects for this!
    #it is going to be fracflux_grz and mag_grz

    np.save(save_path, cnn_input)

    return cnn_input


def compute_channel_mean_std(all_images,use_channels = 6):
    """
    all_images: numpy array of shape (N, 6, 64, 64)
    Returns: mean and std for each of 6 channels (shape: (6,))
    """
    images_tensor = torch.tensor(all_images, dtype=torch.float32)
    # Reshape to (6, N * 64 * 64)
    flattened = images_tensor.permute(1, 0, 2, 3).reshape(use_channels, -1)
    # flattened = images_tensor.permute(1, 0, 2, 3).reshape(6, -1)
    
    mean = flattened.mean(dim=1)
    std = flattened.std(dim=1)
    return mean, std
    

def normalize_image(image, mean, std):
    """
    image: numpy array of shape (6, 64, 64)
    mean, std: torch tensors of shape (6,)
    """
    mean_np = mean.numpy()
    std_np = std.numpy()
    return (image - mean_np[:, None, None]) / std_np[:, None, None]


import matplotlib.pyplot as plt

def plot_image_grid_split_channels(images, probs, true_labs, indices, rows=5, cols=5, file_name = "summary_plot"):
    """
    images: numpy array of shape (N, 6, 64, 64)
    probs: numpy array of shape (N,)
    indices: list/array of indices to pick images from
    """

    fig = plt.figure(figsize=(cols * 3, rows * 2.5))
    outer = gridspec.GridSpec(rows, cols, wspace=0.15, hspace=0.0)

    for i, idx in enumerate(indices):
        img = images[idx]
        #the input probabilities are in the same order as the indices. The images are the full dataset and so we need those idx
        prob = probs[i]
        true_labs_i = true_labs[i] 

        size = 64
        start = (96 - size) // 2
        end = start + size
        img = img[:, start:end, start:end]

        # Split into first 3 and last 3 channels
        rgb1 = img[:3]
        #make the sdss rgb image of this!
        rgb1 = sdss_rgb([rgb1[0],rgb1[1],rgb1[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
     
        rgb2 = img[3:6]
        rgb2 = sdss_rgb([rgb2[0],rgb2[1],rgb2[2]], ["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)
        
        #make the sdss rgb image of this!

        # For each object, make 1 row x 2 columns panel
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, 
                                                 subplot_spec=outer[i], 
                                                 wspace=0.05)

        # Left panel (first 3 channels)
        ax1 = plt.Subplot(fig, inner[0])
        ax1.imshow(rgb1)
        ax1.axis('off')
        fig.add_subplot(ax1)

        # Right panel (last 3 channels)
        ax2 = plt.Subplot(fig, inner[1])
        ax2.imshow(rgb2)
        ax2.axis('off')
        fig.add_subplot(ax2)

        # Set common title on top
        ax1.set_title(f"P_fragment={prob:.2f}, Label = {true_labs_i}", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/{file_name}.pdf",bbox_inches="tight")
    plt.close()


def get_pcnn_data_inputs(sample_name, sample_cat_path = None):
    '''
    In this function, we create the .npy files that we will input into CNN 
    '''
    
    shred_cat = Table.read(sample_cat_path)

    print(f"Total number of shredded objects = {len(shred_cat)}")

    data_file_path = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/all_shred_data_N_6_96_96_{sample_name}.npy"
    metadata_file_path = f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/all_shred_metadata_scaled_{sample_name}.npy" 

    if os.path.exists(data_file_path) and os.path.exists(metadata_file_path):
        #now for this data_label, let us get all the input images!
        all_shred_images_unnorm = []
        
        for i in range(len(shred_cat)):
            shred_image_i = get_inputs(shred_cat["TARGETID"][i], shred_cat["FILE_PATH"][i],  shred_cat["IMAGE_PATH"][i],verbose=False)

            if i % 2000 == 0:
                print(f"{i}/{len(shred_cat)} Done")
            
            all_shred_images_unnorm.append(shred_image_i)
    
        all_shred_images_unnorm = np.array(all_shred_images_unnorm)
    
        np.save(data_file_path, all_shred_images_unnorm)
    
        #save all the metadata for the shredded catalog
    
        metadata_cols = ["MAG_G","MAG_R","MAG_Z","FRACFLUX_G","FRACFLUX_R","FRACFLUX_Z"]
        # Convert table to numpy array (shape: num_samples x num_features)
        all_metadata_array = np.vstack([shred_cat[col] for col in metadata_cols]).T
        print(f"Shape of entire metadata inputs = {np.shape(all_metadata_array)}")
        #Scale the metadata
        scaler = StandardScaler()
        all_metadata_scaled = scaler.fit_transform(all_metadata_array)
    
        ##save this data!!
        np.save(metadata_file_path,all_metadata_scaled )

    else:
        print("These files already exist!")

    return


if __name__ == '__main__':

    train_cnn = False
    use_channels = 6 #3 or 6
    run_data_collect = False

    # if run_data_collect:
    #     ##load the dataset
    #     data = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_VI_labelled.fits")
    #     #let us sort this so that target id is in increasing order and unique!!
    #     _,uni_inds = np.unique(data["TARGETID"].data, return_index=True)
    
    #     data = data[uni_inds]
    
    #     ##let us get the corresponding meta data for these objects!
    #     data_main = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v4.fits")
    #     data_main = data_main["TARGETID","RA","DEC", "MAG_G","MAG_R","MAG_Z","FRACFLUX_G","FRACFLUX_R","FRACFLUX_Z","SAMPLE","FILE_PATH","SIGMA_G","SIGMA_R","SIGMA_Z"]
    
    #     _,uni_inds_main = np.unique(data_main["TARGETID"].data, return_index=True)
    
    #     data_main = data_main[uni_inds_main]
    
    #     #check if the matching is indeed good
    #     print_maxabs_diff(data["TARGETID"].data, data_main["TARGETID"].data)
    #     print_maxabs_diff(data["RA"].data, data_main["RA"].data)
    
    #     #update our catalog with the metadata values!
    #     ##ADD SAMPLE INFO AND MAG TO SEE IF THIS IMRPOVES 
        
    #     for b in "GRZ":
    #         data[f"MAG_{b}"] = data_main[f"MAG_{b}"]
    #         data[f"FRACFLUX_{b}"] = data_main[f"FRACFLUX_{b}"]
    #     data["SAMPLE"] = data_main["SAMPLE"]
    #     data["FILE_PATH"] = data_main["FILE_PATH"]

    #     ## let us only look at objects that have at least two SIGMA > 5. This is only relevant for the ELGs!
        
    #     #get the labelled data set!
    #     data_label = data[(data["IS_SHRED_VI"] == "good") | (data["IS_SHRED_VI"] == "fragment")]

    #     metadata_cols = ["MAG_G","MAG_R","MAG_Z","FRACFLUX_G","FRACFLUX_R","FRACFLUX_Z"]
    #     # Convert table to numpy array (shape: num_samples x num_features)
    #     metadata_array = np.vstack([data_label[col] for col in metadata_cols]).T
    #     print(f"Shape of entire metadata inputs = {np.shape(metadata_array)}")
    #     #Scale the metadata
    #     scaler = StandardScaler()
    #     metadata_scaled = scaler.fit_transform(metadata_array)

    #     ##save this data!!
    #     np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/metadata_scaled.npy", metadata_scaled)

    #     data_label.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/data_labelled.fits",overwrite=True)

    # else:
    #     metadata_scaled = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/metadata_scaled.npy")

    #     data_label = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/data_labelled.fits")

    
    # print(f"Mean and std in one metadata axis = {np.mean(metadata_scaled[:,0]), np.std(metadata_scaled[:,0])}")

    # #let us get the labels now!!
    # #1 if it is shred and 0 if it is not a shred!
    # all_labels = np.zeros(len(data_label))
    # all_labels[ data_label["IS_SHRED_VI"] == "fragment"] = 1

    # print(f"Fragment number = {np.sum(all_labels)}, Good number = {len(all_labels[all_labels == 0])}")

    
    # if run_data_collect:
    #     #now for this data_label, let us get all the input images!
    #     all_input_images_unnorm = []
    #     for i in trange(len(data_label)):
    #         input_image_i = get_inputs(data_label["TARGETID"][i], data_label["FILE_PATH"][i],  data_label["IMAGE_PATH"][i],verbose=False)        
    #         all_input_images_unnorm.append(input_image_i)
    
    #     all_input_images_unnorm = np.array(all_input_images_unnorm)

    #     np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/input_data_N_6_64_64.npy", all_input_images_unnorm)
    
    # else:
    #     all_input_images_unnorm = np.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/input_data_N_6_64_64.npy")
        
    # #crop to relevant number of channels needed for the CNN input!
    # all_input_images_unnorm_f = all_input_images_unnorm[ :, :use_channels, :, : ]
    # print(f"Shape of entire input image data = {all_input_images_unnorm_f.shape}")
    
    # num_samples = len(data_label)
    # # indices = list(range(num_samples))
    # indices = np.arange(len(all_labels))

    # image_channel_mean, image_channel_std = compute_channel_mean_std(all_input_images_unnorm_f,use_channels = use_channels)
    # print("Image Channel means:", image_channel_mean)
    # print("Image Channel stds:", image_channel_std)

    # # ##testing the image normalization!!
    # all_input_images = normalize_image(all_input_images_unnorm_f,  image_channel_mean, image_channel_std)

    # for filti in range(use_channels):
    #     print(f"Entire Shape = {all_input_images.shape}, Single Filter Shape = {all_input_images[:, filti, :, :].shape}, Filter Mean = {all_input_images[:, filti, :, :].mean()}, Filter Std = {all_input_images[:, filti, :, :].std()}")

    # # 80% train, 20% test
    # train_size = int(0.8 * num_samples)
    # test_size = num_samples - train_size
    
    # train_indices, test_indices = random_split(indices, [train_size, test_size])

    # #get the transformations!!
    # train_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.RandomVerticalFlip(),    # Random vertical flip
    # transforms.RandomRotation(10),      # Random rotation between -10 and 10 degrees
    # transforms.CenterCrop(64),          # Crop back to 64x64 after transformations
    # ])

    # # valid_transforms = transforms.Compose([
    # # transforms.CenterCrop(64),          # Crop back to 64x64 after transformations
    # # ])


    # #Create Datasets
    # train_dataset = ShredDataset(all_input_images, metadata_scaled, all_labels, train_indices,transform=train_transforms)

    # #let us just feed the validation dataset the cropped images!
    # size = 64
    # start = (96 - size) // 2
    # end = start + size
    # all_input_images = all_input_images[:,:, start:end, start:end]
    # print(all_input_images.shape)
    # val_dataset = ShredDataset(all_input_images, metadata_scaled, all_labels, test_indices,transform=None)
    
    use_meta_fc = True
    dropout_rate = 0.3
    num_cnn_layers = 4
    
    # #this with 6 channels has pretty good results!
    # # model = SmallShredCNN(metadata_dim=6, img_channels=use_channels, use_meta_fc=True, dropout_rate=0.3, num_cnn_layers=4)
    
    # if train_cnn:
    #     #Create DataLoaders
    #     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    #     ##doing the k fold cross validation
    #     cross_validate_kfold(SmallShredCNN, train_dataset, n_splits=5, batch_size=64, device='cuda', use_channels=use_channels, use_meta_fc=use_meta_fc, dropout_rate=dropout_rate, num_cnn_layers=num_cnn_layers)

    #     #now actually training the model on the full train dataset
    #     model = SmallShredCNN(metadata_dim=6, img_channels=use_channels, use_meta_fc=use_meta_fc, dropout_rate=dropout_rate, num_cnn_layers=num_cnn_layers)
        
    #     train(model, train_loader, val_loader, num_epochs=100, patience=5, lr=5e-5, device='cuda')
    
    #     ##another thing is that I care most about completeness, what is the accuracy at which I am including majority of the shredded objects
    # else:
    
    
    model = SmallShredCNN(metadata_dim=6, img_channels=use_channels, use_meta_fc=use_meta_fc, dropout_rate=dropout_rate, num_cnn_layers=num_cnn_layers)
    
    # Load the saved model state
    model.load_state_dict(torch.load("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/shred_best_model.pth"))
    model = model.to('cuda')

    #validation loop, to print the final model auc scores
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # _, _, val_auc = evaluate(model, val_loader, "cuda")
    # print(f"Validation dataset AUC = {val_auc}")    

    # # Get pCNN scores for the validation dataset!
    # pred_pcnn, true_labels = get_pcnn_scores(model, val_dataset)

    # new_thresh = find_best_threshold(true_labels, pred_pcnn)

    # ## I need to pass it the validation inds!!
    # plot_image_grid_split_channels(all_input_images_unnorm, pred_pcnn, true_labels, test_indices[:50], rows=10, cols=5)

    # ##I should plot the examples that had a bad prediction!
    # # Use threshold (change if you have a better one)
    # threshold = 0.5  
    # pred_classes = (pred_pcnn >= threshold).astype(int)
    
    # test_wrong = (pred_classes != true_labels)

    # test_indices = np.array(test_indices)
    # test_wrong = np.array(test_wrong.tolist())

    # plot_image_grid_split_channels(all_input_images_unnorm, pred_pcnn[test_wrong], true_labels[test_wrong], test_indices[test_wrong][:25], rows=5, cols=5, file_name = "wrong_labels")


    # ##make plot of completeness of shred objects as a function of threshold. Can we find some threshold that will give us a high completeness sample of shredded objects?

    # thresh_grid = np.linspace(0., 0.999, 40)

    # fragment_comp_frac = []
    # good_impure_frac = []

    # for thi in thresh_grid:
    #     #compute what fraction of fragmented objects are above this cut!
    #     tot_fragment = len(pred_pcnn[ (true_labels == 1) ])

    #     tot_above_pcnn_fragment = len(pred_pcnn[ (pred_pcnn >= thi) & (true_labels == 1) ]) 

    #     tot_above_pcnn = len(pred_pcnn[ (pred_pcnn >= thi)])
    #     tot_above_pcnn_good = len(pred_pcnn[ (pred_pcnn >= thi) & (true_labels == 0) ]) 
        

    #     #what fraction of objects are removed with a cut like this for usefulness of this approach purposes
    #     #these will be the objects we call as not shreds!
    #     frac_remove =  len(pred_pcnn[ pred_pcnn < thi ])/len(pred_pcnn)

    #     fragment_comp_frac.append( tot_above_pcnn_fragment/tot_fragment )
    #     good_impure_frac.append( tot_above_pcnn_good / tot_above_pcnn )
        
    #     print(f"Threshold = {thi:.2f}, Fragment Completeness = {tot_above_pcnn_fragment/tot_fragment}, Nfrag_tot = {tot_fragment}, Nfrag_above = {tot_above_pcnn_fragment}, Frac Remove = {frac_remove:.2f}" )


    # #save these as numpy arrays
    # np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/fragment_completeness.npy",np.array(fragment_comp_frac) )
    # np.save("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/good_impurity.npy",np.array(good_impure_frac) )

    ##### RUN CNN ON THE SHRED CATALOG!

    #let us read the different shred catalog into a single file

    file_dict = {"BGS_BRIGHT": "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits",
                "BGS_FAINT": "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_FAINT_shreds_catalog_w_aper_mags.fits",
                "LOWZ": "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_LOWZ_shreds_catalog_w_aper_mags.fits",
                "ELG": "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_ELG_shreds_catalog_w_aper_mags.fits",
                "SGA": "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits"}

    if True:
        get_all_data = False

        for sample_i in file_dict.keys():
            print(f"Reading {sample_i} catalog")
            shred_cat = Table.read(file_dict[sample_i])

            print(f"Total number of shredded objects = {len(shred_cat)}")
        
            if get_all_data:
            
                #now for this data_label, let us get all the input images!
                all_shred_images_unnorm = []
                
                for i in trange(len(shred_cat)):
                    shred_image_i = get_inputs(shred_cat["TARGETID"][i], shred_cat["FILE_PATH"][i],  shred_cat["IMAGE_PATH"][i],verbose=False)      
                    
                    all_shred_images_unnorm.append(shred_image_i)
            
                all_shred_images_unnorm = np.array(all_shred_images_unnorm)
            
                np.save(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/all_shred_data_N_6_96_96_{sample_i}.npy", all_shred_images_unnorm)
            
                #save all the metadata for the shredded catalog
            
                metadata_cols = ["MAG_G","MAG_R","MAG_Z","FRACFLUX_G","FRACFLUX_R","FRACFLUX_Z"]
                # Convert table to numpy array (shape: num_samples x num_features)
                all_metadata_array = np.vstack([shred_cat[col] for col in metadata_cols]).T
                print(f"Shape of entire metadata inputs = {np.shape(all_metadata_array)}")
                #Scale the metadata
                scaler = StandardScaler()
                all_metadata_scaled = scaler.fit_transform(all_metadata_array)
            
                ##save this data!!
                np.save(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/all_shred_metadata_scaled_{sample_i}.npy", all_metadata_scaled)
        
            else:
                all_shred_images_unnorm = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/all_shred_data_N_6_96_96_{sample_i}.npy")
                all_metadata_scaled = np.load(f"/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/shred_classifier_output/all_shred_metadata_scaled_{sample_i}.npy")
    
            print("Images finished generated/reading!")
                
            ##we normalize the image
            #crop to relevant number of channels needed for the CNN input!
            all_shred_images_unnorm_f = all_shred_images_unnorm[ :, :use_channels, :, : ]
            print(f"Shape of entire input image data = {all_shred_images_unnorm_f.shape}")
            
            num_samples = len(shred_cat)
            all_indices = np.arange(len(shred_cat))
        
            print(num_samples, all_indices)
        
            all_image_channel_mean, all_image_channel_std = compute_channel_mean_std(all_shred_images_unnorm_f,use_channels = use_channels)
            print("Image Channel means:", all_image_channel_mean)
            print("Image Channel stds:", all_image_channel_std)
        
            # ##testing the image normalization!!
            all_shred_images = normalize_image(all_shred_images_unnorm_f,  all_image_channel_mean, all_image_channel_std)
        
            #let us just feed the validation dataset the cropped images!
            size = 64
            start = (96 - size) // 2
            end = start + size
            all_shred_images = all_shred_images[:,:, start:end, start:end]
            print(all_shred_images.shape)
            full_dataset = ShredDataset(all_shred_images, all_metadata_scaled, all_indices, all_indices,transform=None)
        
            full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
        
            full_pcnn = get_preds(model, full_loader, device='cuda')
    
            high_cnn_mask = (full_pcnn > 0.5)
    
            print(f"Fraction with PCNN > 0.5 = {np.sum(high_cnn_mask)/len(full_pcnn)}")
        
            #we will now save this column in the catalog
            shred_cat["PCNN_FRAGMENT"] = full_pcnn
        
            shred_cat.write(file_dict[sample_i],overwrite=True)
        
    

    

    



    

