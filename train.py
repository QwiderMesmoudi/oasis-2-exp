import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# Import our classes from the other file
from model_utils import MRIDataset, SliceModel

def train_one_fold(fold_num, train_df, val_df, config, device):
    # Setup Loaders
    train_ds = MRIDataset(train_df, img_size=config['data_load']['img_size'], num_slices=config['data_load']['num_slices'])
    val_ds = MRIDataset(val_df, img_size=config['data_load']['img_size'], num_slices=config['data_load']['num_slices'])
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2)

    # Model Setup
    model = SliceModel(config['train']['backbone']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train']['learning_rate']), weight_decay=float(config['train']['weight_decay']))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(config['train']['epochs']):
        # --- Training ---
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"F{fold_num} E{epoch}"):
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            preds = model(x).view(-1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss, all_probs, all_labels = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                out = model(x).view(-1)
                val_loss += criterion(out, y).item()
                all_probs.extend(torch.sigmoid(out).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        preds_bin = (all_probs > 0.5).astype(int)
        
        auc = roc_auc_score(all_labels, all_probs)
        acc = (preds_bin == all_labels).mean()

        wandb.log({f"fold_{fold_num}/val_auc": auc, f"fold_{fold_num}/val_loss": avg_val_loss})

        # Early Stopping & Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config['train']['model_save_path'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['train']['model_save_path']}best_fold_{fold_num}.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['train']['patience']: break
            
    return {"auc": auc, "acc": acc}

def main():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(config['data_load']['metadata_path'])
    
    wandb.init(project=config['wandb']['project'], name=config['wandb']['name'], config=config)

    sgkf = StratifiedGroupKFold(n_splits=config['data_load']['n_splits'], shuffle=True, random_state=config['data_load']['random_state'])
    
    scores = []
    for fold, (t_idx, v_idx) in enumerate(sgkf.split(df, df['label'], groups=df['Subject ID'])):
        print(f"\n--- FOLD {fold} ---")
        metrics = train_one_fold(fold, df.iloc[t_idx], df.iloc[v_idx], config, device)
        scores.append(metrics)

    print("CV Final AUC Mean:", np.mean([s['auc'] for s in scores]))

if __name__ == "__main__":
    main()