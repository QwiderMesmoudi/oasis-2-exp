import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, confusion_matrix
)
from tqdm import tqdm
from dvclive import Live

# Import our custom classes from model_utils.py
from model_utils import MRIDataset, SliceModel

def train_one_fold(fold_num, train_df, val_df, config, device, live):
    """
    Trains the model for a single fold and logs metrics to DVCLive.
    """
    # 1. Dataset & Loaders
    train_ds = MRIDataset(
        train_df, 
        img_size=config['data_load']['img_size'], 
        num_slices=config['data_load']['num_slices']
    )
    val_ds = MRIDataset(
        val_df, 
        img_size=config['data_load']['img_size'], 
        num_slices=config['data_load']['num_slices']
    )
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2)

    # 2. Model, Optimizer, Loss
    model = SliceModel(config['train']['backbone']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['train']['learning_rate']), 
        weight_decay=float(config['train']['weight_decay'])
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    early_stop_counter = 0

    # 3. Training Loop
    for epoch in range(config['train']['epochs']):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Fold {fold_num} | Epoch {epoch}"):
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            preds = model(x).view(-1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 4. Validation Loop
        model.eval()
        val_loss, all_probs, all_labels = 0, [], []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device).float()
                outputs = model(x).view(-1)
                
                val_loss += criterion(outputs, y).item()
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                # --- 🖼️ OPTIONAL: Log MRI Slices to WandB (First Batch Only) ---
                if i == 0 and epoch % 5 == 0: 
                    # Log middle slice of the first brain in the batch
                    sample_slice = x[0, config['data_load']['num_slices']//2, 0, :, :].cpu().numpy()
                    live.log_image(f"mri_fold_{fold_num}.png", sample_slice)

        # 5. Metric Calculations
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        all_labels, all_probs = np.array(all_labels), np.array(all_probs)
        preds_bin = (all_probs > 0.5).astype(int)
        
        auc = roc_auc_score(all_labels, all_probs)
        acc = (preds_bin == all_labels).mean()

        # 6. --- 📊 LOGGING WITH DVCLIVE ---
        # We use a prefix to distinguish folds in the plots
        live.log_metric(f"fold_{fold_num}/train_loss", avg_train_loss)
        live.log_metric(f"fold_{fold_num}/val_loss", avg_val_loss)
        live.log_metric(f"fold_{fold_num}/auc", auc)
        live.log_metric(f"fold_{fold_num}/accuracy", acc)
        
        # Advance the step for plots
        live.next_step()

        print(f"Fold {fold_num} Ep {epoch} | Loss: {avg_val_loss:.4f} | AUC: {auc:.4f}")

        # 7. Scheduler & Early Stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config['train']['model_save_path'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['train']['model_save_path']}best_fold_{fold_num}.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['train']['patience']:
                print("Early stopping.")
                break
            
    return {"auc": auc, "acc": acc}

def main():
    # Load configuration
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(config['data_load']['metadata_path'])
    
    # Initialize DVCLive with WandB integration
    # report="wandb" automatically syncs DVCLive metrics to your WandB dashboard
    with Live(dir="dvclive", report="wandb") as live:
        
        # Cross-Validation Setup
        sgkf = StratifiedGroupKFold(
            n_splits=config['data_load']['n_splits'], 
            shuffle=True, 
            random_state=config['data_load']['random_state']
        )
        
        scores = []
        for fold, (t_idx, v_idx) in enumerate(sgkf.split(df, df['label'], groups=df['Subject ID'])):
            print(f"\n===== STARTING FOLD {fold} =====")
            metrics = train_one_fold(fold, df.iloc[t_idx], df.iloc[v_idx], config, device, live)
            scores.append(metrics)

        # Log Final Summary Metrics
        mean_auc = np.mean([s['auc'] for s in scores])
        mean_acc = np.mean([s['acc'] for s in scores])
        
        live.log_metric("final/mean_auc", mean_auc, plot=False)
        live.log_metric("final/mean_acc", mean_acc, plot=False)
        
        print(f"\nFinal CV AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    main()