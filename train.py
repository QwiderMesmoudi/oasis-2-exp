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
    recall_score, roc_auc_score, f1_score, precision_score,accuracy_score
    , confusion_matrix
)
from tqdm import tqdm
from dvclive import Live
from model_utils import MRIDataset, SliceModel

def train_one_fold(fold_num, train_df, val_df, config, device, live):
    # --- 1. Setup Data ---
    train_ds = MRIDataset(train_df, img_size=config['data_load']['img_size'], num_slices=config['data_load']['num_slices'])
    val_ds = MRIDataset(val_df, img_size=config['data_load']['img_size'], num_slices=config['data_load']['num_slices'])
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2)

    # --- 2. Setup Model ---
    model = SliceModel(config['train']['backbone']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train']['learning_rate']), weight_decay=float(config['train']['weight_decay']))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    early_stop_counter = 0

    # --- 3. Epoch Loop ---
    for epoch in range(config['train']['epochs']):
        # Training Phase
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Fold {fold_num} | Ep {epoch}"):
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            preds = model(x).view(-1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss, all_probs, all_labels = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                outputs = model(x).view(-1)
                val_loss += criterion(outputs, y).item()
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # --- 📊 4. Metric Calculations ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        all_labels, all_probs = np.array(all_labels), np.array(all_probs)
        preds_bin = (all_probs > 0.5).astype(int)
        
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, preds_bin)
        val_f1 = f1_score(all_labels, preds_bin, zero_division=0)
        val_precision = precision_score(all_labels, preds_bin, zero_division=0)
        recall = recall_score(all_labels, preds_bin, zero_division=0) # Sensitivity
        
        # Specificity Calculation
        tn, fp, fn, tp = confusion_matrix(all_labels, preds_bin, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # --- 📈 5. Log to WandB (Live Charts) ---
        wandb.log({
            f"fold_{fold_num}/epoch": epoch,
            f"fold_{fold_num}/train_loss": avg_train_loss,
            f"fold_{fold_num}/val_loss": avg_val_loss,
            f"fold_{fold_num}/val_auc": auc,
            f"fold_{fold_num}/val_acc": acc,
            f"fold_{fold_num}/val_recall": recall,
            f"fold_{fold_num}/val_precision": val_precision,
            f"fold_{fold_num}/val_f1": val_f1,
            f"fold_{fold_num}/val_specificity": specificity,
            f"fold_{fold_num}/lr": optimizer.param_groups[0]['lr']
        })

        # --- 📉 6. Log to DVCLive (VS Code Plots) ---
        live.log_metric(f"fold_{fold_num}/auc", auc)
        live.log_metric(f"fold_{fold_num}/acc", acc)
        live.log_metric(f"fold_{fold_num}/recall", recall)
        live.log_metric(f"fold_{fold_num}/precision", val_precision)
        live.log_metric(f"fold_{fold_num}/f1", val_f1)
        live.log_metric(f"fold_{fold_num}/specificity", specificity)
        live.log_metric(f"fold_{fold_num}/val_loss", avg_val_loss)
        live.next_step()

        print(f"Fold {fold_num} Ep {epoch} | AUC: {auc:.4f} | Recall: {recall:.4f} | Spec: {specificity:.4f}")

        # --- 7. Save Best Model & Early Stop ---
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config['train']['model_save_path'], exist_ok=True)
            save_path = os.path.join(config['train']['model_save_path'], f"best_fold_{fold_num}.pth")
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['train']['patience']: break
            
    return {"auc": auc, "acc": acc, "recall": recall, "specificity": specificity ,
            "f1": val_f1, "precision": val_precision }

def main():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(config['data_load']['metadata_path'])
    
    # Initialize WandB
    wandb.init(project=config['wandb']['project'], name=config['wandb']['name'], config=config)

    # Initialize DVCLive (Fixed: report="html")
    with Live(dir="dvclive", report="html") as live:
        sgkf = StratifiedGroupKFold(n_splits=config['data_load']['n_splits'], shuffle=True, random_state=config['data_load']['random_state'])
        
        scores = []
        for fold, (t_idx, v_idx) in enumerate(sgkf.split(df, df['label'], groups=df['Subject ID'])):
            metrics = train_one_fold(fold, df.iloc[t_idx], df.iloc[v_idx], config, device, live)
            scores.append(metrics)

        # Log Final Mean Metrics
        # Aggregate
        final_metrics = {
            k: np.mean([s[k] for s in scores]) for k in scores[0]
            }

        print("Final CV Metrics:", final_metrics)
        
        wandb.log({f"Dataset":"Oasis2",
                   "Task":"Hc vs AD",
                   "Model":config['train']['backbone'],
                   "Acc":final_metrics['acc'],
                   "precision":final_metrics['precision'],
                   "Sensitivity":final_metrics['recall'],
                   "Specificity":final_metrics['specificity'],
                   "AUC":final_metrics['auc'],
                   "F1":final_metrics['f1'],
                   })
        
        wandb.finish()

if __name__ == "__main__":
    main()