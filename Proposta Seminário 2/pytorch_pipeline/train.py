import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm

from config import *
from dataset import load_medmnist_data, BloodCellDataset
from models import get_model
from losses import get_loss
from utils import plot_training_history, plot_confusion_matrix, save_json_info

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc='Train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (corrects.double() / total).item() * 100.0
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Val', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = (corrects.double() / total).item() * 100.0
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    # Fixar seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # 1. Carregar caminhos e labels
    print("Buscando dados em:", DATASET_PATH)
    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: Arquivo {DATASET_PATH} não encontrado! Coloque o NPZ lá.")
        return
        
    images, labels, idx_to_class = load_medmnist_data(DATASET_PATH)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Encontradas {len(images)} imagens em {len(class_names)} classes.")
    if len(images) == 0:
        return
        
    # Preparar K-Fold
    skf = StratifiedKFold(n_splits=K_CV, shuffle=True, random_state=SEED)
    
    # Outer Loop: Percorrer cada rede definida em config.py
    for network_name in NETWORKS_TO_TRAIN:
        print(f"\n\n{'*'*60}\nINICIANDO TREINAMENTO: {network_name.upper()}\n{'*'*60}")
        
        # Pasta específica para a rede
        current_model_dir = os.path.join(MODELS_DIR, network_name)
        os.makedirs(current_model_dir, exist_ok=True)
        
        # Dicionário para guardar estatísticas finais DESTA rede
        final_results = {
            'network': network_name,
            'loss_function': LOSS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'folds_acc': []
        }
        
        # Loop de Validação Cruzada
        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels), 1):
            print(f"\n{'='*40}\n[{network_name.upper()}] Iniciando Fold {fold}/{K_CV}\n{'='*40}")
            
            # Subsets de imagens para este fold
            train_images = images[train_idx]
            train_labels = labels[train_idx]
            val_images = images[val_idx]
            val_labels = labels[val_idx]
            
            # Datasets e DataLoaders
            train_dataset = BloodCellDataset(train_images, train_labels, is_train=True)
            val_dataset = BloodCellDataset(val_images, val_labels, is_train=False)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Instanciar Modelo, Loss e Otimizador
            model = get_model(network_name, num_classes=len(class_names)).to(device)
            criterion = get_loss(LOSS)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            
            best_val_acc = 0.0
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            best_preds = None
            best_labels_val = None
            
            # Pasta do fold dentro da pasta da rede
            fold_dir = os.path.join(current_model_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Loop de Épocas
            for epoch in range(1, EPOCHS + 1):
                print(f"Época {epoch}/{EPOCHS} [{network_name.upper()} - Fold {fold}]")
                
                t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                v_loss, v_acc, preds, trues = evaluate(model, val_loader, criterion, device)
                
                scheduler.step(v_acc)
                
                print(f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")
                
                # Registrar histórico
                history['train_loss'].append(t_loss)
                history['val_loss'].append(v_loss)
                history['train_acc'].append(t_acc)
                history['val_acc'].append(v_acc)
                
                # Salvar melhor modelo do fold
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
                    best_preds = preds
                    best_labels_val = trues
                    print(" -> Melhor modelo salvo!")
                    
            # Fim do fold: gerar gráficos
            print(f"Fim do Fold {fold}. Melhor Acurácia de Validação: {best_val_acc:.2f}%")
            final_results['folds_acc'].append(best_val_acc)
            
            plot_training_history(history, fold, fold_dir)
            plot_confusion_matrix(best_labels_val, best_preds, class_names, fold, fold_dir)
            
        # Fim da Validação Cruzada DESTA rede: Consolidar resultados
        mean_acc = np.mean(final_results['folds_acc'])
        std_acc = np.std(final_results['folds_acc'])
        final_results['mean_accuracy'] = mean_acc
        final_results['std_accuracy'] = std_acc
        
        print(f"\n{'='*40}\nTreinamento Concluído para {network_name.upper()}!\n{'='*40}")
        print(f"Acurácia Média ({K_CV}-Folds): {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        save_json_info(final_results, os.path.join(current_model_dir, f'final_results_{network_name}.json'))
        print(f"Resultados, gráficos e pesos da {network_name.upper()} salvos em: {current_model_dir}\n")

if __name__ == '__main__':
    main()
