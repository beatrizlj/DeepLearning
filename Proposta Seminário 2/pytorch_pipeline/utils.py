import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history, fold, save_dir):
    """
    Plota as curvas de Loss e Acurácia de Treino vs Validação.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Curva de Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Treino Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validação Loss')
    plt.title(f'Loss - Fold {fold}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Curva de Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Treino Acurácia')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validação Acurácia')
    plt.title(f'Acurácia - Fold {fold}')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'history_fold_{fold}.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, fold, save_dir):
    """
    Calcula e plota a Matriz de Confusão elegante usando Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusão - Fold {fold}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_fold_{fold}.png'))
    plt.close()

def save_json_info(info_dict, filepath):
    """
    Salva dicionário de informações em um arquivo JSON.
    """
    with open(filepath, 'w') as f:
        json.dump(info_dict, f, indent=4)
