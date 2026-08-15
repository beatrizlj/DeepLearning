import os

# ---------------------------------------------------------
# Configurações Globais (Hiperparâmetros e Paths)
# ---------------------------------------------------------

# Semente global para reprodutibilidade
SEED = 42

# Parâmetros de Treinamento
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# Validação Cruzada
K_CV = 5

# Resolução padrão das imagens
IMAGE_SIZE = (224, 224)

# Escolha da rede: 'resnet', 'vgg', 'densenet', 'efficientnet', 'mobilenet'
NETWORK = 'resnet'

# Escolha da loss: 'bce', 'ce', 'focal', etc.
LOSS = 'ce'

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(WORKSPACE_DIR, 'Dataset', 'bloodmnist.npz')  # Caminho direto para o NPZ
MODELS_DIR = os.path.join(BASE_DIR, 'pytorch_pipeline', 'Models')

# Criar pasta de modelos caso não exista
os.makedirs(MODELS_DIR, exist_ok=True)
