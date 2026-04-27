# =====================================================
# PATHS
# =====================================================
DATA_ROOT = "/home/firiusz/Downloads/deepfakedata/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# =====================================================
# DATA SETTINGS
# =====================================================
IMG_SIZE = 128          # try: 64 / 128 / 256
NUM_WORKERS = 4
PIN_MEMORY = True

# =====================================================
# TRAINING SETTINGS
# =====================================================
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = "cuda"

# =====================================================
# K-FOLD SETTINGS
# =====================================================
N_FOLDS = 2
RANDOM_STATE = 42

# Early stopping
PATIENCE = 3
MIN_DELTA = 0.0

# =====================================================
# MODEL SETTINGS
# =====================================================
NUM_CLASSES = 2

# =====================================================
# AUGMENTATION SETTINGS
# =====================================================
USE_AUGMENTATION = False

# =====================================================
# OPTIMIZER SETTINGS
# =====================================================
OPTIMIZER = "adamw"   # options: adam, adamw
BETAS = (0.9, 0.999)

# =====================================================
# LOGGING / SAVING
# =====================================================
SAVE_DIR = "./checkpoints"
SAVE_BEST_ONLY = True

PRINT_EVERY = 1

# =====================================================
# WAVLET SETTINGS
# =====================================================
WAVELET_TYPE = "db4"
WAVELET_LEVELS = 2
