#project to path
path_project = "/home/akachat/tf_env/Ensemble_CNN_Melanoma"
# Training settings
train_dir = f"{path_project}/data/train"
test_dir = f"{path_project}/data/test"
save_model_path = f"{path_project}/Models/"
PLOTS_DIR = f"{path_project}/plots/"
num_classes = 1

# ============================================================================
# LEARNING RATE AND WEIGHT DECAY CONFIGURATION
# ============================================================================

# Single model, hybrid model , hybrid model ENSEMBLE training
learning_rate = 3e-4
weight_decay = 1e-5

# Model Parameters used for first Combination
learning_rate_one = 3e-4
weight_decay_one = 1e-5

# Model Parameters used for second Combination
learning_rate_two = 3e-4
weight_decay_two = 1e-5

# Model Parameters used for third Combination
learning_rate_three = 3e-4
weight_decay_three = 1e-5

patience = 10
min_delta = 0.001
batch_size = 32
img_size = (224, 224)
seed = 123
input_shape = (224, 224, 3)
epochs = 30
number_aug = 1
threshold = 0.5
metrics_per_model = {}
ml_models = {}
