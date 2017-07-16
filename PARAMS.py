# Global constants describing the MSHAPES data set.
IMAGE_SIZE = 100
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# How many examples to use for training in the queue
NUM_EXAMPLES_TO_LOAD_INTO_QUEUE = 40000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Where to download the MSHAPES dataset from
DATA_URL = 'https://electronneutrino.com/affinity/shapes/datasets/MSHAPES_180DEG_ONECOLOR_SIMPLE_50k_100x100.zip'
