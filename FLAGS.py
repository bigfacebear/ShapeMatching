data_dir = "/home/bfb/Affinity/ShapeMatching/Dataset/Rotation"

batch_size = 128

use_fp16 = False



NOTIFICATION_EMAIL = "qc_zhao@outlook.com"


train_dir = 'MSHAPEStrain/'
max_steps = 1000000
log_device_placement = False  # Whether to log device placement.
log_frequency = 10  # How often to log results to the console.



CHECK_DATASET = True

RESTORE = False
RESTORE_FROM = 'summaries/netstate/saved_state-3000'
