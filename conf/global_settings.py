from datetime import datetime
CHECKPOINT_PATH = 'checkpoint'
# total training epoches
EPOCH = 100

# time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

# tensorboard log dir
LOG_DIR = 'runs'
DATA_PATH = './data'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
