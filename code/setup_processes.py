# %%
# Dataset generation
import os
from mfcc_extractor_lib import setup_logger, process_wav_files

logger = setup_logger(script_name=os.path.splitext(os.path.basename(__file__))[0])
process_wav_files("timit_path", "train")

# %%
# Training process
from pytorch_model import run_training_process

data_dir = "Path"
output_folder = "Path"

run_training_process(data_dir, output_folder, 'model_path')
