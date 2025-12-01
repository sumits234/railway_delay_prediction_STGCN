############################
# imports
############################
print("\n\n\n\nimporting libraries")

# external libraries
import os
import sys
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

print("Current directory:", os.getcwd())

# custom libraries
from src.models.model import STGCN
from src.models.model_train import model_train, model_test
from src.data.data_processing import data_interface

parser = argparse.ArgumentParser()
############################
# initial setup
############################
print("\n\n\n\ninitial setup")
dataset = "raildelays"

data_dir = "./data/processed/"
log_dir = "./models/output/"
log_dir += datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
ckpt_dir = "./models/checkpoints/"
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

# Use PyTorch's SummaryWriter
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")

if dataset == "raildelays":
    n_nodes = 40
    n_timesteps_per_day = 42
    n_timesteps_in = 12
    n_timesteps_future = 1
    
    # --- UPDATED FEATURE COUNT ---
    # 1 = Delay only
    # 6 = Delay + Time_Sin + Time_Cos + Day_Sin + Day_Cos + Travel_Time
    n_features_in = 6  
    
    n_features_out = 1
else:
    print("Specified dataset currently not supported")
    exit()

# data parameters
parser.add_argument("--n_nodes", type=int, default=n_nodes)
parser.add_argument("--n_timesteps_per_day", type=int, default=n_timesteps_per_day)
parser.add_argument("--n_timesteps_in", type=int, default=n_timesteps_in)
parser.add_argument("--n_timesteps_future", type=int, default=n_timesteps_future)
parser.add_argument("--approx", type=str, default="cheb_poly", choices={"cheb_poly", "first_order"})

# kernel sizes for spatial (graph) and temporal convolutions
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--kt", type=int, default=3)

# training parameters
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=30) 
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="ADAM", choices={"ADAM"})
parser.add_argument("--drop_prob", type=float, default=0.0)

# GPU setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using Device: CPU")
parser.add_argument("--device", type=str, default=device)

# Use parse_known_args() to avoid errors in notebooks
args, unknown = parser.parse_known_args()

############################
# data interface
############################
print("\n\n\n\nload from disk: dataset={}".format(dataset))

Lk, data_train, data_test, data_val, output_stats = data_interface(data_dir,
                                                    dataset,
                                                    args.n_nodes,
                                                    args.ks,
                                                    args.approx,
                                                    args.device,
                                                    args.n_timesteps_per_day,
                                                    args.n_timesteps_in,
                                                    args.n_timesteps_future,
                                                    n_features_in)

print(f"Training data shape (Input, Label): {data_train[0].shape}, {data_train[1].shape}")
print(f"Validation data shape (Input, Label): {data_val[0].shape}, {data_val[1].shape}")
print(f"Test data shape (Input, Label): {data_test[0].shape}, {data_test[1].shape}")

############################
# training setup
############################
print("\n\n\n\ntraining setup")

blocks = [[n_features_in, 32, 64], [64, 32, 128], [128, n_features_out]]
model = STGCN(blocks,
                args.n_timesteps_in,
                args.n_timesteps_future,
                args.n_nodes,
                args.device,
                args.ks,
                args.kt,
                args.drop_prob).to(args.device).double() # Ensure model is float64

loss_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

############################
# training loop
############################
if __name__ == "__main__":
    
    print("\n\n\n\ntraining loop")
    fp_optimized_params = model_train(data_train, 
                                        data_val, 
                                        data_test, 
                                        output_stats, 
                                        Lk, 
                                        model, 
                                        optimizer, 
                                        scheduler, 
                                        loss_criterion, 
                                        writer, 
                                        args, 
                                        ckpt_dir)
    
    writer.close()
    
    print("\n\n--- Training Complete ---")
    if os.path.exists(fp_optimized_params):
            print(f"Optimized model saved to: {fp_optimized_params}")
            print(f"Output stats saved to: {os.path.join(ckpt_dir, 'output_stats.json')}")
    else:
            print("Training finished, but no new best model was saved.")