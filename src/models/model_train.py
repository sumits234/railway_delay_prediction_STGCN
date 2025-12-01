############################
# imports
############################
# external libraries
import pdb
import numpy as np
import time
import torch
import os
import copy
import json
import pandas as pd # Used for saving history to CSV

# custom libraries
from src.utils.utils import evaluation

############################
# functions
############################

def model_train(data_train, data_val, data_test, output_stats, graph_kernel, model, optimizer, scheduler, loss_criterion, writer, args, ckpt_dir):
    """
    Main training loop for the STGCN model.
    """
    # --- 1. Prepare Data ---
    train_input, train_label = data_train
    train_input = train_input.permute(0, 3, 1, 2)
    train_label = train_label.permute(0, 3, 1, 2)

    val_input, val_label = data_val
    val_input = val_input.permute(0, 3, 1, 2)
    val_label = val_label.permute(0, 3, 1, 2)

    train_idx = torch.randperm(train_input.shape[0])
    val_idx = np.arange(0, val_input.shape[0], 1)

    # --- 2. Setup Tracking ---
    best_metrics = np.ones((2, 1)) * 1e6 # [RMSE, MAE]
    fp_optimized_params = os.path.join(ckpt_dir, "optimized_model.model")
    
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [], 
        "val_rmse": [],
        "val_mae": [],
        "test_rmse": [],
        "test_mae": []
    }

    # --- 3. Training Loop ---
    for epoch in range(args.n_epochs):
        print("\n\nEpoch: {}/{}".format(epoch, args.n_epochs))
        
        # --- A. TRAINING PHASE ---
        print("Training")
        avg_training_loss = 0
        batch_count = 0
        model.train()
        start_time = time.time()
        
        for i in range(0, train_input.shape[0], args.batch_size):
            optimizer.zero_grad()
            idx = train_idx[i:i+args.batch_size]
            X = train_input[idx].to(args.device)
            y = train_label[idx].to(args.device)
            
            y_hat = model(X, graph_kernel)
            loss = loss_criterion(y_hat, y)
            avg_training_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            batch_count += 1

            if i % 5000 == 0 and i > 0:
                print("Step: {}/{}".format(i, train_input.shape[0]))
        
        if batch_count == 0:
             print("Warning: No training batches were processed.")
             continue

        print("Training Time: {} sec".format(round(time.time() - start_time, 2)))
        avg_training_loss /= batch_count
        writer.add_scalar("Loss/Training", avg_training_loss, epoch)

        # --- B. VALIDATION PHASE ---
        print("\nValidation")
        avg_val_loss = 0 # <--- ADDED
        rmse = 0
        mae = 0
        batch_count = 0
        
        with torch.no_grad():
            model.eval()
            start_time = time.time()
            
            for i in range(0, val_input.shape[0], args.batch_size):
                idx = val_idx[i:i+args.batch_size]
                X = val_input[idx].to(args.device)
                y_tensor = val_label[idx].to(args.device) # Move label to GPU for Loss calc
                
                y_hat = model(X, graph_kernel)
                
                # Calculate Validation Loss (MSE) on normalized data (same as training loss)
                val_loss = loss_criterion(y_hat, y_tensor)
                avg_val_loss += val_loss.item()

                # Move to CPU for RMSE/MAE calculation (un-normalized)
                y_cpu = val_label[idx] 
                y_hat_cpu = y_hat.cpu()

                rmse_batch, mae_batch = evaluation(y_cpu, y_hat_cpu, output_stats)
                rmse += rmse_batch
                mae += mae_batch
                batch_count += 1
            
            if batch_count == 0:
                print("Warning: No validation batches were processed.")
                continue

            avg_val_loss /= batch_count # <--- ADDED
            rmse /= batch_count
            mae /= batch_count

            print("Validation Loss (MSE): ", avg_val_loss)
            print("MAE (validation): ", mae)
            print("RMSE (validation): ", rmse)
            print("Inference Time: {} sec".format(round(time.time() - start_time, 2)))
            
            # --- C. SAVE BEST MODEL ---
            if rmse < best_metrics[0] and mae < best_metrics[1]:
                print(f"New best model found! MAE: {mae:.4f} (Previous: {best_metrics[1][0]:.4f})")
                best_metrics[0] = rmse
                best_metrics[1] = mae
                
                torch.save(model.state_dict(), fp_optimized_params)

                stats_fn = "output_stats.json"
                fp_output_stats = os.path.join(ckpt_dir, stats_fn)
                with open(fp_output_stats, 'w') as f:
                    stats_serializable = {k: float(v) for k, v in output_stats.items()}
                    json.dump(stats_serializable, f)
                
                print(f"Saved best model and stats.")

            writer.add_scalar("Loss/Validation", avg_val_loss, epoch) # <--- ADDED
            writer.add_scalar("Root Mean Squared Error (validation)", rmse, epoch)
            writer.add_scalar("Mean Absolute Error (validation)", mae, epoch)
            writer.flush()

            # --- D. TESTING PHASE ---
            print("\nTesting")
            test_rmse = 0
            test_mae = 0
            
            test_model = copy.copy(model)
            if os.path.exists(fp_optimized_params):
                test_model.load_state_dict(torch.load(fp_optimized_params))
                test_model.to(args.device)
                test_rmse, test_mae = model_test(data_test, output_stats, graph_kernel, test_model, writer, args, epoch)
            else:
                print("Skipping test: No best model saved yet.")

            scheduler.step()

            # --- E. SAVE HISTORY TO CSV ---
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_training_loss)
            history["val_loss"].append(avg_val_loss) # <--- ADDED
            history["val_rmse"].append(rmse)
            history["val_mae"].append(mae)
            history["test_rmse"].append(test_rmse)
            history["test_mae"].append(test_mae)

            df_history = pd.DataFrame(history)
            df_history.to_csv(os.path.join(ckpt_dir, "training_history.csv"), index=False)

    return fp_optimized_params


def model_test(data_test, output_stats, graph_kernel, model, writer, args, epoch):
    """
    Runs evaluation on the test set.
    """
    test_input, test_label = data_test
    test_input = test_input.permute(0, 3, 1, 2)
    test_label = test_label.permute(0, 3, 1, 2)
    
    test_idx = np.arange(0, test_input.shape[0], 1)

    batch_count = 0
    rmse = 0
    mae = 0

    with torch.no_grad():
        model.eval()
        start_time = time.time()
        for i in range(0, test_input.shape[0], args.batch_size):
            idx = test_idx[i:i+args.batch_size]
            X = test_input[idx].to(args.device)
            y = test_label[idx] # CPU
            y_hat = model(X, graph_kernel)
            y_hat = y_hat.cpu() # CPU

            rmse_batch, mae_batch = evaluation(y, y_hat, output_stats)
            rmse += rmse_batch
            mae += mae_batch
            batch_count += 1
        
        if batch_count == 0:
            return 0, 0

        rmse /= batch_count
        mae /= batch_count

        print("MAE (testing): ", mae)
        print("RMSE (testing): ", rmse)
        print("Inference Time: {} sec".format(round(time.time() - start_time, 2)))

        writer.add_scalar("Root Mean Squared Error (testing)", rmse, epoch)
        writer.add_scalar("Mean Absolute Error (testing)", mae, epoch)
        writer.flush()
        
        return rmse, mae