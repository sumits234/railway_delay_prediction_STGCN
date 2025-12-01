############################
# imports
############################
# external libraries
import numpy as np
import os
import json
import torch
import sys
import pandas as pd
from scipy.sparse.linalg import eigs
import pdb
# custom libraries
from src.utils.utils import z_score


############################
# functions
############################

def data_interface(data_dir, dataset, n_nodes, ks, approx, device, n_timesteps_per_day, n_timesteps_in, n_timesteps_future, n_features_in):
    """Performs final data preprocessing for STGCN model"""
    Lk = process_adjacency(data_dir, dataset, ks, n_nodes, approx, device) #Calculates the Graph Laplacian (the math for "Spatial" connections")
    dataset_seq = sequence_data(data_dir, n_nodes, n_timesteps_per_day, n_timesteps_in, n_timesteps_future, n_features_in) # Loads the .npy file and chops it into sliding windows.
    data_train, data_test, data_val, output_stats = split_data(dataset_seq, n_timesteps_in)
    return Lk, data_train, data_test, data_val, output_stats


def sequence_data(data_dir, n_nodes, n_timesteps_per_day, n_timesteps_in, n_timesteps_out, n_features_in):
    """loads data from disk and processes into sequences"""
    dataset = np.load(data_dir + "dataset.npy")
    n_days = int(np.shape(dataset)[0] / n_timesteps_per_day)
    n_timesteps_seq = n_timesteps_in + n_timesteps_out
    n_slot = n_timesteps_per_day - n_timesteps_seq + 1
    n_sequences = n_slot * n_days
    dataset_seq = np.zeros((n_sequences, n_timesteps_seq, n_nodes, n_features_in))

    counter = 0
    for i in range(n_days):
        curr_data = dataset[i * n_timesteps_per_day : (i + 1) * n_timesteps_per_day]
        for j in range(n_slot):
            input_seq = curr_data[j : j + n_timesteps_in]
            output_seq = curr_data[j + n_timesteps_in : j + n_timesteps_in + n_timesteps_out]
            if (j + n_timesteps_in + n_timesteps_out) <= len(curr_data):
                tmp_data = np.expand_dims(np.concatenate((input_seq, output_seq)), 0)
                dataset_seq[counter] = tmp_data
                counter += 1
    
    dataset_seq = dataset_seq[:counter]
    return dataset_seq.astype("float64")


def split_data(dataset_seq, n_timesteps_in, percent_train=0.7, percent_test=0.1, percent_val=0.2):
    """Splits a dataset of sequences into train, test, val AND Normalizes correctly"""
    
    # randomize data
    dataset_seq = np.random.permutation(dataset_seq)
    train_start = 0
    train_end = train_start + int(len(dataset_seq) * percent_train)
    val_start = train_end 
    val_end = val_start + int(len(dataset_seq) * percent_val)
    test_start = val_end 
    test_end = len(dataset_seq)

    dataset_train = dataset_seq[train_start:train_end, :]
    dataset_test = dataset_seq[test_start:test_end, :]
    dataset_val = dataset_seq[val_start:val_end, :]

    train_input, train_label = generate_input_label(dataset_train, n_timesteps_in)
    test_input, test_label = generate_input_label(dataset_test, n_timesteps_in)
    val_input, val_label = generate_input_label(dataset_val, n_timesteps_in)

    # --- CORRECT NORMALIZATION LOGIC ---
    
    # 1. Calculate Stats for the OUTPUT (Label) - Always Feature 0 (Delay)
    # We need this to "un-normalize" the predictions later
    output_stats = {
        "mean": np.mean(train_label.numpy()), 
        "std": np.std(train_label.numpy())
    }
    print(f"Label (Delay) Stats: Mean={output_stats['mean']:.4f}, Std={output_stats['std']:.4f}")

    # 2. Normalize LABELS (Delay)
    train_label = z_score(train_label, output_stats["mean"], output_stats["std"])
    test_label = z_score(test_label, output_stats["mean"], output_stats["std"])
    val_label = z_score(val_label, output_stats["mean"], output_stats["std"])

    # 3. Normalize INPUTS (Channel-wise)
    # We iterate through each feature channel and normalize it independently
    # This ensures Delay, Speed, and Time are all scaled correctly
    n_features = train_input.shape[3]
    
    for i in range(n_features):
        # Calculate mean/std for this specific feature from TRAINING data
        feat_mean = torch.mean(train_input[:, :, :, i])
        feat_std = torch.std(train_input[:, :, :, i])
        
        # Skip normalization for Time/Day features (indices 1,2,3,4) if they are already -1 to 1
        # But normalizing them doesn't hurt, so we do it for consistency.
        # However, if std is 0 (constant feature), avoid division by zero
        if feat_std < 1e-5:
            feat_std = 1.0
            
        # Apply Z-score to Train, Test, Val for this channel
        train_input[:, :, :, i] = (train_input[:, :, :, i] - feat_mean) / feat_std
        test_input[:, :, :, i] = (test_input[:, :, :, i] - feat_mean) / feat_std
        val_input[:, :, :, i] = (val_input[:, :, :, i] - feat_mean) / feat_std

    data_train = train_input, train_label
    data_test = test_input, test_label
    data_val = val_input, val_label

    return data_train, data_test, data_val, output_stats


def generate_input_label(dataset, n_timesteps_in):
    """Generates input-label pairs"""
    data_input = torch.from_numpy(dataset[:, 0:n_timesteps_in, :, :]).double()
    # Label is only the 0th feature (Delay), so we select [:, 0]
    data_label = torch.from_numpy(dataset[:, -1, :, 0]).double().unsqueeze(1)
    
    # standardize to 4 dimensional tensor
    if len(data_label.shape) == 3:
        data_label = data_label.unsqueeze(3)
    return data_input, data_label


def process_adjacency(data_dir, dataset, ks, n_nodes, approx, device): #If Node A has high delay, multiplying by $L$ helps pass that value to Neighbor B.
    """helper function to cover multiple possibilities of Laplacian approximation"""
    A = np.load(data_dir + "adj.npy")
    L = scaled_laplacian(A)
    if approx == "cheb_poly":
        Lk = cheb_poly_approx(L, ks, n_nodes)
    elif approx == "first_order":
        Lk = first_approx(L, n_nodes)
    Lk = torch.from_numpy(Lk).double().to(device)
    return Lk


def scaled_laplacian(W):
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(L, k=1, which="LR")[0][0].real
    return np.asmatrix(2 * L / lambda_max - np.identity(n)).astype("float64")


def cheb_poly_approx(L, Ks, n): #Use Chebyshev polynomials (T_k(L)) to approximate the filter efficiently (O(K. E)
    L0, L1 = np.asmatrix(np.identity(n)), np.asmatrix(np.copy(L))
    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.asmatrix(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.asmatrix(np.copy(L1)), np.asmatrix(np.copy(Ln))
        return np.concatenate(L_list, axis=-1).astype("float64")
    elif Ks == 1:
        return np.asarray(L0).astype("float64")
    else:
        raise ValueError("ERROR: Spatial kernel size must be > 1")


def first_approx(W, n):
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.asmatrix(np.diag(d)).I)
    return np.asmatrix(np.identity(n) + sinvD * A * sinvD).astype("float64")