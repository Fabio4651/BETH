import os
import pickle

from datetime import datetime # DEBUG

from dataset import DATASETS
#from dataset import BETHDataset, GaussianDataset, DATASETS
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from benchmarks import WhitenedBenchmark

from plotting import plot_data, plot_line
from training import get_marginal_posterior, test_sklearn, test_vae, train_sklearn, train_vae, validate_sklearn, validate_vae

import tqdm
from torch import optim
from vae import VAE
from dose import get_summary_stats

def train():
    visualize = True
    dataset = 'beth'
    patience = 3
    seed = 1
    benchmark = 'ifor' #'dose' #'rcov'
    outliers_fraction = 0.05

    model_name = benchmark
    use_vae = True if model_name == "dose" else False

    ##########################
    # Data
    ##########################
    train_dataset, val_dataset, test_dataset = [DATASETS[dataset](split=split, subsample=0) for split in ["train", "val", "test"]]
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, pin_memory=True)
    if hasattr(train_dataset, "plot"):  # Plot the first two dimensions of each tensor dataset
        plot_data([train_dataset, val_dataset, test_dataset], ["train", "val", "test"], train_dataset, prefix=f"{dataset}_gaussian")



def main():
    start = datetime.now() # DEBUG
    print("Start: ", start) # DEBUG
    #args = configure()

    train()
    #test()

    end = datetime.now() # DEBUG
    print(f"Time to Complete: {end - start}") # DEBUG


if __name__ == "__main__":
    main()
