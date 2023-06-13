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

    ##########################
    # Model
    ##########################
    #model_name = args.benchmark
    #use_vae = True if model_name == "dose" else False
    #if model_name == "dose":
        #use_vae = True
        #input_shape = train_dataset.get_input_shape()
        #model = VAE(input_shape=input_shape, latent_size=2, hidden_size=64, observation=train_dataset.get_distribution())
        #device = torch.device('cuda') #'cpu'
        #device = torch.device('cpu')
        #model.to(device=device)
        #optimiser = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
        #prior = MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device))
    ''' else: 
        use_vae = False
        if model_name == "rcov":
            model = EllipticEnvelope(contamination=args.outliers_fraction) # Robust Covariance
        elif model_name == "svm":
            base_model = SGDOneClassSVM(random_state=args.seed)
            model = WhitenedBenchmark(model_name, base_model, args)
        elif model_name == "ifor":
            base_model = IsolationForest(contamination=args.outliers_fraction, random_state=args.seed)
            model = WhitenedBenchmark(model_name, base_model, args)'''
    
    use_vae = False
    device = torch.device('cpu')
    base_model = IsolationForest(contamination=outliers_fraction, random_state=seed)
    model = WhitenedBenchmark(model_name, base_model)
    #model.to(device=device)
    optimiser = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
    prior = MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device))

    ##########################
    # Train & Validate
    ##########################
    train_loss_log, val_loss_log = [], []
    pbar = tqdm.trange(1, 20 + 1) #epoch
    for epoch in pbar:
        # Train model
        if use_vae:
            # Train generative model
            train_loss, zs = train_vae(epoch, train_loader, model, prior, optimiser, device)
        else:
            train_loss, model = train_sklearn(epoch, train_dataset, model)
        pbar.set_description(f"Epoch: {epoch} | Train Loss: {train_loss}")
        train_loss_log.append(train_loss)

        # Validate model
        if use_vae:
            val_loss = validate_vae(epoch, val_loader, model, prior, device)
        else:
            val_loss = validate_sklearn(epoch, val_dataset, model)
        pbar.set_description(f"Epoch: {epoch} | Val Loss: {val_loss}")

        # Save best model 
        if len(val_loss_log) == 0 or val_loss < min(val_loss_log):
            filename = os.path.join("results", f"{dataset}_{benchmark}_{seed}.pth")
            pickle.dump(model, open(filename, 'wb'))
        # Early stopping on validation loss
        if len(val_loss_log[-patience:]) >= patience and val_loss >= max(val_loss_log[-patience:]):
            print(f"Early stopping at epoch {epoch}")
            break
        val_loss_log.append(val_loss)

        # Plot losses
        if visualize:
            plot_line(range(1, epoch + 1), train_loss_log, filename=f"{dataset}_{model_name}_loss_train", xlabel="Epoch", ylabel="Training Loss")
            plot_line(range(1, epoch + 1), val_loss_log, filename=f"{dataset}_{model_name}_loss_val", xlabel="Epoch", ylabel="Validation Loss")

        # Visualise model performance
        if visualize:
            if use_vae:
                with torch.no_grad():
                    samples = model.decode(prior.sample((100, ))).sample().cpu()
                    plot_data([train_dataset, samples], ["training", model_name], train_dataset, prefix=f"{dataset}_{model_name}", suffix=str(epoch))
                    #if args.vis_latents: 
                    #    prior_means = torch.stack([prior.mean])
                    #    plot_data([prior.sample((2000, )).cpu(), zs.cpu(), prior_means.cpu()], 
                    #              ["Prior Samples", "Posterior Samples", "Prior Means"], "",
                    #              prefix=f"{dataset}_gaussian_latents", suffix=str(epoch), xlim=[-8, 8], ylim=[-8, 8])
            else:
                plot_data([train_dataset, model], ["training", model_name], train_dataset, prefix=f"{dataset}_{model_name}", suffix=str(epoch))

    # Calculate summary statistics for DoSE later
    if use_vae: 
        # Calculate marginal posterior distribution q(Z)
        marginal_posterior = get_marginal_posterior(train_loader, model, device)
        # Collect summary statistics of model on datasets
        train_summary_stats = get_summary_stats(train_loader, model, marginal_posterior, 16, 4, seed, device)
        val_summary_stats = get_summary_stats(val_loader, model, marginal_posterior, 16, 4, seed, device)
        test_summary_stats = get_summary_stats(test_loader, model, marginal_posterior, 16, 4, seed, device)
        # Save summary statistics
        torch.save(train_summary_stats, os.path.join("stats", f"{dataset}_{benchmark}_{seed}_stats_train.pth"))
        torch.save(val_summary_stats, os.path.join("stats", f"{dataset}_{benchmark}_{seed}_stats_val.pth"))
        torch.save(test_summary_stats, os.path.join("stats", f"{dataset}_{benchmark}_{seed}_stats_test.pth"))
    print(f"Min Val Loss: {min(val_loss_log)}")  # Print minimum validation loss




def main():
    start = datetime.now() # DEBUG
    print("Start: ", start) # DEBUG
    #args = configure()

    train()
    #test()3

    end = datetime.now() # DEBUG
    print(f"Time to Complete: {end - start}") # DEBUG


if __name__ == "__main__":
    main()
