import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os.path
import torch

from torch import nn
from torch import optim
from torchvision import datasets, transforms
from tqdm import tqdm

INPUT_SHAPE = (28, 28)
INPUT_DIM = INPUT_SHAPE[0] * INPUT_SHAPE[1]

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=16):
        "Build simple Variations Autoencoder."
        super(VAE, self).__init__()

        self.fc_Q = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        self.fc_Ph = nn.Linear(z_dim, h_dim)
        self.fc_P = nn.Linear(h_dim, input_dim)

        # NOTE: size_average=True would average over both batch and spatial dimensions,
        #       but we don't want any averaging or at most over batch dimension only.
        self.reconstruction_loss = nn.BCELoss(size_average=False)
        # NOTE: See Appendix B from VAE paper (Kingma 2014):
        #       https://arxiv.org/abs/1312.6114
        self.KL_loss = lambda mu, logvar: \
            torch.sum((1. + logvar - mu**2 - torch.exp(logvar))) / 2

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'n' is number of inputs to each neuron
                n = len(m.weight.data[1])
                # "Xavier" initialization
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def sample_z(self, mu, logvar):
        """Sample latent variable from Gaussian."""
        eps = torch.randn_like(mu)
        return eps * torch.exp(logvar / 2) + mu

    def encoder(self, x):
        """Encodes input sample and returns latent distribution params."""
        h = torch.relu(self.fc_Q(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def decoder(self, z):
        """Decodes latent variable and returns output sample."""
        h = torch.relu(self.fc_Ph(z))
        return torch.sigmoid(self.fc_P(h))

    def loss(self, x):
        """Compute variational lower bound loss."""
        mu, logvar = self.encoder(x)
        z = self.sample_z(mu, logvar)
        x_ = self.decoder(z)

        return self.reconstruction_loss(x_, x) - self.KL_loss(mu, logvar)

    def load_ckpt(self, path):
        """Saves current model state dict to path."""
        self.load_state_dict(torch.load(path))

    def save_ckpt(self, path):
        """Saves current model state dict to path."""
        torch.save(self.state_dict(), path) 


def plot_samples(samples, sample_shape, grid):
    fig = plt.figure(figsize=grid)
    gs = gridspec.GridSpec(*grid)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(*sample_shape), cmap='Greys_r')


def evaluate(args, net, grid=(4, 4)):
    # Plot random samples
    z = torch.randn(grid[0] * grid[1], args.z_dim)
    net.eval()
    samples = net.decoder(z).detach().numpy()

    plt.close()
    plot_samples(samples, INPUT_SHAPE, grid=grid)
    plt.draw()
    plt.pause(0.001)


def train(args, net):
    # Get training data (MNIST)
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True)

    # Create optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Train for # epochs
    for epoch in range(args.epochs):
        net.train()
        total_loss = 0
        pbar = tqdm(data_loader, ascii=True, desc="Epoch: {:2d}/{}".format(epoch + 1, args.epochs))

        # Train on all dataset examples
        for data, _ in pbar:
            x = data.view(-1, INPUT_DIM)
            optimizer.zero_grad()
            loss = net.loss(x)
            loss.backward()
            optimizer.step()

            # Bookkeeping
            if total_loss == 0:
                total_loss = loss.item()
            else:
                total_loss = total_loss * .95 + loss.item() * .05
            pbar.set_postfix(loss=total_loss)

        # Plot example outputs
        evaluate(args, net)

        # Save checkpoint if path given
        if args.ckpt is not None:
            net.save_ckpt(args.ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--ckpt', type=str, default=None, metavar='PATH',
                        help='checkpoint path to load from/save to model (default: None)')
    parser.add_argument('--z_dim', type=int, default=20, metavar='N',
                        help='dimensionality of latent variable (default: 20)')
    parser.add_argument('--h_dim', type=int, default=400, metavar='N',
                        help='size of hidden layer (default: 400)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Disable training, evaluate checkpoint (default: False)')
    args = parser.parse_args()

    # Create net
    net = VAE(input_dim=INPUT_DIM, h_dim=args.h_dim, z_dim=args.z_dim)

    # Load checkpoint if available
    if args.ckpt is not None and os.path.exists(args.ckpt):
        net.load_ckpt(args.ckpt)

    # Run training
    if not args.eval:
        train(args, net)
    elif args.ckpt is not None:
        evaluate(args, net, grid=(8,8))
    else:
        print("[!] You need to pass path to checkpoint in order to evaluate model.")
        exit(1)
        
    # Stop before exit
    plt.show()

