import pathlib
from argparse import Namespace
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LABELS = [0, 0, 1, 1]
CENTERS = [(-3, -3), (3, 3), (3, -3), (-3, 3)]


def get_toy_data(batch_size):
    assert len(CENTERS) == len(LABELS), 'centers should have equal number labels'

    x_data = []
    y_targets = np.zeros(batch_size)
    n_centers = len(CENTERS)

    for batch_i in range(batch_size):
        center_idx = np.random.randint(0, n_centers)
        x_data.append(np.random.normal(loc=CENTERS[center_idx]))
        y_targets[batch_i] = LABELS[center_idx]

    x_data = np.array(x_data)

    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.int64)


class MultilayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_size=2, output_size=3,
                 num_hidden_layers=1, hidden_activation=nn.Sigmoid):
        """Initialize weights.

        Args:
            input_size (int): size of the input
            hidden_size (int): size of the hidden layers
            output_size (int): size of the output
            num_hidden_layers (int): number of hidden layers
            hidden_activation (torch.nn.*): the activation class
        """
        super().__init__()
        self.module_list = nn.ModuleList()

        interim_input_size = input_size
        interim_output_size = hidden_size

        for _ in range(num_hidden_layers):
            self.module_list.append(nn.Linear(interim_input_size, interim_output_size))
            self.module_list.append(hidden_activation())
            interim_input_size = interim_output_size

        self.fc_final = nn.Linear(interim_input_size, output_size)

        self.last_forward_cache = []

    def forward(self, x, apply_softmax=False):
        """The forward pass of the MLP

        Args:
            x (torch.Tensor): an input data tensor.
                x.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        self.last_forward_cache = []
        self.last_forward_cache.append(x.to("cpu").detach().numpy())

        for module in self.module_list:
            x = module(x)
            self.last_forward_cache.append(x.to("cpu").detach().numpy())

        output = self.fc_final(x)
        self.last_forward_cache.append(output.to("cpu").detach().numpy())

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output


def visualize_results(perceptron, x_data, y_truth, ax=None, epoch=None, title='', levels=None, linestyles=None):
    if levels is None:
        levels = [0.3, 0.4, 0.5]

    if linestyles is None:
        linestyles = ['--', '-', '--']

    _, y_pred = perceptron(x_data, apply_softmax=True).max(dim=1)
    y_pred = y_pred.data.numpy()

    x_data = x_data.data.numpy()
    y_truth = y_truth.data.numpy()

    n_classes = len(set(LABELS))

    all_x = [[] for _ in range(n_classes)]
    all_colors: List = [[] for _ in range(n_classes)]

    colors = ['black', 'white']
    markers = ['o', 'X']

    for x_i, y_pred_i, y_true_i in zip(x_data, y_pred, y_truth):
        all_x[y_true_i].append(x_i)
        if y_pred_i == y_true_i:
            all_colors[y_true_i].append("white")
        else:
            all_colors[y_true_i].append("black")

    all_x = [np.stack(x_list) for x_list in all_x]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    for x_list, color_list, marker in zip(all_x, all_colors, markers):
        ax.scatter(x_list[:, 0], x_list[:, 1], edgecolor="black", marker=marker, facecolor=color_list, s=100)

    xlim = (min(x_list[:, 0].min() for x_list in all_x),
            max(x_list[:, 0].max() for x_list in all_x))

    ylim = (min(x_list[:, 1].min() for x_list in all_x),
            max(x_list[:, 1].max() for x_list in all_x))

    # hyperplane
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    for i in range(n_classes):
        Z = perceptron(torch.tensor(xy, dtype=torch.float32),
                       apply_softmax=True)
        Z = Z[:, i].data.numpy().reshape(XX.shape)
        ax.contour(XX, YY, Z, colors=colors[i], levels=levels, linestyles=linestyles)

    # plotting niceties
    plt.suptitle(title)

    if epoch is not None:
        plt.text(xlim[0], ylim[1], f'Epoch = {epoch}')


def initial_data_plot(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    x_data, y_truth = get_toy_data(batch_size=args.batch_size)

    x_data = x_data.data.numpy()
    y_truth = y_truth.data.numpy().astype(np.int64)

    n_classes = len(set(LABELS))

    all_x = [[] for _ in range(n_classes)]
    all_colors = [[] for _ in range(n_classes)]

    colors = ['black', 'white']
    markers = ['o', 'X']

    for x_i, y_true_i in zip(x_data, y_truth):
        all_x[y_true_i].append(x_i)
        all_colors[y_true_i].append(colors[y_true_i])

    all_x = [np.stack(x_list) for x_list in all_x]

    _, ax = plt.subplots(1, 1, figsize=(10, 5))

    for x_list, marker in zip(all_x, markers):
        ax.scatter(x_list[:, 0], x_list[:, 1], edgecolor='black', marker=marker, facecolor="white", s=100)

    plt.tight_layout()
    plt.axis('off')

    plt.title("")

    plt.savefig(args.images_path / "data.png")
    plt.savefig(args.images_path / "data.pdf")


def early_termination(last_loss, loss_threshold, loss_change, change_threshold, epoch, max_epochs):
    terminate_for_loss = last_loss < loss_threshold
    terminate_for_loss_change = loss_change < change_threshold
    terminate_for_epochs = epoch > max_epochs

    return terminate_for_epochs or terminate_for_loss_change or terminate_for_loss


def training_perceptron(args):
    input_size = 2
    output_size = len(set(LABELS))
    num_hidden_layers = 0
    hidden_size = 2  # isn't ever used but we still set it

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    mlp1 = MultilayerPerceptron(input_size=input_size,
                                hidden_size=hidden_size,
                                num_hidden_layers=num_hidden_layers,
                                output_size=output_size)

    print(mlp1)
    args.mlp1 = mlp1
    batch_size = args.batch_size

    x_data_static, y_truth_static = get_toy_data(batch_size)

    with PdfPages(args.images_path / 'Perceptron.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(mlp1, x_data_static, y_truth_static,
                          ax=ax, title='Initial Perceptron State', levels=[0.5])

        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        losses = []
        n_batches = 10
        max_epochs = 10

        loss_value = 10.0
        loss_change = 1.0
        last_loss = 10.0
        loss_threshold = 0.3
        change_threshold = 1e-3
        epoch = 0

        lr = 0.01
        optimizer = optim.Adam(params=mlp1.parameters(), lr=lr)
        cross_ent_loss = nn.CrossEntropyLoss()

        while not early_termination(last_loss, loss_threshold, loss_change, change_threshold, epoch, max_epochs):
            for _ in range(n_batches):
                # step 0: fetch the data
                x_data, y_target = get_toy_data(batch_size)

                # step 1: zero the gradients
                mlp1.zero_grad()

                # step 2: run the forward pass
                y_pred = mlp1(x_data).squeeze()

                # step 3: compute the loss
                loss = cross_ent_loss(y_pred, y_target.long())

                # step 4: compute the backward pass
                loss.backward()

                # step 5: have the optimizer take an optimization step
                optimizer.step()

                # auxillary: bookkeeping
                loss_value = loss.item()
                losses.append(loss_value)
                loss_change = abs(last_loss - loss_value)
                last_loss = loss_value

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            visualize_results(mlp1, x_data_static, y_truth_static, ax=ax, epoch=epoch,
                              title=f"{loss_value:0.2f}; {loss_change:0.4f}")
            plt.axis('off')
            epoch += 1
            pdf.savefig(fig)
            plt.close(fig)

        # Perceptron final
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(mlp1, x_data_static, y_truth_static,
                          epoch=None, levels=[0.5], ax=ax, title='Perceptron final')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


def training_2layer_perceptron(args):
    input_size = 2
    output_size = len(set(LABELS))
    num_hidden_layers = 1
    hidden_size = 2

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    mlp2 = MultilayerPerceptron(input_size=input_size,
                                hidden_size=hidden_size,
                                num_hidden_layers=num_hidden_layers,
                                output_size=output_size)

    print(mlp2)
    args.mlp2 = mlp2
    batch_size = args.batch_size

    x_data_static, y_truth_static = get_toy_data(batch_size)

    with PdfPages(args.images_path / '2-Layer-MLP.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(mlp2, x_data_static, y_truth_static,
                          ax=ax, title='Initial 2-Layer MLP State', levels=[0.5])

        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        losses = []
        n_batches = 10
        max_epochs = 15

        loss_value = 10.0
        loss_change = 1.0
        last_loss = 10.0
        loss_threshold = 0.3
        change_threshold = 1e-5
        epoch = 0

        lr = 0.01
        optimizer = optim.Adam(params=mlp2.parameters(), lr=lr)
        cross_ent_loss = nn.CrossEntropyLoss()

        while not early_termination(last_loss, loss_threshold, loss_change, change_threshold, epoch, max_epochs):
            for _ in range(n_batches):
                # step 0: fetch the data
                x_data, y_target = get_toy_data(batch_size)

                # step 1: zero the gradients
                mlp2.zero_grad()

                # step 2: run the forward pass
                y_pred = mlp2(x_data).squeeze()

                # step 3: compute the loss
                loss = cross_ent_loss(y_pred, y_target.long())

                # step 4: compute the backward pass
                loss.backward()

                # step 5: have the optimizer take an optimization step
                optimizer.step()

                # auxillary: bookkeeping
                loss_value = loss.item()
                losses.append(loss_value)
                loss_change = abs(last_loss - loss_value)
                last_loss = loss_value

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            visualize_results(mlp2, x_data_static, y_truth_static, ax=ax, epoch=epoch,
                              title=f"{loss_value:0.2f}; {loss_change:0.4f}")
            plt.axis('off')
            epoch += 1
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(
            mlp2, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=ax, title='2-Layer MLP final')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


def training_3layer_perceptron(args):
    input_size = 2
    output_size = len(set(LABELS))
    num_hidden_layers = 2
    hidden_size = 2

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    mlp3 = MultilayerPerceptron(input_size=input_size,
                                hidden_size=hidden_size,
                                num_hidden_layers=num_hidden_layers,
                                output_size=output_size)

    print(mlp3)
    args.mlp3 = mlp3
    batch_size = args.batch_size

    x_data_static, y_truth_static = get_toy_data(batch_size)

    with PdfPages(args.images_path / '3-Layer-MLP.pdf') as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(mlp3, x_data_static, y_truth_static,
                          ax=ax, title='Initial 3-Layer MLP State', levels=[0.5])

        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        losses = []
        n_batches = 10
        max_epochs = 15

        loss_value = 10.0
        loss_change = 1.0
        last_loss = 10.0
        loss_threshold = 0.3
        change_threshold = 1e-5
        epoch = 0

        lr = 0.01
        optimizer = optim.Adam(params=mlp3.parameters(), lr=lr)
        cross_ent_loss = nn.CrossEntropyLoss()

        while not early_termination(last_loss, loss_threshold, loss_change, change_threshold, epoch, max_epochs):
            for _ in range(n_batches):
                # step 0: fetch the data
                x_data, y_target = get_toy_data(batch_size)

                # step 1: zero the gradients
                mlp3.zero_grad()

                # step 2: run the forward pass
                y_pred = mlp3(x_data).squeeze()

                # step 3: compute the loss
                loss = cross_ent_loss(y_pred, y_target.long())

                # step 4: compute the backward pass
                loss.backward()

                # step 5: have the optimizer take an optimization step
                optimizer.step()

                # auxillary: bookkeeping
                loss_value = loss.item()
                losses.append(loss_value)
                loss_change = abs(last_loss - loss_value)
                last_loss = loss_value

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            visualize_results(mlp3, x_data_static, y_truth_static, ax=ax, epoch=epoch,
                              title=f"{loss_value:0.2f}; {loss_change:0.4f}")
            plt.axis('off')
            epoch += 1
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(
            mlp3, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=ax, title='3-Layer MLP final')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = Namespace(
        seed=1732,
        batch_size=1000,
        images_path=pathlib.Path('/home/alex/tmp/tmp100/images'),
        run_initial_data_plot=1,
        run_training_perceptron=1,
        mlp1=None,
        run_training_2layer_perceptron=1,
        mlp2=None,
        run_training_3layer_perceptron=1,
        mlp3=None,
    )
    args.images_path.mkdir(parents=True, exist_ok=True)

    if args.run_initial_data_plot:
        initial_data_plot(args)
    if args.run_training_perceptron:
        training_perceptron(args)
    if args.run_training_2layer_perceptron:
        training_2layer_perceptron(args)
    if args.run_training_3layer_perceptron:
        training_3layer_perceptron(args)


if __name__ == '__main__':
    main()
