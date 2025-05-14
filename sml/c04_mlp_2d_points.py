from math import ceil
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
    if title:
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

            epoch += 1
            pdf.savefig(fig)
            plt.close(fig)

        # Perceptron final
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(mlp1, x_data_static, y_truth_static,
                          epoch=None, levels=[0.5], ax=ax, title='Perceptron final')

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

            epoch += 1
            pdf.savefig(fig)
            plt.close(fig)

        # Print 2-Layer MLP final
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(
            mlp2, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=ax, title='2-Layer MLP final')

        pdf.savefig(fig)
        plt.close(fig)

        # Print comparison
        mlp1 = args.mlp1
        if mlp1 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            visualize_results(
                mlp1, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=axes[0], title='Perceptron final')
            visualize_results(
                mlp2, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=axes[1], title='2-Layer MLP final')

            axes[0].set_title('Perceptron final')
            axes[1].set_title('2-Layer MLP final')
            plt.suptitle('Comparison of Perceptron and 2-Layer MLP')
            plt.tight_layout()
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

            epoch += 1
            pdf.savefig(fig)
            plt.close(fig)

        # Print 3-Layer MLP final
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        visualize_results(
            mlp3, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=ax, title='3-Layer MLP final')

        pdf.savefig(fig)
        plt.close(fig)

        # Print comparison
        mlp1 = args.mlp1
        mlp2 = args.mlp2
        if mlp1 is not None or mlp2 is not None:
            models = [m for m in (
                (mlp1, 'Perceptron'),
                (mlp2, '2-Layer MLP'),
                (mlp3, '3-Layer MLP')
            ) if m is not None]
            models, titles = zip(*models)
            fig, axes = plt.subplots(1, len(models), figsize=(len(models) * 5 + 1, 5))
            for model, title, ax in zip(models, titles, axes):
                visualize_results(model, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=ax)
                ax.set_title(title)

            plt.suptitle(f'Comparison of {", ".join(titles)}')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_intermediate_representations(mlp_model, plot_title):
    batch_size = 10

    seed = 111
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    x_data, y_target = get_toy_data(batch_size)

    mlp_model(x_data, True)

    x_data = x_data.numpy()
    y_target = y_target.numpy()

    markers = ['o', 'X']

    class_zero_indices = []
    class_one_indices = []
    for i in range(y_target.shape[0]):
        if y_target[i] == 0:
            class_zero_indices.append(i)
        else:
            class_one_indices.append(i)

    class_zero_indices = np.array(class_zero_indices)
    class_one_indices = np.array(class_one_indices)

    len_cache = len(mlp_model.last_forward_cache)
    num_rows = ceil(len_cache / 4)
    num_cols = min(4, len_cache)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

    axes = axes.flatten()
    for i in range(len_cache, len(axes)):
        axes[i].axis('off')

    for class_index, data_indices in enumerate([class_zero_indices, class_one_indices]):

        axes[0].scatter(
            x_data[data_indices, 0],
            x_data[data_indices, 1],
            edgecolor='black',
            facecolor="white",
            marker=markers[class_index],
            s=[200, 200][class_index]
        )
        for i, activations in enumerate(mlp_model.last_forward_cache[1:], 1):
            ax = axes[i]
            ax.scatter(
                activations[data_indices, 0],
                activations[data_indices, 1],
                edgecolor='black',
                facecolor="white",
                marker=markers[class_index],
                s=[200, 200][class_index]
            )

    plt.tight_layout()

    plt.suptitle(plot_title, size=15)
    plt.subplots_adjust(top=0.75)

    return fig


def inspect_representations(args):
    mlp1 = args.mlp1
    mlp2 = args.mlp2
    mlp3 = args.mlp3

    with PdfPages(args.images_path / 'intermediate_representations.pdf') as pdf:
        if mlp1 is not None:
            fig = plot_intermediate_representations(mlp1, 'Perceptron')
            pdf.savefig(fig)
            plt.close(fig)

        if mlp2 is not None:
            fig = plot_intermediate_representations(mlp2, '2-Layer MLP')
            pdf.savefig(fig)
            plt.close(fig)

        if mlp3 is not None:
            fig = plot_intermediate_representations(mlp3, '3-Layer MLP')
            pdf.savefig(fig)
            plt.close(fig)


def main():
    args = Namespace(
        seed=1726,
        batch_size=1000,
        images_path=pathlib.Path('/home/alex/tmp/tmp100/images'),
        run_initial_data_plot=1,
        run_training_perceptron=1,
        mlp1=None,
        run_training_2layer_perceptron=1,
        mlp2=None,
        run_training_3layer_perceptron=1,
        mlp3=None,
        run_inspect_representations=1,
    )
    args.images_path.mkdir(parents=True, exist_ok=True)

    if args.run_initial_data_plot:
        initial_data_plot(args)
        plt.close('all')
    if args.run_training_perceptron:
        training_perceptron(args)
        plt.close('all')
    if args.run_training_2layer_perceptron:
        training_2layer_perceptron(args)
        plt.close('all')
    if args.run_training_3layer_perceptron:
        training_3layer_perceptron(args)
        plt.close('all')
    if args.run_inspect_representations:
        inspect_representations(args)
        plt.close('all')


if __name__ == '__main__':
    main()
