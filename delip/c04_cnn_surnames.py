import argparse
import json
import os

import numpy as np
import pandas as pd

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class Vocabulary:
    """Class to process text and extract vocabulary for mapping."""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)


    def to_serializable(self):
        """Returns a dictionary that can be serialized."""
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """Instantiates the Vocabulary from a serialized dictionary."""
        return cls(**contents)

    def add_token(self, token):
        """
        Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        try:
            index = self._token_to_idx[token]
        except KeyError:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """
        Add a list of tokens into the Vocabulary.

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Retrieve the index associated with the token or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """
        Return the token associated with the index.

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError(f"the index ({index}) is not in the Vocabulary")
        return self._idx_to_token[index]

    def __str__(self):
        return f"<Vocabulary(size={len(self)})>"

    def __len__(self):
        return len(self._token_to_idx)


class SurnameVectorizer:
    """The Vectorizer which coordinates the Vocabularies and puts them to use."""

    def __init__(self, surname_vocab, nationality_vocab, max_surname_length, lower_case=False):
        """
        Args:
            surname_vocab (Vocabulary): maps characters to integers
            nationality_vocab (Vocabulary): maps nationalities to integers
            max_surname_length (int): the length of the longest surname
            lower_case (bool): whether to convert surnames to lower case
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length
        self.lower_case = lower_case

    def vectorize(self, surname):
        """
        Args:
            surname (str): the surname

        Returns:
            one_hot_matrix (np.ndarray): a matrix of one-hot vectors
        """
        if self.lower_case:
            surname = surname.lower()

        one_hot_matrix_size = (len(self.surname_vocab), self._max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

        for position_index, character in enumerate(surname):
            character_index = self.surname_vocab.lookup_token(character)
            one_hot_matrix[character_index][position_index] = 1

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df, lower_case=False):
        """
        Instantiate the vectorizer from the dataset dataframe.

        Args:
            surname_df (pandas.DataFrame): the surnames dataset
            lower_case (bool): whether to convert surnames to lower case
        Returns:
            an instance of the SurnameVectorizer
        """
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for _, row in surname_df.iterrows():
            surname = row.surname
            if lower_case:
                surname = surname.lower()

            max_surname_length = max(max_surname_length, len(surname))

            for letter in surname:
                surname_vocab.add_token(letter)

            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_surname_length, lower_case=lower_case)

    @classmethod
    def from_serializable(cls, contents):
        surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab =  Vocabulary.from_serializable(contents['nationality_vocab'])
        lower_case = contents.get('lower_case', False)
        return cls(
            surname_vocab=surname_vocab,
            nationality_vocab=nationality_vocab,
            max_surname_length=contents['max_surname_length'],
            lower_case=lower_case
        )

    def to_serializable(self):
        return {
            'surname_vocab': self.surname_vocab.to_serializable(),
            'nationality_vocab': self.nationality_vocab.to_serializable(),
            'max_surname_length': self._max_surname_length,
            'lower_case': self.lower_case
        }


# pylint: disable=too-many-instance-attributes
class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        """
        Args:
            surname_df (pandas.DataFrame): the dataset
            vectorizer (SurnameVectorizer): vectorizer instatiated from dataset
        """
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # Class weights
        class_counts = surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv, lower_case=False):
        """
        Load dataset and make a new vectorizer from scratch.

        Args:
            surname_csv (str): location of the dataset
            lower_case (bool): whether to convert surnames to lower case
        Returns:
            an instance of SurnameDataset
        """
        surname_df = pd.read_csv(surname_csv)
        train_surname_df = surname_df[surname_df.split=='train']
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df, lower_case=lower_case))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, surname_csv, vectorizer_filepath):
        """
        Load dataset and the corresponding vectorizer.

        Used in the case in the vectorizer has been cached for re-use.

        Args:
            surname_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of SurnameDataset
        """
        surname_df = pd.read_csv(surname_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(surname_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """
        A static method for loading the vectorizer from file.

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of SurnameVectorizer
        """
        with open(vectorizer_filepath, encoding="UTF-8") as fp:
            return SurnameVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """
        Saves the vectorizer to disk using json.

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w", encoding="UTF-8") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """Returns the vectorizer."""
        return self._vectorizer

    def set_split(self, split="train"):
        """Selects the splits in the dataset using a column in the dataframe."""
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """The primary entry point method for PyTorch datasets.

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        surname_matrix = \
            self._vectorizer.vectorize(row.surname)

        nationality_index = \
            self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_surname': surname_matrix,
                'y_nationality': nationality_index}

    def get_num_batches(self, batch_size):
        """
        Given a batch size, return the number of batches in the dataset.

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device=torch.device("cpu")):
    """
    A generator function which wraps the PyTorch DataLoader.

    It will ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict


class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        """
        Args:
            initial_num_channels (int): size of the incoming feature vector
            num_classes (int): size of the output prediction vector
            num_channels (int): constant channel size to use throughout network
        """
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x, apply_softmax=False):
        """
        The forward pass of the classifier.

        Args:
            x (torch.Tensor): an input data tensor.
                x.shape should be (batch, initial_num_channels, max_surname_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        features = self.convnet(x).squeeze(dim=2)

        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector


def make_train_state(args):
    return {
        'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val': 1e-8,
        'learning_rate': args.learning_rate,
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1,
        'model_filename': args.model_state_file
    }


def update_train_state(args, model, train_state):
    """
    Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_t = train_state['val_loss'][-1]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def test_model(args, dataset, classifier, train_state):
    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred =  classifier(batch_dict['x_surname'])

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_nationality'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_nationality'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc


# pylint: disable=too-many-statements
def train(args):
    # ------------------------------------------------------------
    # Settings and some prep work
    # ------------------------------------------------------------
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

        print("Expanded filepaths: ")
        print(f"\t{args.vectorizer_file}")
        print(f"\t{args.model_state_file}")

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using CUDA: {args.cuda}")

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    handle_dirs(args.save_dir)

    # ------------------------------------------------------------
    # Initializations
    # ------------------------------------------------------------
    if args.reload_from_files:
        # training from a checkpoint
        print("Reloading!")
        dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv, args.vectorizer_file)
    else:
        # create dataset and vectorizer
        print("Creating fresh!")
        dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv, lower_case=args.lower_case)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()
    classifier = SurnameClassifier(
        initial_num_channels=len(vectorizer.surname_vocab),
        num_classes=len(vectorizer.nationality_vocab),
        num_channels=args.num_channels
    )

    train_state = make_train_state(args)

    # -------------------------------------------------------------
    # compute initial loss & accuracy
    # -------------------------------------------------------------
    classifier = classifier.to(args.device)
    test_model(args, dataset, classifier, train_state)
    print()
    print("Initial test results:")
    print("=====================")
    print(f"Test loss: {train_state['test_loss']};")
    print(f"Test Accuracy: {train_state['test_acc']}\n")

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)

    loss_func = nn.CrossEntropyLoss(weight=dataset.class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=1)

    train_state = make_train_state(args)

    epoch_bar = tqdm(desc='training routine', total=args.num_epochs, position=0)

    dataset.set_split('train')
    train_bar = tqdm(desc='split=train', total=dataset.get_num_batches(args.batch_size), position=1, leave=True)

    dataset.set_split('val')
    val_bar = tqdm(desc='split=val', total=dataset.get_num_batches(args.batch_size), position=1, leave=True)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(batch_dict['x_surname'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_nationality'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_nationality'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred =  classifier(batch_dict['x_surname'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_nationality'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_nationality'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=classifier, train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")
    finally:
        train_bar.close()
        val_bar.close()
        epoch_bar.close()

    # -------------------------------------------------------------
    # compute the loss & accuracy on the test set using the best available model
    # -------------------------------------------------------------
    classifier.load_state_dict(torch.load(train_state['model_filename']))
    test_model(args, dataset, classifier, train_state)
    print()
    print("Final test results:")
    print("===================")
    print(f"Test loss: {train_state['test_loss']};")
    print(f"Test Accuracy: {train_state['test_acc']}\n")


def predict_nationality(surname, classifier, vectorizer):
    """
    Predict the nationality from a new surname.

    Args:
        surname (str): the surname to classifier
        classifier (SurnameClassifer): an instance of the classifier
        vectorizer (SurnameVectorizer): the corresponding vectorizer
    Returns:
        a dictionary with the most likely nationality and its probability
    """
    vectorized_surname = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
    result = classifier(vectorized_surname, apply_softmax=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {'nationality': predicted_nationality, 'probability': probability_value}


def predict_topk_nationality(name, classifier, vectorizer, k=5):
    """
    Predict the top K nationalities from a new name.

    Args:
        name (str): the name to classifier
        classifier (SurnameClassifer): an instance of the classifier
        vectorizer (SurnameVectorizer): the corresponding vectorizer
        k (int): the number of top nationalities to return
    Returns:
        list of dictionaries, each dictionary is a nationality and a probability
    """
    vectorized_name = vectorizer.vectorize(name)
    vectorized_name = torch.tensor(vectorized_name).unsqueeze(dim=0)
    prediction_vector = classifier(vectorized_name, apply_softmax=True)
    probability_values, indices = torch.topk(prediction_vector, k=k)

    # returned size is 1,k
    probability_values = probability_values[0].detach().numpy()
    indices = indices[0].detach().numpy()

    results = []
    for prob_value, index in zip(probability_values, indices):
        nationality = vectorizer.nationality_vocab.lookup_index(index)
        results.append({'nationality': nationality, 'probability': prob_value})

    return results


def inference(args):
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

    device = torch.device("cpu")

    # Load the vectorizer
    with open(args.vectorizer_file, encoding="UTF-8") as fp:
        vectorizer_contents = json.load(fp)
    vectorizer = SurnameVectorizer.from_serializable(vectorizer_contents)

    # Load the model
    classifier = SurnameClassifier(
        initial_num_channels=len(vectorizer.surname_vocab),
        num_classes=len(vectorizer.nationality_vocab),
        num_channels=args.num_channels
    )
    classifier.load_state_dict(torch.load(args.model_state_file, map_location=device))
    classifier.eval()
    classifier = classifier.to(device)
    print("Model and vectorizer loaded.")

    new_surname = input("Enter a surname to classify: ")
    prediction = predict_nationality(new_surname, classifier, vectorizer)
    print(f"{new_surname} -> {prediction['nationality']} (p={prediction['probability']:0.2f})")

    # ----------------------------------------------------------------------
    # Top-K Inference
    # ----------------------------------------------------------------------
    k = int(input("How many of the top predictions to see? "))
    if k > len(vectorizer.nationality_vocab):
        print("Sorry! That's more than the # of nationalities we have.. defaulting you to max size :)")
        k = len(vectorizer.nationality_vocab)

    predictions = predict_topk_nationality(new_surname, classifier, vectorizer, k=k)

    print(f"Top {k} predictions:")
    print("===================")
    for prediction in predictions:
        print(f"{new_surname} -> {prediction['nationality']} (p={prediction['probability']:0.2f})")


def main(args):
    print('\nArgs: ', args)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--surname_csv', type=str, default='delip/data/surnames/surnames_with_splits.csv')
    _parser.add_argument('--vectorizer_file', type=str, default='vectorizer.json')
    _parser.add_argument('--model_state_file', type=str, default='model.pth')
    _parser.add_argument('--save_dir', type=str, default='delip/model_storage/ch4/surname_cnn')
    _parser.add_argument('--hidden_dim', type=int, default=100)
    _parser.add_argument('--num_channels', type=int, default=256)
    _parser.add_argument('--seed', type=int, default=1337)
    _parser.add_argument('--num_epochs', type=int, default=100)
    _parser.add_argument('--early_stopping_criteria', type=int, default=5)
    _parser.add_argument('--learning_rate', type=float, default=0.001)
    _parser.add_argument('--dropout_p', type=float, default=0.1)
    _parser.add_argument('--batch_size', type=int, default=128)
    _parser.add_argument('--cuda', type=str2bool, default=True)
    _parser.add_argument('--reload_from_files', type=str2bool, default=False)
    _parser.add_argument('--expand_filepaths_to_save_dir', type=str2bool, default=True)
    _parser.add_argument('--catch_keyboard_interrupt', type=str2bool, default=True)
    _parser.add_argument('--mode', type=str, default='train')  # 'train' or 'inference'
    _parser.add_argument('--lower_case', type=str2bool, default=False)
    _args = _parser.parse_args()
    main(_args)
