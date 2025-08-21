import argparse
import json
import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# pylint: disable=too-many-instance-attributes
class Vocabulary:
    """Class to process text and extract vocabulary for mapping."""

    def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            mask_token (str): the MASK token to add into the Vocabulary; indicates
                a position that will not be used in updating the model's parameters
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary

        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def get_token_to_idx_mapping(self) -> dict:
        return dict(self._token_to_idx)

    def to_serializable(self) -> dict:
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token,
                'mask_token': self._mask_token}

    @classmethod
    def from_serializable(cls, data: dict) -> 'Vocabulary':
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(
            token_to_idx=data['token_to_idx'],
            mask_token=data['mask_token'],
            add_unk=data['add_unk'],
            unk_token=data['unk_token'],
        )

    def add_token(self, token: str) -> int:
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens: list[str]) -> list[int]:
        """Add a list of tokens into the Vocabulary

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token: str) -> int:
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

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

    def lookup_index(self, index: int) -> str:
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError(f"the index ({index:d}) is not in the Vocabulary")
        return self._idx_to_token[index]

    def __str__(self):
        return f"<Vocabulary(size={len(self):d})>"

    def __len__(self):
        return len(self._token_to_idx)


class CBOWVectorizer:
    """The Vectorizer which coordinates the Vocabularies and puts them to use."""

    def __init__(self, cbow_vocab: Vocabulary):
        """
        Args:
            cbow_vocab (Vocabulary): maps words to integers
        """
        self.cbow_vocab = cbow_vocab

    def vectorize(self, context: str, vector_length: int = -1) -> np.ndarray:
        """
        Args:
            context (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        """

        indices = [self.cbow_vocab.lookup_token(token) for token in context.split(' ')]
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.cbow_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, cbow_df: pd.DataFrame) -> 'CBOWVectorizer':
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            cbow_df (pandas.DataFrame): the target dataset
        Returns:
            an instance of the CBOWVectorizer
        """
        cbow_vocab = Vocabulary()
        for _, row in cbow_df.iterrows():
            for token in row.context.split(' '):
                cbow_vocab.add_token(token)
            cbow_vocab.add_token(row.target)

        return cls(cbow_vocab)

    @classmethod
    def from_serializable(cls, data: dict) -> 'CBOWVectorizer':
        cbow_vocab = Vocabulary.from_serializable(data['cbow_vocab'])
        return cls(cbow_vocab=cbow_vocab)

    def to_serializable(self) -> dict:
        return {'cbow_vocab': self.cbow_vocab.to_serializable()}


class CBOWDataset(Dataset):
    def __init__(self, cbow_df: pd.DataFrame, vectorizer: CBOWVectorizer):
        """
        Args:
            cbow_df (pandas.DataFrame): the dataset
            vectorizer (CBOWVectorizer): vectorizer instatiated from dataset
        """
        self.cbow_df = cbow_df
        self._vectorizer = vectorizer

        def measure_len(_context: str) -> int:
            return len(_context.split(" "))

        self._max_seq_length = max(map(measure_len, cbow_df.context))

        self.train_df = self.cbow_df[self.cbow_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.cbow_df[self.cbow_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.cbow_df[self.cbow_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, cbow_csv: str) -> 'CBOWDataset':
        """Load dataset and make a new vectorizer from scratch

        Args:
            cbow_csv (str): location of the dataset
        Returns:
            an instance of CBOWDataset
        """
        cbow_df = pd.read_csv(cbow_csv)
        train_cbow_df = cbow_df[cbow_df.split=='train']
        return cls(cbow_df, CBOWVectorizer.from_dataframe(train_cbow_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, cbow_csv: str, vectorizer_filepath: str) -> 'CBOWDataset':
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            cbow_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of CBOWDataset
        """
        cbow_df = pd.read_csv(cbow_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(cbow_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath: str) -> CBOWVectorizer:
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of CBOWVectorizer
        """
        with open(vectorizer_filepath, 'r', encoding='UTF-8') as fp:
            return CBOWVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath: str) -> None:
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w", encoding='UTF-8') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self) -> CBOWVectorizer:
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split:str = "train") -> None:
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        context_vector = \
            self._vectorizer.vectorize(row.context, self._max_seq_length)
        target_index = self._vectorizer.cbow_vocab.lookup_token(row.target)

        return {'x_data': context_vector,
                'y_target': target_index}

    def get_num_batches(self, batch_size: int) -> int:
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def generate_batches(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool=True,
        drop_last: bool=True,
        device: torch.device = torch.device("cpu")
):
    """
    A generator function which wraps the PyTorch DataLoader.

    It will ensure each tensor is on the write device location.

    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict


class CBOWClassifier(nn.Module):
    """Simplified CBOW Model."""

    def __init__(self, vocabulary_size: int, embedding_size: int, padding_idx: int = 0):
        """
        Args:
            vocabulary_size (int): number of vocabulary items, controls the
                number of embeddings and prediction vector size
            embedding_size (int): size of the embeddings
            padding_idx (int): default 0; Embedding will not use this index
        """
        super().__init__()

        self.embedding =  nn.Embedding(num_embeddings=vocabulary_size,
                                       embedding_dim=embedding_size,
                                       padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size,
                             out_features=vocabulary_size)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        x_embedded_sum = F.dropout(self.embedding(x_in).sum(dim=1), 0.3)
        y_out = self.fc1(x_embedded_sum)

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        return y_out


def make_train_state(args: argparse.Namespace) -> dict:
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


def update_train_state(args: argparse.Namespace, model: nn.Module, train_state: dict) -> dict:
    """Handle the training state updates.

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


def compute_accuracy(y_pred: torch.Tensor, y_target: torch.Tensor) -> float:
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed: int, cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath: str) -> None:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


# pylint: disable=too-many-statements
def train(args: argparse.Namespace):
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

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

    # Initializations
    if args.reload_from_files:
        print("Loading dataset and loading vectorizer")
        dataset = CBOWDataset.load_dataset_and_load_vectorizer(args.cbow_csv,
                                                               args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer")
        dataset = CBOWDataset.load_dataset_and_make_vectorizer(args.cbow_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    classifier = CBOWClassifier(vocabulary_size=len(vectorizer.cbow_vocab),
                                embedding_size=args.embedding_size)

    # Training loop
    classifier = classifier.to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)
    train_state = make_train_state(args)

    epoch_bar = tqdm(desc='training routine',
                     total=args.num_epochs,
                     position=0)

    dataset.set_split('train')
    train_bar = tqdm(desc='split=train',
                     total=dataset.get_num_batches(args.batch_size),
                     position=1,
                     leave=True)

    dataset.set_split('val')
    val_bar = tqdm(desc='split=val',
                   total=dataset.get_num_batches(args.batch_size),
                   position=1,
                   leave=True)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on

            dataset.set_split('train')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(x_in=batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss, acc=running_acc,
                                      epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred =  classifier(x_in=batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                val_bar.set_postfix(loss=running_loss, acc=running_acc,
                                    epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")

    # compute the loss & accuracy on the test set using the best available model
    classifier = CBOWClassifier(vocabulary_size=len(vectorizer.cbow_vocab),
                                embedding_size=args.embedding_size)
    classifier.load_state_dict(torch.load(train_state['model_filename']))
    classifier = classifier.to(args.device)
    loss_func = nn.CrossEntropyLoss()

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred =  classifier(x_in=batch_dict['x_data'])

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print(f"Test loss: {train_state['test_loss']};")
    print(f"Test Accuracy: {train_state['test_acc']}")


def pretty_print(results: Sequence[tuple[str, torch.Tensor]]) -> None:
    """
    Pretty print embedding results.
    """
    for item in results:
        print(f"...[{item[1]:.2f}] - {item[0]}")


def get_closest(
        target_word: str,
        word_to_idx: dict[str, int],
        embeddings: torch.Tensor,
        n: int = 5
) -> Sequence[tuple[str, torch.Tensor]]:
    """
    Get the n closest words to your word.
    """
    # Calculate distances to all other words
    target_word = target_word.lower()
    word_embedding = embeddings[word_to_idx[target_word]]
    distances = []
    for word, index in word_to_idx.items():
        if word in ("<MASK>", target_word):
            continue
        distances.append((word, torch.dist(word_embedding, embeddings[index])))

    results = sorted(distances, key=lambda x: x[1])[:n]
    return results


def do_inference(args: argparse.Namespace):
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

        print("Expanded filepaths: ")
        print(f"\t{args.vectorizer_file}")
        print(f"\t{args.model_state_file}")

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")

    print(f"Using CUDA: {args.cuda}")

    # handle dirs
    handle_dirs(args.save_dir)

    # Initializations
    print("Loading dataset and loading vectorizer")
    dataset = CBOWDataset.load_dataset_and_load_vectorizer(args.cbow_csv,
                                                           args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    classifier = CBOWClassifier(vocabulary_size=len(vectorizer.cbow_vocab),
                                embedding_size=args.embedding_size)
    classifier.load_state_dict(torch.load(args.model_state_file))
    classifier = classifier.to(args.device)

    # Trained Embeddings
    word = input('Enter a word: ')
    embeddings = classifier.embedding.weight.data
    word_to_idx = vectorizer.cbow_vocab.get_token_to_idx_mapping()
    pretty_print(get_closest(word, word_to_idx, embeddings, n=5))

    target_words = ['queen', 'croquet', 'dinah', 'cat', 'cheshire', 'rabbit']

    for target_word in target_words:
        print(f"======={target_word}=======")
        if target_word not in word_to_idx:
            print("Not in vocabulary")
            continue
        pretty_print(get_closest(target_word, word_to_idx, embeddings, n=5))


def main(args: argparse.Namespace):
    print('\nArgs: ', args)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        do_inference(args)
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
    _parser.add_argument('--cbow_csv', type=str, default='delip/data/books/alice_cbow_dataset.csv')
    _parser.add_argument('--vectorizer_file', type=str, default='vectorizer.json')
    _parser.add_argument('--model_state_file', type=str, default='model.pth')
    _parser.add_argument('--save_dir', type=str, default='delip/model_storage/ch5/cbow')
    _parser.add_argument('--embedding_size', type=int, default=400)
    _parser.add_argument('--seed', type=int, default=1337)
    _parser.add_argument('--num_epochs', type=int, default=100)
    _parser.add_argument('--learning_rate', type=float, default=0.0001)
    _parser.add_argument('--batch_size', type=int, default=32)
    _parser.add_argument('--early_stopping_criteria', type=int, default=5)
    _parser.add_argument('--cuda', type=str2bool, default=True)
    _parser.add_argument('--catch_keyboard_interrupt', type=str2bool, default=True)
    _parser.add_argument('--reload_from_files', type=str2bool, default=False)
    _parser.add_argument('--expand_filepaths_to_save_dir', type=str2bool, default=True)
    _parser.add_argument('--mode', type=str, default='train')  # 'train' or 'inference'

    _args = _parser.parse_args()
    main(_args)
