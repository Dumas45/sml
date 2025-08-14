from typing import Dict, List

import numpy as np
from annoy import AnnoyIndex


class PreTrainedEmbeddings:
    """A wrapper around pre-trained word vectors and their use."""

    GLOVE_TXT_FILE_PATH = '/home/alex/data/glove/glove.6B.100d.txt'

    def __init__(self, word_to_index: Dict[str, int], word_vectors: List[np.ndarray]):
        """
        Args:
            word_to_index: mapping from word to integers
            word_vectors: list of word vectors,
                where the i-th vector corresponds to the i-th word in the word_to_index mapping
        """
        if len(word_to_index) != len(word_vectors):
            raise ValueError("The length of word_to_index must match the length of word_vectors")

        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = {index: word for word, index in word_to_index.items()}

        self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')

        print("Building Annoy index ...")
        for i, vector in enumerate(word_vectors):
            self.index.add_item(i, vector)

        self.index.build(50)
        print('Finished building Annoy index.')

    @classmethod
    def from_embeddings_file(cls, embeddings_file_path: str) -> 'PreTrainedEmbeddings':
        """
        Load pre-trained embeddings from a file.

        Args:
            embeddings_file_path: path to the embeddings file
        """
        word_to_index = {}
        word_vectors = []

        with open(embeddings_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                word_to_index[word] = len(word_to_index)
                word_vectors.append(vector)

        return cls(word_to_index, word_vectors)

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding for a given word.

        Args:
            word: the word to get the embedding for

        Returns:
            The embedding vector for the word, or None if the word is not found
        """
        index = self.word_to_index.get(word)
        if index is None:
            raise ValueError("The word not found in the embeddings")
        else:
            return self.word_vectors[index]

    def get_closest_to_vector(self, vector: np.ndarray, n: int = 1) -> List[str]:
        """
        Get the closest words to a given vector.

        Args:
            vector: the vector to find the closest words to
            n: number of closest words to return

        Returns:
            A list of the closest words
        """
        indices = self.index.get_nns_by_vector(vector, n)
        return [self.index_to_word[i] for i in indices]

    def compute_and_print_analogy(self, word1: str, word2: str, word3: str):
        """
        Prints the solutions to analogies using word embeddings.

        Analogies are word1 is to word2 as word3 is to __
        This method will print: word1 : word2 :: word3 : word4

        Args:
            word1 (str)
            word2 (str)
            word3 (str)
        """
        vector1 = self.get_embedding(word1)
        vector2 = self.get_embedding(word2)
        vector3 = self.get_embedding(word3)

        analogy_vector = vector2 - vector1 + vector3
        closest_words = self.get_closest_to_vector(analogy_vector, n=4)

        if not closest_words:
            raise ValueError("No closest words found for the analogy")

        closest_words = [w for w in closest_words if w not in (word1, word2, word3)]

        for word in closest_words:
            print(f"{word1} : {word2} :: {word3} : {word}")
