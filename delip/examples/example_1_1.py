import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'Time flies like an arrow.',
    'Fruit flies like a banana.',
]

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()

vocab = one_hot_vectorizer.get_feature_names_out()
print('vocab: ', vocab)

sns.heatmap(
    one_hot,
    annot=True,
    cbar=False,
    xticklabels=vocab,
    yticklabels=['Sentence 1', 'Sentence 2']
)

plt.show()
