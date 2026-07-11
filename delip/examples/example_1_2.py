import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'Time flies like an arrow.',
    'Fruit flies like a banana.',
]

vectorizer = TfidfVectorizer(binary=True)
data = vectorizer.fit_transform(corpus).toarray()

vocab = vectorizer.get_feature_names_out()
print('vocab: ', vocab)

sns.heatmap(
    data,
    annot=True,
    cbar=False,
    xticklabels=vocab,
    yticklabels=['Sentence 1', 'Sentence 2']
)

plt.show()
