from gensim.models import LogEntropyModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
dct = Dictionary(common_texts)  # fit dictionary
corpus = [dct.doc2bow(row) for row in common_texts]  # convert to BoW format
model = LogEntropyModel(corpus, normalize=True)  # fit model
print("Vector 1")
vector1 = model[corpus]
for i in vector1:
    print(i)

tfidf = TfidfModel(corpus=corpus)
vector12 = tfidf[corpus]
print("Vector 2")
for i in vector12:
    print(i)