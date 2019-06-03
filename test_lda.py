import gensim

from gensim.test.utils import get_tmpfile
from gensim import corpora


output_fname = get_tmpfile(dump_path + "test-serialize-tfidf.mm")
gensim.corpora.mmcorpus.MmCorpus.serialize(output_fname, corpus_tfidf)

serialize_corpus_tfidf = corpora.MmCorpus(output_fname)


from gensim.sklearn_api import TfIdfTransformer
tfidf = TfIdfTransformer()
pdb.set_trace()
corpus_tfidf_ = tfidf.fit_transform(corpus)