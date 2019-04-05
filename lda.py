import itertools as it
import logging
import time
import numpy as np
import pickle
import gensim
import multiprocessing as mp

from collections import OrderedDict
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.tfidfmodel import TfidfModel
from multiprocessing import Pool, Array, Value
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import get_tmpfile

# CONFIG FOR LOGGING MEMORY_PROFILER
import sys
from memory_profiler import profile, LogFile

sys.stdout = LogFile(__name__)


@profile
def genkmers(k):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in it.product(bases, repeat=k)]


@profile
def extract_k_mers(sequence, k):
    res = []
    for i in range(len(sequence) - k + 1):
        res.append(sequence[i:i + k])
    return res


def create_document(reads, k=[]):
    """
    Create a set of document from reads, consist of all k-mer in each read
    For example:
    k = [3, 4, 5]
    documents =
    [
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 1
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 2
        ...
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read n
    ]
    :param reads:
    :param k: list of int
    :return: list of str
    """
    # create a set of document
    documents = []
    for read in reads:
        k_mers_read = []
        for value in k:
            k_mers_read += [read[j:j + value] for j in range(0, len(read) - value + 1)]
        documents.append(k_mers_read)
    return documents


def save_documents(documents, file_path):
    with open(file_path, 'w') as f:
        for d in documents:
            f.write("%s\n" % d)


@profile
def parallel_create_document(reads, k=[], n_workers=2):
    """
    Create a set of document from reads, consist of all k-mer in each read
    For example:
    k = [3, 4, 5]
    documents =
    [
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 1
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 2
        ...
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read n
    ]
    :param reads:
    :param k: list of int
    :return: list of str
    """

    t1 = time.time()
    # create k-mer dictionary
    k_mers_set = [genkmers(val) for val in k]
    logging.info("Creating k-mers dictionary ...")
    dictionary = corpora.Dictionary(k_mers_set)

    documents = []
    reads_str_chunk = [list(item) for item in np.array_split(reads, n_workers)]
    chunks = [(reads_str_chunk[i], k) for i in range(n_workers)]
    pool = Pool(processes=n_workers)

    result = pool.starmap(create_document, chunks)
    for item in result:
        documents += item
    t2 = time.time()
    logging.info("Finished create document in %f (s)." % (t2 - t1))
    return dictionary, documents


@profile
def create_corpus(dictionary, documents, is_tfidf=False, smartirs=None):
    logging.info("Creating corpus ...")
    t1 = time.time()
    corpus = [dictionary.doc2bow(d, allow_update=True) for d in documents]
    if is_tfidf:
        logging.info("Creating corpus with TFIDF ...")
        tfidf = TfidfModel(corpus=corpus, smartirs=smartirs)
        corpus = tfidf[corpus]
    t2 = time.time()
    logging.info("Finished creating corpus in %f (s)." % (t2 - t1))
    return corpus


@profile
def serializeCorpus(corpus_tfidf, dump_path, file_name):
    logging.info("Seriallize and creating Mmcorpus")
    t1 = time.time()
    output_fname = get_tmpfile(dump_path + 'corpus-tfidf-%s.mm' % file_name)
    gensim.corpora.mmcorpus.MmCorpus.serialize(output_fname, corpus_tfidf)

    serialize_corpus_tfidf = corpora.MmCorpus(output_fname)
    t2 = time.time()
    logging.info("Finished creating Mmcorpus in %f (s)." % (t2 - t1))
    return serialize_corpus_tfidf


@profile
def getDocTopicDist(model, corpus, kwords=False, n_topics=10):
    """
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    :param model: LDA model object
    :param corpus:
    :param kwords:
    :return:
    """
    t1 = time.time()
    top_dist = []
    keys = []
    logging.info("Getting topic distribution ...")
    for d in corpus:
        tmp = {i: 0 for i in range(n_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [np.array(vals)]
        if kwords:
            keys += [np.array(vals).argmax()]
    t2 = time.time()
    logging.info("Finished get topic distribution in %f (s)." % (t2 - t1))
    return np.array(top_dist), keys


@profile
def do_LDA(corpus, dictionary, n_topics=10, n_worker=2, n_passes=15, max_iters=200):
    t1 = time.time()
    # training LDA model
    lda_model = LdaMulticore(corpus, id2word=dictionary, num_topics=n_topics, passes=n_passes, workers=n_worker,
                             iterations=max_iters)
    t2 = time.time()
    logging.info("Finished training LDA model in %f (s)." % (t2 - t1))
    return lda_model


@profile
def do_LDA_Mallet(path_to_mallet_binary, corpus, dictionary, n_topics=10, n_worker=2):
    """
    A wapper of LDA Mallet
    If the program is run out of memory, you should try to use LDA or LDA LdaMulticore
    :param corpus:
    :param dictionary:
    :param n_topics:
    :param n_worker:
    :param n_passes:
    :param max_iters:
    :return:
    """
    t1 = time.time()
    # training LDA model
    lda_model = LdaMallet(mallet_path=path_to_mallet_binary,
                          corpus=corpus, id2word=dictionary, num_topics=n_topics, workers=n_worker,
                          optimize_interval=100, iterations=200)
    t2 = time.time()
    logging.info("Finished training LDA Mallet model in %f (s)." % (t2 - t1))
    return lda_model


@profile
def load_LDAModel(path, file_name):
    lda = gensim.models.ldamodel.LdaModel.load(path + file_name)
    return lda


@profile
def load_dictionary(path, file_name):
    dictionary = gensim.corpora.Dictionary.load(path + file_name)
    return dictionary


@profile
def load_corpus(path, file_name):
    corpus = pickle.load(open(path + file_name, 'rb'))
    return corpus
