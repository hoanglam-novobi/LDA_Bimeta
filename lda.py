import itertools as it
import logging
import time
import numpy as np
import pickle
import gensim
import multiprocessing as mp

from collections import OrderedDict
from gensim import corpora
from gensim.models import LogEntropyModel
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.tfidfmodel import TfidfModel
from multiprocessing import Pool, Array, Value
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import get_tmpfile
from gensim.matutils import corpus2dense, Dense2Corpus

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
def create_corpus(dictionary, documents, is_tfidf=False, smartirs=None, is_log_entropy=False, is_normalize=True):
    logging.info("Creating BoW corpus ...")
    t1 = time.time()
    corpus = [dictionary.doc2bow(d, allow_update=True) for d in documents]
    if is_tfidf:
        logging.info("Creating corpus with TFIDF ...")
        tfidf = TfidfModel(corpus=corpus, smartirs=smartirs)
        corpus = tfidf[corpus]
    elif is_log_entropy:
        logging.info("Creating corpus with Log Entropy ...")
        log_entropy_model = LogEntropyModel(corpus, normalize=is_normalize)
        corpus = log_entropy_model[corpus]

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


def getDocTopicDist(model, corpus, n_topics, kwords=False):
    """
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    """
    top_dist = []
    keys = []

    for d in corpus:
        tmp = {i: 0 for i in range(n_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [np.array(vals)]
        if kwords:
            keys += [np.array(vals).argmax()]
    return top_dist, keys



@profile
def getDocTopicDist_mp(model, corpus, n_topics=10, kwords=False, n_workers=20):
    top_dist = []
    keys = []
    corpus_chunk = [list(item) for item in np.array_split(corpus, n_workers)]
    chunks = [(model, corpus_chunk[i], n_topics, True) for i in range(n_workers)]
    pool = Pool(processes=n_workers)

    result = pool.starmap(getDocTopicDist, chunks)

    for item in result:
        top_dist += item[0]
        keys += item[1]
    return np.array(top_dist), keys


@profile
def do_LDA(corpus, dictionary, n_topics=10, n_worker=2, n_passes=15, max_iters=200):
    t1 = time.time()
    # training LDA model
    lda_model = LdaMulticore(corpus, id2word=dictionary, num_topics=n_topics, passes=n_passes, workers=n_worker,
                             iterations=max_iters, alpha=1.0, eval_every=20, decay=0.7, eta=0.1)
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
                          optimize_interval=20, iterations=200)
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


def BM25_transform(term_frequency_matrix, k=1.2, b=0.75):
    """
    Implement BM25 ranking function
    A1 = (k + 1).tf(w, d)
    A2 = tf(w, d) + k(1 - b + b.(|d|/avdl))
    B = ln((C + 1)/df(W))
    :param term_frequency_matrix: a Numpy array
    :param k,b: parameters of BM25 ranking algoritms
    :return: a Numpy array with the same shape
    """
    # make a copy of matrix
    data = np.copy(term_frequency_matrix)
    # compute number of documents: C
    C = data.shape[0]
    print("Number of document in the corpus: ", C)
    # compute |d|
    number_of_words_in_documents = np.sum(data, axis=1)
    print("Number of words in each documents: ", len(number_of_words_in_documents), number_of_words_in_documents)
    # compute avdl
    avg_document_length = np.mean(number_of_words_in_documents)
    print("Average of document length: ", avg_document_length)
    # compute df(W)
    count_word_frequencies = np.sum(data, axis=0)
    word_document_frequencies = np.where(count_word_frequencies > 0, count_word_frequencies, 0)
    print("Frequency of work in document: ", len(word_document_frequencies))

    # compute the component A1
    A1 = (k + 1) * data
    print("A1 :", A1.shape)

    # compute the component A2
    A2 = data + k * (1 - b + b * number_of_words_in_documents.reshape(-1, 1) / avg_document_length)
    print("A2 :", A2.shape)

    # compute the component B
    B = np.log((C + 1) / word_document_frequencies.reshape(1, -1))
    print("B: ", B.shape, np.sum(np.where(B < 0, 1, 0)))

    result = A1 * B / A2
    return result


def pivoted_length_transform(term_frequency_matrix, b=0.4):
    """
    Implement Pivoted Length Normalization
    A = ln(1 + ln(1 + tf(w, d)))/(1 - b + b*|d|/avdl)*log((C + 1)/df(w))
    B = ln((C + 1)/df(W))
    :param term_frequency_matrix: a Numpy array
    :param b: parameters of Pivoted Length Normalization algoritm
    :return: a Numpy array with the same shape
    """
    # make a copy of matrix
    data = np.copy(term_frequency_matrix)
    # compute number of documents: C
    C = data.shape[0]
    print("Number of document in the corpus: ", C)
    # compute |d|
    number_of_words_in_documents = np.sum(data, axis=1)
    print("Number of words in each documents: ", len(number_of_words_in_documents), number_of_words_in_documents)
    # compute avdl
    avg_document_length = np.mean(number_of_words_in_documents)
    print("Average of document length: ", avg_document_length)
    # compute df(W)
    count_word_frequencies = np.sum(data, axis=0)
    word_document_frequencies = np.where(count_word_frequencies > 0, count_word_frequencies, 0)
    print("Frequency of work in document: ", len(word_document_frequencies))

    # compute component A
    A = (1 / (1 - b + b * number_of_words_in_documents.reshape(-1, 1) / avg_document_length)) * np.log(
        1 + np.log(1 + data))
    print("A: ", A.shape)

    # compute the component B
    B = np.log2((C + 1) / word_document_frequencies.reshape(1, -1))
    print("B: ", B.shape, np.sum(np.where(B < 0, 1, 0)))

    result = A * B
    return result


def jelinek_mercer_pior_transform(term_frequency_matrix, a=0.2):
    """
    Implement Jelinek-Mercer prior
    A = (1 - a)*(tf(w, d)/|d|)
    B = a*p(w|C)
    :param term_frequency_matrix: a Numpy array
    :param a: lambda used in the fomular
    :return: a Numpy array with the same shape
    """
    # make a copy of matrix
    data = np.copy(term_frequency_matrix)
    # compute number of documents: C
    C = data.shape[0]
    print("Number of document in the corpus: ", C)
    # compute |d|
    number_of_words_in_documents = np.sum(data, axis=1)
    print("Number of words in each documents: ", len(number_of_words_in_documents), number_of_words_in_documents.shape)

    # compute p(w|C)
    prob_word_per_corpus = data / (sum(number_of_words_in_documents))

    # compute A
    A = (1 - a) * data / number_of_words_in_documents.reshape(-1, 1)

    B = a * prob_word_per_corpus
    result = A * B
    return result


def dirichlet_prior_transform(term_frequency_matrix, u=2000):
    """
    Implement Dirichlet prior
    A = tf(w,d/(|d| + u))
    B = u*p(w}C)/(|d| + u)
    :param term_frequency_matrix: a Numpy array
    :param u: parameter used in the fomular
    :return: a Numpy array with the same shape
    """
    # make a copy of matrix
    data = np.copy(term_frequency_matrix)
    # compute number of documents: C
    C = data.shape[0]
    print("Number of document in the corpus: ", C)
    # compute |d|
    number_of_words_in_documents = np.sum(data, axis=1)
    print("Number of words in each documents: ", len(number_of_words_in_documents), number_of_words_in_documents.shape)

    # compute p(w|C)
    prob_word_per_corpus = data / (sum(number_of_words_in_documents))

    # compute A
    A = data / (number_of_words_in_documents.reshape(-1, 1) + u)

    B = u * prob_word_per_corpus / (number_of_words_in_documents.reshape(-1, 1) + u)
    result = A * B
    return result


def PL2_transform(term_frequency_matrix, c=1):
    """
    Implement PL2
    A = tf(w, d)*log(1 + c*avdl/|d|)
    B = |C|/tf(w, C)
    :param term_frequency_matrix: a Numpy array
    :param c: parameter used in the fomular
    :return: a Numpy array with the same shape
    """
    # make a copy of matrix
    data = np.copy(term_frequency_matrix)
    # compute number of documents: C
    C = data.shape[0]
    print("Number of document in the corpus: ", C)
    # compute |d|
    number_of_words_in_documents = np.sum(data, axis=1)
    print("Number of words in each documents: ", len(number_of_words_in_documents), number_of_words_in_documents.shape)
    # compute avdl
    avg_document_length = np.mean(number_of_words_in_documents)
    print("Average of document length: ", avg_document_length)
    # compute A
    A = data * np.log2(1 + c * avg_document_length / number_of_words_in_documents.reshape(-1, 1))
    print("A: ", A.shape)
    # compute B
    word_frequency_per_corpus = np.sum(data, axis=0).reshape(-1, 1)
    print("Word frequency per corpus: ", word_frequency_per_corpus.shape)
    B = C / word_frequency_per_corpus
    result = A * np.log2(B * A) + np.log2(np.e) * (1 / B - A) + 0.5 * np.log2(2 * np.pi * A) / (A + 1)
    return result
