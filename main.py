import os
import time
import logging
import pickle

from read_fasta import read_fasta_file, create_labels
from lda import create_document, create_corpus, do_LDA, getDocTopicDist
from kmeans import do_kmeans, evalQualityCluster

# LOGGING
logging.basicConfig(level=logging.INFO,
                    filename='LDABimeta.log',
                    filemode='a', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    data_path = 'D:/HOCTAP/HK181/DeCuongLV/datasets/bimeta_dataset/'
    extension_file = '.fna'
    name = 'R4'
    file_name = name + extension_file
    OUTPUT_DIR = 'D:/HOCTAP/HK182/Luanvantotnghiep/sourcecode/output/' + name
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logging.info("++++++++++++++++++++ START ++++++++++++++++++++")
    ##########################################
    # READING FASTA FILE
    ##########################################
    reads = read_fasta_file(data_path + file_name, pair_end=False)

    ##########################################
    # CREATE DOCUMENTS, CORPUS
    ##########################################
    k = [3, 4, 5]
    dictionary, documents = create_document(reads, k=k)
    logging.info("Writing dictionary into %s." % OUTPUT_DIR)
    dictionary.save(OUTPUT_DIR + 'dictionary-k%s.gensim' % (''.join(str(val) for val in k)))

    # if you want to create corpus with TFIDF, set it is True
    is_tfidf = False
    corpus = create_corpus(dictionary, documents, is_tfidf=is_tfidf)
    logging.info("Writing corpus into %s." % OUTPUT_DIR)
    corpus_name = 'corpus-tfidf.pkl' if is_tfidf else 'corpus.pkl'
    pickle.dump(corpus, open(OUTPUT_DIR + corpus_name, 'wb'))

    ##########################################
    # CREATE LDA MODEL
    ##########################################
    workers = 2
    lda_model = do_LDA(corpus, dictionary, n_worker=workers)
    logging.info("Saving LDA model into %s." % OUTPUT_DIR)
    lda_model.save(OUTPUT_DIR + 'model-lda.gensim')

    # get topic distribution
    top_dist, lda_keys = getDocTopicDist(lda_model, corpus, True)

    ##########################################
    # CLUSTERING
    ##########################################
    # create label for the dataset
    labels, n_clusters = create_labels(name)
    predictions = do_kmeans(top_dist, n_clusters=n_clusters)

    ##########################################
    # EVALUATE THE RESULT
    ##########################################
    prec, recall = evalQualityCluster(labels, predictions, n_clusters=n_clusters)
    logging.info("Clustering result of %s: Prec = %f ; Recall = %f ." % (name, prec, recall))

    ##########################################
    # LDA + Bimeta
    ##########################################


    logging.info("++++++++++++++++++++ END ++++++++++++++++++++++")
