import os
import time
import logging
import pickle
import sys, getopt
import ast
import numpy as np

from read_fasta import read_fasta_file, create_labels
from lda import create_document, create_corpus, do_LDA, getDocTopicDist
from kmeans import do_kmeans, evalQualityCluster
from bimeta import read_bimeta_input, create_characteristic_vector
# LOGGING
logging.basicConfig(level=logging.INFO,
                    filename='LDABimeta.log',
                    filemode='a', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    logging.info("++++++++++++++++++++ START ++++++++++++++++++++")
    n_arguments = len(sys.argv)
    logging.info("Number of arguments: %d ." % n_arguments)
    logging.info("Argument list: %s ." % str(sys.argv))

    # the directory to load the result from phase 1 Bimeta
    BIMETA_INPUT = ''
    # the directory to save the output
    OUTPUT_DIR = ''
    # the name of file in dataset. Ex: S1, S4, R4
    name = ''
    # the list of integer
    k = ''
    n_topics = ''
    n_passes = ''
    # the direction for read dataset
    data_path = ''
    n_workers = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ho:d:b:i:k:n:p:j:')
    except getopt.GetoptError:
        print("Example for command line.")
        print(
            "main.py -o <output_dir> -d <input dir> -b <bimeta_input> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers>")
        print("main.py -o ../output_dir/ -d ../input_dir/ -b ../bimeta_output/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("Example for command line.")
            print(
                "main.py -o <output_dir> -d <input dir> -b <bimeta_input> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers>")
            print(
                "main.py -o ../output_dir/ -d ../input_dir/ -b ../bimeta_output/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40")
            sys.exit()
        elif opt == '-o':
            OUTPUT_DIR = arg
        elif opt == '-d':
            data_path = arg
        elif opt == '-b':
            BIMETA_INPUT = arg
        elif opt == '-i':
            name = arg
        elif opt == '-k':
            k = ast.literal_eval(arg)
        elif opt == '-n':
            n_topics = int(arg)
        elif opt == '-p':
            n_passes = int(arg)
        elif opt == '-j':
            n_workers = int(arg)

    ##############
    extension_file = '.fna'
    file_name = name + extension_file
    OUTPUT_DIR = OUTPUT_DIR + name
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ##########################################
    # READING FASTA FILE
    ##########################################
    is_pairend = True if name[0] == 'S' else False
    reads = read_fasta_file(data_path + file_name, pair_end=is_pairend)

    ##########################################
    # CREATE DOCUMENTS, CORPUS
    ##########################################
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
    lda_model = do_LDA(corpus, dictionary, n_worker=n_workers, n_topics=n_topics, n_passes=n_passes)
    logging.info("Saving LDA model into %s." % OUTPUT_DIR)
    lda_model.save(OUTPUT_DIR + 'model-lda.gensim')

    # get topic distribution
    top_dist, lda_keys = getDocTopicDist(lda_model, corpus, True)

    ##########################################
    # CLUSTERING WITH LDA
    ##########################################
    # create label for the dataset
    labels, n_clusters = create_labels(name)
    lda_predictions = do_kmeans(top_dist, seeds=np.array([]), n_clusters=n_clusters, n_workers=n_workers)

    ##########################################
    # EVALUATE THE RESULT
    ##########################################
    lda_prec, lda_recall = evalQualityCluster(labels, lda_predictions, n_clusters=n_clusters)
    logging.info("Clustering result of %s with LDA: Prec = %f ; Recall = %f ." % (name, lda_prec, lda_recall))

    ##########################################
    # LDA + Bimeta
    ##########################################
    bimeta_extensions = '.fna.seeds.txt'
    seeds_dict = read_bimeta_input(BIMETA_INPUT, name + bimeta_extensions)
    seeds = create_characteristic_vector(top_dist, seeds_dict)

    ##########################################
    # CLUSTERING WITH LDA + BIMETA
    ##########################################
    lda_bimeta_predictions = do_kmeans(top_dist, seeds=seeds, n_clusters=n_clusters, n_workers=n_workers)

    ##########################################
    # EVALUATE THE RESULT
    ##########################################
    lda_bimeta_prec, lda_bimeta_recall = evalQualityCluster(labels, lda_bimeta_predictions, n_clusters=n_clusters)
    logging.info("Clustering result of %s with LDA + Bimeta: Prec = %f ; Recall = %f ." % (
    name, lda_bimeta_prec, lda_bimeta_recall))

    logging.info("++++++++++++++++++++ END ++++++++++++++++++++++")
