import os
import time
import logging
import pickle
import sys, getopt
import ast

from read_fasta import read_fasta_file, create_labels
from lda import create_document, create_corpus, do_LDA, getDocTopicDist
from kmeans import do_kmeans, evalQualityCluster

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
        opts, args = getopt.getopt(sys.argv[1:], 'ho:d:i:k:n:p:l:')
    except getopt.GetoptError:
        print("Example for command line.")
        print("main.py -o <output_dir> -d <input dir> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers>")
        print("main.py -o ../output_dir/ -d ../input_dir/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("Example for command line.")
            print(
                "main.py -o <output_dir> -d <input dir> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers>")
            print("main.py -o ../output_dir/ -d ../input_dir/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40")
            sys.exit()
        elif opt == '-o':
            OUTPUT_DIR = arg
        elif opt == '-d':
            data_path = arg
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
    reads = read_fasta_file(data_path + file_name, pair_end=False)

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
    # CLUSTERING
    ##########################################
    # create label for the dataset
    labels, n_clusters = create_labels(name)
    predictions = do_kmeans(top_dist, n_clusters=n_clusters, n_workers=n_workers)

    ##########################################
    # EVALUATE THE RESULT
    ##########################################
    prec, recall = evalQualityCluster(labels, predictions, n_clusters=n_clusters)
    logging.info("Clustering result of %s: Prec = %f ; Recall = %f ." % (name, prec, recall))

    ##########################################
    # LDA + Bimeta
    ##########################################
    logging.info("++++++++++++++++++++ END ++++++++++++++++++++++")
