import os
import time
import logging
import pickle
import sys, getopt
import ast
import numpy as np
import pandas as pd

from read_fasta import read_fasta_file, create_labels
from lda import create_document, create_corpus, do_LDA, do_LDA_Mallet, getDocTopicDist_mp, save_documents, \
    parallel_create_document, serializeCorpus
from kmeans import do_kmeans, evalQuality
from bimeta import read_bimeta_input, create_characteristic_vector

# LOGGING
logging.basicConfig(level=logging.INFO,
                    filename='LDABimeta.log',
                    filemode='a', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S')


def create_prediction_df(actual, prediction):
    df = pd.DataFrame({'actual': actual, 'prediction': prediction})
    return df


if __name__ == "__main__":
    try:
        logging.info("++++++++++++++++++++ START ++++++++++++++++++++")
        t1 = time.time()
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
        is_tfidf = False
        smartirs = None
        is_seed = True
        run_with_LDAMallet = False

        try:
            opts, args = getopt.getopt(sys.argv[1:], 'ho:d:b:i:k:n:p:j:c:w:s:m:')
        except getopt.GetoptError:
            print("Example for command line.")
            print(
                "main.py -o <output_dir> -d <input dir> -b <bimeta_input> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers> -c <is_tfidf> -w <localW, globalW> -s <seeds or groups>")
            print(
                "main.py -o ../output_dir/ -d ../input_dir/ -b ../bimeta_output/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40 -c 1 -w nfn -s 1")
            sys.exit(2)

        for opt, arg in opts:
            if opt == '-h':
                print("Example for command line.")
                print(
                    "main.py -o <output_dir> -d <input dir> -b <bimeta_input> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers> -c <is_tfidf> -w <localW, globalW>")
                print(
                    "main.py -o ../output_dir/ -d ../input_dir/ -b ../bimeta_output/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40 -c 1 -w nfn")
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
            elif opt == '-c':
                is_tfidf = True if arg == '1' else False
            elif opt == '-w':
                smartirs = arg if arg else None
            elif opt == '-s':
                is_seed = True if arg == '1' else False
            elif opt == '-m':
                run_with_LDAMallet = True if arg == '1' else False
        ##############
        extension_file = '.fna'
        file_name = name + extension_file
        OUTPUT_DIR = OUTPUT_DIR + name + '/'
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
        dictionary, documents = parallel_create_document(reads, k=k, n_workers=n_workers)
        del reads
        logging.info("Delete reads for saving memory ...")
        logging.info("Writing dictionary into %s." % OUTPUT_DIR)
        dictionary.save(OUTPUT_DIR + 'dictionary-k%s.gensim' % (''.join(str(val) for val in k)))

        # if you want to create corpus with TFIDF, set it is True
        corpus = create_corpus(dictionary, documents, is_tfidf=is_tfidf, smartirs=smartirs)
        # corpus_tfidf, dump_path, file_name
        mmcorpus = serializeCorpus(corpus_tfidf=corpus, dump_path=OUTPUT_DIR, file_name=file_name)
        # logging.info("Saving documents as .txt file for using later into %s ." % OUTPUT_DIR)
        # save_documents(documents, OUTPUT_DIR + 'documents.txt')
        logging.info("Deleting documents for saving memory ...")
        del documents
        logging.info("Writing corpus into %s." % OUTPUT_DIR)
        corpus_name = 'corpus-tfidf.pkl' if is_tfidf else 'corpus.pkl'
        pickle.dump(corpus, open(OUTPUT_DIR + corpus_name, 'wb'))

        ##########################################
        # CREATE LDA MODEL IN GENSIM
        ##########################################
        if run_with_LDAMallet:
            # DON'T EDIT THIS PATH
            path_to_lda_mallet = "/home/student/data/Mallet/bin/mallet"
            lda_model = do_LDA_Mallet(path_to_lda_mallet, corpus=mmcorpus, dictionary=dictionary, n_topics=n_topics,
                                      n_worker=n_workers)
        else:
            lda_model = do_LDA(mmcorpus, dictionary, n_worker=n_workers, n_topics=n_topics, n_passes=n_passes)

        logging.info("Saving LDA model into %s." % OUTPUT_DIR)
        lda_model.save(OUTPUT_DIR + 'model-lda.gensim')

        # get topic distribution
        top_dist, lda_keys = getDocTopicDist_mp(model=lda_model, corpus=mmcorpus, n_workers=n_workers,
                                                n_topics=n_topics)
        logging.info("Shape of topic distribution: %s ", str(top_dist.shape))
        logging.info("Deleting reads for saving memory ...")
        del corpus
        logging.info("Saving topic distribution into %s." % OUTPUT_DIR)
        np.savetxt(OUTPUT_DIR + name + "top_dist.csv", top_dist, fmt='%.5e', delimiter=",",
                   header=','.join([str(i) for i in range(n_topics)]))
        t2 = time.time()
        ##########################################
        # CLUSTERING WITH LDA
        ##########################################
        # create label for the dataset
        labels, n_clusters = create_labels(name)
        lda_predictions = do_kmeans(top_dist, seeds=np.array([]), n_clusters=n_clusters, n_workers=n_workers)

        logging.info("Save LDA clustering result to csv file into %s ." % OUTPUT_DIR)
        lda_cluster_df = create_prediction_df(labels, lda_predictions)
        lda_cluster_df.to_csv(OUTPUT_DIR + name + 'lda_prediction_result.csv', index=False)

        ##########################################
        # EVALUATE THE RESULT
        ##########################################
        lda_prec, lda_recall = evalQuality(labels, lda_predictions, n_clusters=n_clusters)
        lda_res = np.array([lda_prec, lda_recall])

        # save the cluster result into .txt file
        np.savetxt(OUTPUT_DIR + name + "prec_recall_LDA.txt", lda_res, fmt="%.4f", delimiter=",")
        logging.info("Clustering result of %s with LDA: Prec = %f ; Recall = %f ." % (name, lda_prec, lda_recall))
        t3 = time.time()
        ##########################################
        # LDA + Bimeta
        ##########################################
        bimeta_extensions = '.fna.seeds.txt' if is_seed else '.fna.groups.txt'
        seeds_dict = read_bimeta_input(BIMETA_INPUT, name + bimeta_extensions)
        seeds = create_characteristic_vector(top_dist, seeds_dict)

        ##########################################
        # CLUSTERING WITH LDA + BIMETA
        ##########################################
        lda_bimeta_predictions = do_kmeans(top_dist, seeds=seeds, n_clusters=n_clusters, n_workers=n_workers)

        logging.info("Save LDA + Bimeta clustering result to csv file into %s ." % OUTPUT_DIR)
        lda_bimeta_cluster_df = create_prediction_df(labels, lda_bimeta_predictions)
        lda_bimeta_cluster_df.to_csv(OUTPUT_DIR + name + 'lda_bimeta_prediction_result.csv', index=False)

        ##########################################
        # EVALUATE THE RESULT
        ##########################################
        lda_bimeta_prec, lda_bimeta_recall = evalQuality(labels, lda_bimeta_predictions, n_clusters=n_clusters)
        lda_bimeta_res = np.array([lda_bimeta_prec, lda_bimeta_recall])
        # save the cluster result into .txt file
        np.savetxt(OUTPUT_DIR + name + "prec_recall_LDA_Bimeta.txt", lda_bimeta_res, fmt="%.4f", delimiter=",")
        logging.info("Clustering result of %s with LDA + Bimeta: Prec = %f ; Recall = %f ." % (
            name, lda_bimeta_prec, lda_bimeta_recall))
        t4 = time.time()
        logging.info("Total time of programe: %f ." % (t2 - t1))
        logging.info("Save running times.")
        with open(OUTPUT_DIR + "running_times.txt", 'w') as f:
            lines = [
                "Running time of LDA: %f" % (t3 - t1),
                "Running time of LDA + Bimeta: %f" % (t2 - t1 + t4 - t3)
            ]
            for line in lines:
                f.writelines(line)
                f.write("\n")
        logging.info("++++++++++++++++++++ END ++++++++++++++++++++++")
    except Exception:
        logging.exception("Exception occurred.", exc_info=True)
        raise
