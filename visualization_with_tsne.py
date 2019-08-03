# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import time
import sys
import getopt
import ast

from MulticoreTSNE import MulticoreTSNE

# LOGGING
logging.basicConfig(level=logging.INFO,
                    filename='LDABimeta.log',
                    filemode='a', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S')


def convert_to_dataframe(X_embedded, n_components=2):
    df = pd.DataFrame()
    if n_components == 2:
        df = pd.DataFrame({'X': X_embedded[:, 0], 'Y': X_embedded[:, 1]})
    elif n_components == 3:
        df = pd.DataFrame({'X': X_embedded[:, 0], 'Y': X_embedded[:, 1], 'Z': X_embedded[:, 2]})
    return df


if __name__ == "__main__":
    try:
        logging.info("++++++++++++++++++++ START ++++++++++++++++++++")
        t1 = time.time()
        n_arguments = len(sys.argv)
        logging.info("Number of arguments: %d ." % n_arguments)
        logging.info("Argument list: %s ." % str(sys.argv))

        # the directory to load k-mer frequency matrix
        INPUT_DIR = ''

        # the directory to save the output
        OUTPUT_DIR = ''

        # the name of file in dataset. Ex: S1, S4, R4
        names = ''

        # number of dimensions to reduce
        n_dims = ''

        # number of workers
        n_workers = ''

        try:
            opts, args = getopt.getopt(sys.argv[1:], 'ho:i:n:d:j:')
        except getopt.GetoptError:
            print("Example for command line.")
            print(
                "visualization_with_tsne.py -o <output_dir> -i <input_dir> -n <file_name> -d <n_dims> -j <n_workers>"
            )
            print(
                "visualization_with_tsne.py -o <output_dir> -i <input_dir> -n <file_name> -d <n_dims> -j <n_workers>"
            )
            sys.exit(2)

        for opt, arg in opts:
            if opt == '-h':
                print("Example for command line.")
                print(
                    "visualization_with_tsne.py -o <output_dir> -i <input_dir> -n <file_name> -d <n_dims> -j <n_workers>"
                )
                print(
                    "visualization_with_tsne.py -o <output_dir> -i <input_dir> -n <file_name> -d <n_dims> -j <n_workers>"
                )
                sys.exit(2)
            elif opt == '-o':
                OUTPUT_DIR = arg
            elif opt == '-i':
                INPUT_DIR = arg
            elif opt == '-n':
                names = ast.literal_eval(arg)
            elif opt == '-d':
                n_dims = ast.literal_eval(arg)
            elif opt == '-j':
                n_workers = ast.literal_eval(arg)

        for name in names:
            logging.info("Processing for %s." % name)
            prefix_name = "kmer_freq_"
            extension_file = '.csv'
            file_name = prefix_name + name + extension_file
            OUTPUT_DIR = OUTPUT_DIR + name + '/'
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            # read data
            logging.info("Reading data ...")
            data = pd.read_csv(INPUT_DIR + file_name)

            df = pd.DataFrame(data)

            # get values
            kmer_matrix = df.values

            # running MulticoreTSNE
            for d in n_dims:
                t1 = time.time()
                logging.info("Running MulticoreTSNE with n_dims = %d..." % d)
                tsne = MulticoreTSNE(n_components=d)
                X_embedded = tsne.fit_transform(kmer_matrix)
                logging.info("X embedded shape: %s." % str(X_embedded.shape))
                t2 = time.time()
                logging.info("Finished MulticoreTSNE with n_dims = %d in %f." % (d, t2 - t1))
                logging.info("Save the result into %s." % OUTPUT_DIR)
                result_df = convert_to_dataframe(X_embedded, n_components=d)
                result_df.to_csv(OUTPUT_DIR + name + '/' + ("tsne_%s_%d_d.csv" % (name, d)), index=False)

    except Exception:
        logging.exception("An exception occurred.", exc_info=True)
        raise
