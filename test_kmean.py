from kmeans import do_kmeans, evalQualityCluster
from lda import load_LDAModel, load_corpus, getDocTopicDist
from read_fasta import create_labels
from bimeta import read_bimeta_input, create_characteristic_vector

BIMETA_INPUT = '/home/student/data/Bimeta/Bimeta/output/'
OUTPUT_DIR = '/home/student/data/lthoang/LDABimeta_output/'

lda_model = load_LDAModel(OUTPUT_DIR + 'R4model-lda.gensim')
corpus = load_corpus(OUTPUT_DIR + 'R4corpus.pkl')

top_dist, lda_keys = getDocTopicDist(lda_model, corpus, True)

name = 'R4'
n_workers = 40
labels, n_clusters = create_labels(name)
lda_predictions = do_kmeans(top_dist, seeds=None, n_clusters=n_clusters, n_workers=n_workers)

bimeta_extensions = '.fna.seeds.txt'
seeds_dict = read_bimeta_input(BIMETA_INPUT, name + bimeta_extensions)
seeds = create_characteristic_vector(top_dist, seeds_dict)

print(seeds)
