options(java.parameters = "-Xmx4096m") # set up heap size to 4GB
options(java.parameters = "--XX:-UseGCOverheadLimit") # disabled java gc 
library(mallet)
library(dplyr)
require(Biostrings)
library(clv)

ptm <- proc.time()
allreads <- readDNAStringSet("D:/Data/Dataset/ShortReads/S1.fna.gz", "fasta")
n <- length(allreads)
oi <- seq(from = 1, to = n, by = 2) # danh sách các chỉ số lẻ
ei <- seq(from = 2, to = n, by = 2) # danh sách các chỉ số chẵn
# short reads hay paired-end reads là hai read được thành lập từ việc cắt hai đầu
# của một chuỗi DNA -> paired-end reads là thuộc cùng một loài.
sreads <- paste(allreads[oi], allreads[ei], sep = "") # nối từng cặp paired-end reads (cùng loài) lại
proc.time() - ptm

ptm2 <- proc.time()
n = length(sreads)
l = 4
# Tạo các k-mers doc từ các read (bước này chưa tốt và còn tốn nhiều thời gian)
docs <- lapply(sreads, function(x) 
  paste(DNAStringSet(x, start=1:(width(x)-l+1), width=l), collapse = " "))
proc.time() - ptm2

ptm3 <- proc.time()
instances <- mallet.import(id.array = as.character(1:length(docs)), 
                           text.array = unlist(docs), 
                           stoplist.file = "D:/Metagenomics/en.txt",
                           token.regexp = "\\p{L}[\\p{L}\\p{P}]+\\p{L}")
proc.time() - ptm3

# create a topic trainer object
ptm4 <- proc.time()
topic.model <- MalletLDA(num.topics=10, alpha.sum = 1, beta = 0.1)
proc.time() - ptm4

# Load the documents
ptm5 <- proc.time()
topic.model$loadDocuments(instances)
proc.time() - ptm5

# Get the vocabulary, and some statistics about word frequencies
#vocabulary <- topic.model$getVocabulary()
#head(vocabulary)
#word.freqs <- mallet.word.freqs(topic.model)
#head(word.freqs)

# Optimize hyperparameters ( and ) every 20 iterations, after 50 burn-in iterations
topic.model$setAlphaOptimization(20, 50)

# train a model (specify the number of iterations)
ptm6 <- proc.time()
topic.model$train(200)
proc.time() - ptm6

# run through a few iterations where we pick the best topic for each token,
# rather than sampling from the posterior distribution
ptm7 <- proc.time()
topic.model$maximize(10)
proc.time() - ptm7

# Get the probability of topics in documents and the probability of words in topics
# By default, these functions return raw word counts. 
# Here we want probabilities, so we normalize, 
# and add “smoothing” so that nothing has exactly 0 probability.
ptm8 <- proc.time()
doc.topics <- mallet.doc.topics(topic.model, smoothed=TRUE, normalized=TRUE)
#topic.words <- mallet.topic.words(topic.model, smoothed=TRUE, normalized=TRUE)
proc.time() - ptm8

# Get the top words in topic 2 
# Notice that R indexes from 1 and Java from 0, 
# so this will be the topic that mallet called topic 1.
#mallet.top.words(topic.model, word.weights = topic.words[2,], num.top.words = 10)

# Show the first document with at least 5% tokens belonging to topic 2
#docs[doc.topics[,2] > 0.05][1]

str(doc.topics)

# LDA
ptm9 <- proc.time()
# Gom cụm các doc.topics (reads) dùng k-means và tính precision, recall, f-measure
labels <- c(as.integer(rep(1, 44405)), as.integer(rep(2, 51962))) #S1
clusters <- kmeans(doc.topics, 2)
cm <- confusion.matrix(labels, as.integer(clusters$cluster))
colMaxs <- apply(cm, 2, max)
rowMaxs <- apply(cm, 1, max)
precision <- sum(colMaxs)/sum(cm)
recall <- sum(rowMaxs)/sum(cm)
f_measure <- 2/(1/precision + 1/recall)
precision
recall
f_measure
proc.time() - ptm9


# LDA + Bimeta
ptm10 <- proc.time()
# chạy Bimeta trên tập short reads S1 để tạo S1.fna.seeds.txt và S1.fna.groups.txt
# load S1.fna.seeds.txt và S1.fna.groups.txt và tạo group
# ghi chú: 
#	S1.fna.seeds.txt chứa các non-overlaping read thuộc cùng group (mỗi dòng ứng với mỗi group)
#	S1.fna.groups.txt chứa các read còn lại của mỗi group ((mỗi dòng ứng với mỗi group và có thể rỗng)
seedsFile = "D:/Metagenomics/BiMetaOut/Server/S1.fna.seeds.txt"
groupsFile = "D:/Metagenomics/BiMetaOut/Server/S1.fna.groups.txt"
groupids = lapply(readLines(groupsFile),function(x) as.integer(strsplit(x, ",")[[1]]))
seedids = lapply(readLines(seedsFile),function(x) as.integer(strsplit(x, ",")[[1]]))
groups = mapply(union, seedids, groupids)
proc.time() - ptm10

ptm11 <- proc.time()
# tính các groupvectors, mỗi groupvector là centroid (trung bình) của các vector trong cùng group
groupvectors <- lapply(groups, function(x) 
  if(length(x) > 1) colMeans(doc.topics[x,]) else doc.topics[x,])
proc.time() - ptm11

ptm12 <- proc.time()
# hàm chuyển từ vector -> data frame
vector2df <- function(data) {
  nCol <- max(vapply(data, length, 0))
  #data <- lapply(data, function(row) c(row, rep(NA, nCol-length(row))))
  data <- matrix(unlist(data), nrow=length(data), ncol=nCol, byrow=TRUE)
  data.frame(data)
}

topicsdf = vector2df(groupvectors)
# gom cụm các groupvector
groupclusters <- kmeans(topicsdf, 2)
proc.time() - ptm12

ptm13 <- proc.time()
# gán cụm cho các vector trong cùng group giống với cụm của groupcluster
ngroups = length(groups)
nreads = length(labels)
predictedclusters <- c(as.integer(rep(-1, nreads)))
for(i in 1:ngroups) {
  predictedclusters[groups[[i]]] = groupclusters$cluster[i]
}
predictedclusters <- as.integer(predictedclusters)
proc.time() - ptm13


# Tính precision, recall, và F-measure
proc.time() - ptm14
cm <- confusion.matrix(labels, predictedclusters)
colMaxs <- apply(cm, 2, max)
rowMaxs <- apply(cm, 1, max)
precision <- sum(colMaxs)/sum(cm)
recall <- sum(rowMaxs)/sum(cm)
f_measure <- 2/(1/precision + 1/recall)
precision
recall
f_measure
proc.time() - ptm14

