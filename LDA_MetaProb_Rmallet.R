options(java.parameters = "-Xmx4096m") # set up heap size to 4GB
options(java.parameters = "--XX:-UseGCOverheadLimit") # disabled java gc 
options(encoding="utf-8")
# load the package rJava
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_171')
require(mallet)
require(dplyr)
require(Biostrings)
require(clv)
require(rgl)

ptm <- proc.time()
# labels <- c(as.integer(rep(1, 16447)), as.integer(rep(2, 18010))) #R4
# labels <- c(as.integer(rep(1, 44405)), as.integer(rep(2, 51962))) #S1
# labels <- c(as.integer(rep(1, 38664)), as.integer(rep(2, 38629))) #R2
labels <- c(as.integer(rep(1, 22027)), as.integer(rep(2, 18016))) #R5
allreads <- readDNAStringSet("D:/HOCTAP/HK181/DeCuongLV/datasets/bimeta_dataset/R5.fna", "fasta")
n <- length(allreads)
# oi <- seq(from = 1, to = n, by = 2) # danh sÃ¡ch cÃ¡c chá» sá» láº»
# ei <- seq(from = 2, to = n, by = 2) # danh sÃ¡ch cÃ¡c chá» sá» cháºµn
# short reads hay paired-end reads lÃ  hai read ÄÆ°á»£c thÃ nh láº???p tá»« viá»c cáº¯t hai Äáº§u
# cá»§a má»t chuá»i DNA -> paired-end reads lÃ  thuá»c cÃ¹ng má»t loÃ i.
# sreads <- paste(allreads[oi], allreads[ei], sep = "") # ná»i tá»«ng cáº·p paired-end reads (cÃ¹ng loÃ i) láº¡i

sreads <- paste(allreads[seq(from = 1, to = n, by = 1)])
proc.time() - ptm

ptm2 <- proc.time()
n = length(sreads)
l = 4
l1 = 4
l2 = 3
l3 = 5
# Táº¡o cÃ¡c k-mers doc tá»« cÃ¡c read (bÆ°á»c nÃ y chÆ°a tá»t vÃ  cÃ²n tá»n nhiá»u thá»i gian)
# docs <- lapply(sreads,
#                function(x) paste(DNAStringSet(x, start=1:(width(x)-l+1), width=l), collapse = " "))
# part2 <- lapply(sreads, 
#                function(x) paste(DNAStringSet(x, start=1:(width(x)-l2+1), width=l2), collapse = " "))
# part3 <- lapply(sreads, 
#                function(x) paste(DNAStringSet(x, start=1:(width(x)-l3+1), width=l3), collapse = " "))
docs <- lapply(sreads,
                function(x)
                  paste(
                    c(paste(DNAStringSet(x, start=1:(width(x)-l1+1), width=l1), collapse = " "),
                      paste(DNAStringSet(x, start=1:(width(x)-l2+1), width=l2), collapse = " "),
                      paste(DNAStringSet(x, start=1:(width(x)-l3+1), width=l3), collapse = " ")),  collapse = " "))

proc.time() - ptm2

ptm3 <- proc.time()
instances <- mallet.import(id.array = as.character(1:length(docs)), 
                           text.array = unlist(docs), 
                           stoplist.file = "D:/HOCTAP/HK181/DeCuongLV/en.txt",
                           token.regexp = "\\p{L}[\\p{L}\\p{P}]+\\p{L}")
proc.time() - ptm3

# create a topic trainer object
ptm4 <- proc.time()
topic.model <- MalletLDA(num.topics=10)
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
# topic.model$setAlphaOptimization(20, 50)

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
# and add âsmoothingâ so that nothing has exactly 0 probability.
ptm8 <- proc.time()
doc.topics <- mallet.doc.topics(topic.model, smoothed=TRUE, normalized=TRUE)
#topic.words <- mallet.topic.words(topic.model, smoothed=TRUE, normalized=TRUE)

# convert doc.topics to df
prob_df <- as.data.frame(doc.topics)
# save to csv
write.csv(prob_df, "prob_R5_fna_3_4_5_LDA_10.csv", row.names=FALSE)
# plot3d(doc.topics, col=labels, main="actual read clusters (k = 3)")
proc.time() - ptm8

# Get the top words in topic 2 
# Notice that R indexes from 1 and Java from 0, 
# so this will be the topic that mallet called topic 1.
#mallet.top.words(topic.model, word.weights = topic.words[2,], num.top.words = 10)

# Show the first document with at least 5% tokens belonging to topic 2
#docs[doc.topics[,2] > 0.05][1]

str(doc.topics)





# LDA + Bimeta
ptm10 <- proc.time()

# doc ket qua co duoc tu viec chay MetaProb
groupsFile = "/home/hoang/Desktop/MetaProb/output/R4.fna.groups.csv"
groups <- read.csv(groupsFile, header = FALSE)

doc.topics$group = groups[,2] + 1
avgtopicgroups = doc.topics[,-c(1,2)] %>% group_by(group) %>% summarise_all(funs(mean))   # code này cần thư viện dplyr
groupclusters <- kmeans(avgtopicgroups[,2:4], k = 2, start = "random", iter.max = 100, nstart = 1)

doc.topics$cluster = groupclusters$cluster[doc.topics$group]
plot3d(avgtopicgroups[,2:4], col=groupclusters$cluster, main="MetaProb group clusters")
plot3d(doc.topics[,3:5], col=as.integer(doc.topics$cluster), main="LDAMetaProb clusters (k = 3)")

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
ptm14 <- proc.time()
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

