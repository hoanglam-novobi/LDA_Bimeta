# import library for visualazation 3D
require(clv)
require(rgl)

# load the prob file created by t-SNE
# data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_R4_fna_LDA_10.csv"
# data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_S1_fna_LDA_10_1_0.1.csv"
# data_dir <- "D:/HOCTAP/HK181/DeCuongLV/plot/t_SNE_10_3_S1_fna_LDA_10_1_0.1.csv"
# data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_S2_fna_LDA_10.csv"
data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_S1_fna_3_4_5_LDA_10.csv"
# data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_R4_fna_3_4_5_LDA_10.csv"
# data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_R5_fna_4_LDA_10.csv"
# data_dir <- "C:/Users/HOANG/Documents/t_SNE_10_3_R2_fna_4_LDA_10.csv"
data <- read.csv(data_dir, header = FALSE)

# get lables for R4.fna

# labels <- c(as.integer(rep(1, 22027)), as.integer(rep(2, 18016))) #R5
# labels <- c(as.integer(rep(1, 16447)), as.integer(rep(2, 18010))) #R4
labels <- c(as.integer(rep(1, 38664)), as.integer(rep(2, 38629))) #R2
# labels <- c(as.integer(rep(1, 114177)), as.integer(rep(2, 81162))) #S2
# labels <- c(as.integer(rep(1, 44405)), as.integer(rep(2, 51962))) #S1

# visualization in 3D
plot3d(data, col=labels, main="t-SNE with LDA, #topic = 10")