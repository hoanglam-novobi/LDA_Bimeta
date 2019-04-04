# LDA_Bimeta
Python source code for running LDA with Bimeta

Example for runscript

./main.py -o <output_dir> -d <input dir> -b <bimeta_input> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers> -c <is_tfidf> -w <localW, globalW, document normalized> -s <seeds or groups> -m <LDAmallet>"

./main.py -o ../output_dir/ -d ../input_dir/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40 -c 1 -smartirs nfn

In the server

python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R4 -k [4] -n 10 -p 15 -j 40 -c 1 -w nfn -s 1 -m 1

To run in background:
nohup <your command> &
Example:
nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R4 -k [4] -n 10 -p 15 -j 40 -c 1 -w nfn -s 1 -m 1&

To kill process in background:
pkill -f <a part of the name of the process>

Example:
pkill -f "python3.6 main.py -o"

Some type of Local weight and Global weight in TF-IDF model
Term frequency weighing:
n - natural,
l - logarithm,
a - augmented,
b - boolean,
L - log average.
Document frequency weighting:
n - none,
t - idf,
p - prob idf.
Document normalization:
n - none,
c - cosine.

Install LDAMallet
- su yum install java-1.8.0-openjdk
- su yum install ant
- git clone https://github.com/mimno/Mallet.git
- cd Mallet/
- ant
If you got an error, so edit the line in build.xml to:
<javac srcdir="${src.dir}" destdir="${build.classes.dir}" encoding="iso-8859-1" />

Path for run LDAMallet in server: /home/student/data/Mallet/bin/mallet