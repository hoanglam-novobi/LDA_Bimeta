# LDA_Bimeta
Python source code for running LDA with Bimeta

Example for runscript

./main.py -o <output_dir> -d <input dir> -b <bimeta_input> -i <input file> -k <k-mers> -n <num topics> -p <num passes> -j <n_workers>

./main.py -o ../output_dir/ -d ../input_dir/ -i R4 -k [3, 4, 5] -n 10 -p 15 -j 40

In the server

python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R4 -k [4] -n 10 -p 15 -j 40
