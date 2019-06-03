#!/bin/bash

nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/seeds/ntn/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R2 -k [4] -n 10 -p 15 -j 40 -c 1 -w ntn -s 1 -m 0

if [[ $? -eq 0 ]]
then
    nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/seeds/ntn/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R3 -k [4] -n 10 -p 15 -j 40 -c 1 -w ntn -s 1 -m 0
fi

if [[ $? -eq 0 ]]
then
    nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/seeds/ntn/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R4 -k [4] -n 10 -p 15 -j 40 -c 1 -w ntn -s 1 -m 0
fi

if [[ $? -eq 0 ]]
then
    nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/seeds/ntn/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R5 -k [4] -n 10 -p 15 -j 40 -c 1 -w ntn -s 1 -m 0
fi

if [[ $? -eq 0 ]]
then
    nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/seeds/ntn/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R6 -k [4] -n 10 -p 15 -j 40 -c 1 -w ntn -s 1 -m 0
fi

if [[ $? -eq 0 ]]
then
    nohup python3.6 main.py -o /home/student/data/lthoang/LDABimeta_output/seeds/ntn/ -d /home/student/data/dataset/ -b /home/student/data/Bimeta/Bimeta/output/ -i R7 -k [4] -n 10 -p 15 -j 40 -c 1 -w ntn -s 1 -m 0
fi

