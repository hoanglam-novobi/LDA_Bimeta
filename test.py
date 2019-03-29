from bimeta import read_bimeta_input

path = 'D:/HOCTAP/HK181/DeCuongLV/datasets/bimeta_output/'
file = 'R4.fna.seeds.txt'

res = read_bimeta_input(path, file)

for key, value in res.items():
    print(key, value)