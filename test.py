import numpy as np

from bimeta import read_bimeta_input, create_characteristic_vector

path = 'D:/HOCTAP/HK181/DeCuongLV/datasets/bimeta_output/'
file = 'R4.fna.seeds.txt'

res = read_bimeta_input(path, file)

for key, value in res.items():
    print(key, value)

a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
seeds = {
    1: [1, 2],
    2: [4, 5],
    3: [0]
}

result = create_characteristic_vector(a, seeds)
print(result)