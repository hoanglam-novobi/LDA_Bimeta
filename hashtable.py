def convert_char_to_num(s):
    """The function convert each nucleotide in sequence to number"""
    res = -1
    if s == 'A':
        res = 0
    elif s == 'C':
        res = 1
    elif s == 'G':
        res = 2
    elif s == 'T':
        res = 3
    return res


def hashfunction(read, l=None):
    """"""
    if l == None:
        l = len(read)
    hashvalue = 0
    for i in range(len(read)):
        hashvalue += convert_char_to_num(read[i][0:l]) * (4 ** i)

    return hashvalue


class Node:
    def __init__(self, value, idx):
        """
        Initial function
        :param value: hashed value of k-mer
        :param idx: start from 1
        """
        self._read_ids = [idx]
        self._value = value

    def add_read_id(self, id):
        """
        Check if ``id`` exist in self._read_ids.
        If not, append ``id`` into self._read_ids
        :param id:
        :return:
        """
        if id not in self._read_ids:
            self._read_ids.append(id)

    def get_number_read_ids(self):
        """
        Return number of read_ids which have the same k-mer
        :return:
        """
        return len(self._read_ids)

    def is_exist_kmer(self, k_mer_int):
        return self._value == k_mer_int


def create_hash_table(reads, q_mer=20):
    hash_table = {}
    for read_id, read in enumerate(reads):
        # we only use the first 10 characters in the read
        # to create the index of each bin in hash table
        idx = hashfunction(read, l=10)
        if idx not in hash_table:
            hash_table[idx] = []
        kmer_list = [read[i:i + q_mer] for i in range(read)]
        for kmer in kmer_list:
            hashed_kmer = hashfunction(kmer)
            found = False
            i = 0
            size = len(hash_table[idx])
            while not found and i < size:
                if hash_table[idx][i].is_exist_kmer(hashed_kmer):
                    found = True
                i += 1
            if i < size:
                hash_table[idx][i].add_read_id(read_id)
            else:
                hash_table[idx][i].append(Node(hashed_kmer, read_id))
    return hash_table


def find_neighbors(hash_table):
    for key, value in hash_table.items():
        pass
