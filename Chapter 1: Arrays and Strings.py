#1.1 Is Unique
def is_unique(s):
    checker = 0
    for char in s:
        val = ord(char) - ord('a')
        if checker & (1 << val) > 0:
            return False
        checker |= (1 << val)
    return True

#1.2.1 Check Permutation
def is_permutation1(a, b):
    return sorted(a) == sorted(b)

#1.2.2
def count_letters(a):
    result = {}
    for char in a:
        if char not in result.keys():
            result[char] = 0
        result[char] += 1
    return result

def is_permutation2(a, b):
    count_a = count_letters(a)
    count_b = count_letters(b)

    return count_a == count_b

#1.3 URLify
def URLify(s):
    s = " ".join(s.split())
    s = s.strip()
    s = s.replace(' ', '%20')
    return s

#1.4 Palindrome Permutation
def palindrome_permutation(string):
    string = string.replace(' ', '')
    string = string.lower()
    letters_count = count_letters(string)
    odd = sum([val % 2 == 1 for val in letters_count.values()])
    return (len(string) % 2 == 0 and odd == 0) or (len(string) % 2 == 1 and odd == 1)

#1.5 One Away
def are_similar(a, b):
    if len(a) < len(b):
        tmp = a
        a = b
        b = tmp
    if a == b:
        return True
    for i in range(len(a)):
        if a[:i] + a[i+1:] == b:
            return True
    for i in range(len(a)):
        if a[:i] + a[i+1:] == b[:i] + b[i+1:]:
            return True
    return False

def modify_distance(s1, s2):
    index1 = 0
    index2 = 0
    while index1 < len(s1) and index2 < len(s2):
        if s1[index1] != s2[index2]:
            if index1 != index2:
                return False
            index2 += 1
        else:
            index1 += 1
            index2 += 1
    return True

def insert_distance(s1, s2):
    if len(s1) != len(s2):
        return False
    n_differences = sum([s1[idx] != s2[idx] for idx in len(s1)])
    return n_differences < 2

def are_similar1(s1, s2):
    if abs(len(s1) - len(s2)) > 1:
        return False
    if len(s1) > len(s2):
        tmp = s1
        s1 = s2
        s2 = tmp
    if modify_distance(s1, s2):
        return True
    if insert_distance(s1, s2):
        return True
    return False

#1.6 String Compression
def compress(s):
    repeated = 1
    char_count = []
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            repeated += 1
        elif s[i] != s[i + 1]:
            char_count.append((s[i], str(repeated)))
            repeated = 1
    char_count.append((s[-1], str(repeated)))
    compressed = [i for sub in char_count for i in sub]
    compressed = "".join(compressed)

    if len(s) < len(compressed):
        return s
    return compressed

#1.7 Rotate Matrix
def rotate(matrix):
    n = len(matrix[0])

    for rep in range(n // 2):
        tmp = matrix[rep][rep : n - rep]
        for i in range(rep, n - rep):
            matrix[rep][n - 1 - rep - i] = matrix[i + rep][rep]
        for i in range(rep, n - rep):
            print(i, rep, n - 1-rep, i)
            matrix[i][rep] = matrix[n - 1 - rep][i]
        for i in range(rep, n - rep):
            matrix[n - 1 - rep][i] = matrix[n - 1 - i][n - 1 - rep]
        for i in range(len(tmp)):
            matrix[rep + i][n - 1 - rep] = tmp[i]
    return matrix

#1.8 Zero Matrix
def zero_matrix(mat):
    n = len(mat)
    m = len(mat[0])
    zeroed = []
    for i in range(n):
        for j in range(m):
            if mat[i][j] == 0:
                zeroed.append((i, j))

    for i, j in zeroed:
        for k in range(n):
            mat[k][j] = 0
        for k in range(m):
            mat[i][k] = 0

    return mat

#1.9 String Rotation
def string_rotation(s1, s2):
    if len(s1) != len(s2) or not s1 or not s2:
        return False
    s1 += s1
    return s2 in s1

