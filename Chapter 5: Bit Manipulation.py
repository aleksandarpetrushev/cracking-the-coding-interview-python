#5.1 Insertion
def insertion(n, m, i, j):
    mask = ~0
    mask <<= j + 1
    mask &= ((1 << i) - 1)
    n &= mask
    m <<= i
    return (n | m)


#5.2 Binary to String
def decimal_to_binary(n):
    digits = 0
    result = []
    while digits <= 32:
        n *= 2
        digits += 1
        if n > 1:
            result.append('1')
            n -= 1
        elif n < 1:
            result.append('0')
        else:
            result.append('1')
            break
    if digits >= 33:
        print("ERROR")
    else:
        print("0.{}".format("".join(result)))

#5.3 Flip Bit to Win
def get_max(n):
    result = 0
    current = 0
    for digit in n:
        if digit == '1':
            current += 1
        else:
            result = max(result, current)
            current = 0
    result = max(result, current)
    return result

def flip_to_win(n):
    n = list(bin(n)[2:])
    result = get_max(n)
    for digit in range(len(n)):
        if n[digit] == '0':
            temp = n.copy()
            temp[digit] = '1'
            new_max = get_max(temp)
            result = max(result, new_max)
    return result

def flip_to_win2(n):
    n = list(bin(n)[2:])
    previous = -1
    current = -1
    result = 0
    for i in range(len(n)):
        if n[i] == '1':
            if previous == -1:
                previous = 1
            elif previous != -1 and current == -1:
                previous += 1
                result = max(result, previous)
            elif previous != -1 and current != -1:
                current += 1
                result = max(result, previous + current)
        else:
            if previous != -1 and current == -1:
                result = max(result, previous)
                current = 0
            elif previous != -1 and current != -1:
                result = max(result, previous + current)
                previous = current
                current = 0
        #print("{0}: previous = {1}, current = {2}, result = {3}"\
        #       .format(i, previous, current, result))
    return result

#5.4 Next Number
def clear_bit(n, i):
    mask = ~(1 << i)
    return n & mask

def set_bit(n, i):
    return n | (1 << i)

def next_number(n):
    first_one = -1
    first_zero_after_one = -1
    n_ones = 0
    n_zeros = 0
    i = 0
    next_smallest = n
    while n > 0:
        if n & 1 == 1:
            n_ones += 1
            if first_one == -1:
                first_one = i
        else:
            n_zeros += 1
            if first_one != -1 and first_zero_after_one == -1:
                first_zero_after_one = i

        #print(n)
        i += 1
        n >>= 1

    next_smallest = clear_bit(next_smallest, first_one)
    next_smallest = set_bit(next_smallest, first_zero_after_one)
    next_largest = (((1 << n_ones) - 1) << n_zeros)

    return next_smallest, next_largest

#5.6 Conversion
def conversion(a, b):
    bits_flipped = 0
    while a > 0 or b > 0:
        if a & 1 != b & 1:
            bits_flipped += 1

        a >>= 1
        b >>= 1

    return bits_flipped

def conversion1(a, b):
    c = a ^ b
    bits_flipped = 0
    while c > 0:
        if c & 1 == 1:
            bits_flipped += 1
        c >>= 1

    return bits_flipped

#5.7 Pairwise Swap
def pairwise_swap(num):
    result = num
    i = 0
    while num > 0:
        first = num & 1
        second = num & 2
        if second == 0:
            result = clear_bit(result, i)
        else:
            result = set_bit(result, i)
        if first == 0:
            result = clear_bit(result, i + 1)
        else:
            resullt = set_bit(result, i + 1)
        num >>= 2
        i += 2

def pairwise_swap1(num):
    even = 0
    odd = 0
    i = 0
    while num > 0:
        even += ((num & 1) << i)
        num >>= 1
        i += 1
        odd += ((num & 1) << i)
        num >>= 1
        i += 1

    result = 0
    i = 0
    while odd > 0 or even > 0:
        result += ((odd & 1) << i)
        odd >>= 1
        i += 1
        result += ((even & 1) << i)
        even >>= 1
    return result

def num_bits(num):
    comparator = 1
    result = 0
    while num < comparator:
        result += 1
        comparator <<= 1
    return result

def pairwise_swap2(num):
    result = num ^ ((1 << num_bits(num)) - 1)
    i = 0
    while num > 0:
        if num & 3 == 3:
            result = set_bit(result, i)
            result = set_bit(result, i + 1)

        num >>= 2
        i += 2
    return result

#5.8 Draw Line
def draw_line(screen, width, x1, x2, y):
    start_byte = y * width + (x1 // 8)
    end_byte = y * width + (x2 // 8)
    j = x1
    x1 = y * width * 8 + x1
    x2 = y * width * 8 + x2
    bit = start_byte * 8
    for i in range(start_byte * 8, (end_byte + 1) * 8):
        if x2 >= bit >= x1:
            screen[i // 8][j % 8] = 1
            j += 1
        bit += 1

    row = []
    for i in range(len(screen)):
        if i % width == 0 and i > 0:
            print(row)
            row = []

        row.append(screen[i])
    print(row)
