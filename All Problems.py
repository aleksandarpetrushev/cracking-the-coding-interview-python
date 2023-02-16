#1.1
def is_unique(s):
    checker = 0
    for char in s:
        val = ord(char) - ord('a')
        if checker & (1 << val) > 0:
            return False
        checker |= (1 << val)
    return True

#1.2.1
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
        
#1.3
def URLify(s):
    s = " ".join(s.split())
    s = s.strip()
    s = s.replace(' ', '%20')
    return s

#1.4
def palindrome_permutation(string):
    string = string.replace(' ', '')
    string = string.lower()
    letters_count = count_letters(string)
    odd = sum([val % 2 == 1 for val in letters_count.values()])
    return (len(string) % 2 == 0 and odd == 0) or (len(string) % 2 == 1 and odd == 1)

#1.5
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
       
#1.6
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

#1.7
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

#1.8
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

#1.9
def string_rotation(s1, s2):
    if len(s1) != len(s2) or not s1 or not s2:
        return False
    s1 += s1
    return s2 in s1

#2
class SLLNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        
    def __repr__(self):
        return str(self.data)
    
    def __eq__(self, other):
        return self.data == other.data


class SLL:
    def __init__(self, lst = []):
        self.head = None
        self.last = None
        for item in lst:
            self.add_last(item)

    def add_last(self, data):
        node = SLLNode(data)
        if self.head is None:
            self.head = node
            self.last = node
            return node
        self.last.next = node
        self.last = node
        return node

    def add_first(self, data):
        node = SLLNodeb(data)
        if self.head is None:
            self.head = node
            self.last = node
            return node
        node.next = self.head
        self.head = node
        return node
        
        
    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(str(node.data))
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)

#2.1
def remove_duplicates(ll):
    prev = ll.head
    cur = ll.head
    while cur is not None:
        it = cur.next
        while it is not None:
            if it == cur:
                if prev == ll.head:
                    ll.head = cur.next
                else:
                    prev.next = cur.next
                break
            it = it.next
            
        prev = cur
        cur = cur.next
    return ll

#2.2
def kth_to_last(ll, k):
    if k <= 0:
        raise Exception('value of k can not be smaller than 0')
    result = []
    cur = 1
    node = ll.head
    while node is not None:
        result.append(node.data)
        node = node.next
    return result[:-k]

def kth(node, k):
    if node is None:
        return 0
    index = kth(node.next, k) + 1
    if index == k:
        print('Result is {0}'.format(node.data))
    return index

#2.3
def delete_middle(ll, node):
    if ll.head == node:
        raise Exception('given node is head')
    
    it = ll.head
    while it.next is not None:
        if it.next == node:
            if it.next.next is None:
                raise Exception('given node is last')
            it.next = it.next.next
            return ll
        it = it.next

    return -1

#2.4
def partition(ll, partition):
    node = ll.head
    prev = ll.head
    while node.next is not None:
        if node.next.data < partition:
            ll.add_first(node.next.data)
            node.next = node.next.next
        prev = node
        node = node.next

    if node.data < partition:
        ll.add_first(node.data)
        prev.next = None
    return ll

import math
#2.5
def sum_list(ll):
    #assume n is even
    node = ll.head
    num = 0
    mult = 1
    while node is not None:
        num += mult * node.data
        mult *= 10
        node = node.next
    mult = math.sqrt(mult)
    total = int(num % mult + num // mult)
    result = [int(d) for d in str(total)]
    result.reverse()
    return SLL(result)

def sum1(ll):
    node = ll.head
    num = 0
    mult = 1
    while node is not None:
        num += mult * node.data
        mult *= 10
        node = node.next
    mult = math.sqrt(mult)
    first = int(num // mult)
    second = int(num % mult)
    result = first + second
    print(first,second,result)
    result = [int(d) for d in str(result)]
    return SLL(result)

#2.6
def reverse(ll):
    if ll.head.next is None:
        return ll
    
    if ll.head.next.next is None:
        ll.last.next = ll.head
        ll.head.next = None
        temp = ll.last
        ll.last = ll.head
        ll.head = temp
        return ll

    current = ll.head.next.next
    previous = ll.head.next
    previous2 = ll.head
    while current is not None:
        previous.next = previous2
        previous2 = previous
        previous = current
        current = current.next
    temp = ll.head
    ll.head = ll.last
    ll.last = temp
    previous.next = previous2
    ll.last.next = None
    return ll

from copy import deepcopy
 
def is_palindrome(ll):
    ll_forward = deepcopy(ll)
    reverse(ll)
    return ll_forward == ll

def is_pal_recursive(node, size, current):
    if size % 2 == 1:
        if current == size // 2:
            return True, node.next
    elif current == size // 2 - 1:
        return node == node.next, node.next.next

    is_equal, right = is_pal_recursive(node.next, size, current + 1)
    result = is_equal and node == right

    return result, right.next
    
#2.7
def intersection(ll1, ll2):
    current1 = ll1.head
    while current1 is not None:
        current2 = ll2.head
        while current2 is not None:
            if current1 is current2:
                return current1
            current2 = current2.next
        current1 = current1.next
    return None

def intersection2(ll1, ll2):
    nodes = set()
    current = ll1.head
    while current is not None:
        nodes.add(current)
        current = current.next
    current = ll2.head
    while current is not None:
        if current in nodes:
            return current
        current = current.next
    return None

def intersection3(ll1, ll2):
    find = find_intersect(ll1, ll2.last)
    if find is not None:
        return find
    find = find_intersect(ll2, ll1.last)
    if find is not None:
        return find
    return None

def find_intersect(ll, node):
    current = ll.head
    while current is not None:
        if current is node:
            return current
        current = current.next
    return None

#3
class MyStack:
    def __init__(self, lst = []):
        self.ll = SLL()
        for item in lst:
            self.ll.add_first(item)

    def pop(self):
        node = self.ll.head
        self.ll.head = self.ll.head.next
        return node.data

    def push(self, item):
        self.ll.add_first(item)
        return item

    def top(self):
        return self.ll.head

    def is_empty(self):
        return self.head is None

    def clear(self):
        self.ll.head = None

    def __repr__(self):
        return self.ll.__repr__()

class MyQueue:
    def __init__(self, lst=[]):
        self.ll = SLL(lst)

    def add(self, item):
        self.ll.add_last(item)

    def dequeue(self):
        node = self.ll.head
        self.ll.head = self.ll.head.next
        return node.data

    def peek(self):
        return self.ll.head

    def is_empty(self):
        return self.ll.head is None

    def clear(self):
        self.ll.head = None

    def __repr__(self):
        return self.ll.__repr__()


#3.2
from collections import deque

class StackEmptyError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

class StackMin:
    def __init__(self):
        self.stack = deque()

    def push(self, item):
        new_min = item
        if self.stack:
            new_min = min(item, self.stack[-1][1])
        self.stack.append((item, new_min))

    def pop(self):
        if not self.stack:
            raise StackEmptyError("Stack is empty")
        return self.stack.pop()[0]

    def get_min(self):
        if not self.stack:
            raise StackEmptyError("stack is empty")
        return self.stack[-1][1]

    def __repr__(self):
        return self.stack.__repr__()

#3.3
class Stacks:
    def __init__(self, size):
        self.size = size
        self.stacks = [deque()]
        self.last = 0

    def pop(self):
        if not self.stacks[0]:
            raise StackEmptyError("stack is empty");
        result = self.stacks[-1].pop()
        if not self.stacks[-1]:
            self.stacks.pop()
            self.last -= 1
        return result

    def push(self, item):
        if len(self.stacks[-1]) == self.size:
            self.stacks.append(deque([item]))
            return item
        self.stacks[-1].append(item)
        return item

    def pop_at(self, position):
        if not self.stacks[position]:
            raise StackEmptyError("stack is empty");
        result = self.stacks[position].pop()
        if not self.stacks[position]:
            del self.stacks[position]
        return result

    def __repr__(self):
        if len(self.stacks) == 1:
            return self.stacks[0].__repr__()
        result = [stack.__repr__() for stack in self.stacks]
        return " ; ".join(result)

#3.4
class MyQueue:
    def __init__(self):
        self.s1 = deque()
        self.s2 = deque()

    def push(self, item):
        self.s1.append(item)
        return item

    def pop(self):
        if not self.s1:
            raise StackEmptyError("stack is empty");
        
        while self.s1:
            self.s2.append(self.s1.pop())
        result = self.s2.pop()
        while self.s2:
            self.s1.append(self.s2.pop())
        return result

    def __repr__(self):
        return self.s1.__repr__()
        

#3.5
def stack_sort(s):
    for i in range(1, len(s)):
        s2 = deque()
        for j in range(i):
            s2.append(s.pop())    
        current = s.pop()
        while current < s2[-1] and s2:
            s.append(s2.pop())
        s.append(current)
        while s2:
            s.append(s2.pop())


#3.6
class Shelter:
    def __init__(self, lst = []):
        self.q = deque(lst)
    def enqueue(self, item):
        self.q.append(item)
        return item
    def dequeueAny(self):
        item = self.q.popleft()
        return item
    def dequeueDog(self):
        item = next( (x for x in self.q if x.startswith('dog')), None)
        if item is not None:
            self.q.remove(item)
        return item
    def dequeueCat(self):
        item = next( (x for x in self.q if x.startswith('cat')), None)
        if item is not None:
            self.q.remove(item)
        return item
    def __repr__(self):
        return str(self.q)

#4.2

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    def __repr__(self):
        return str(self.data)
    

class BST:
    def __init__(self, lst = []):
        lst = sorted(lst)
        if lst:
            sz = len(lst)
            self.root = Node(lst[sz // 2])
            self.root.left = self.build(self.root.left, lst, 0, sz // 2 - 1)
            self.root.right = self.build(self.root.right, lst, sz // 2 + 1, sz - 1)

    def build(self, node, lst, left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        node = Node(lst[mid])
        node.left = self.build(node.left, lst, left, mid - 1)
        node.right = self.build(node.right, lst, mid + 1, right)
        return node

    def print2DUtil(self, root, space):
        if (root == None): 
            return
        space += 10

        self.print2DUtil(root.right, space)  
        print()  
        for i in range(10, space): 
            print(end = " ")  
        print(root.data)  
        self.print2DUtil(root.left, space)  
              
    def print(self): 
        self.print2DUtil(self.root, 0)

    #4.3
    def create_lists(self):
        self.linked_lists = []
        self.create_lists_util(self.root, 0)
        return self.linked_lists

    def create_lists_util(self, root, depth):
        if root is None:
            return
        
        if len(self.linked_lists) <= depth:
            self.linked_lists.append(SLL())
            
        self.linked_lists[depth].add_last(root.data)
            
        self.create_lists_util(root.left, depth + 1)
        self.create_lists_util(root.right, depth + 1)

    def create_lists2(self):
        self.linked_lists = []
        current = deque()
        if self.root is not None:
            current.append(self.root)
        while current:
            self.linked_lists.append(current)
            parents = current
            current = []
            for parent in parents:
                if parent.left is not None:
                    current.append(parent.left)
                if parent.right is not None:
                    current.append(parent.right)
        return self.linked_lists

#4.4
def check_balanced_util(root):
    if root is None:
        return 0, True

    left, res1 = check_balanced_util(root.left)
    right, res2 = check_balanced_util(root.right)
    result = abs(left - right) <= 1 and res1 and res2
    
    return 1 + max(left, right), result
    
def check_balanced(tree):
    height, result = check_balanced_util(tree.root)
    return result


#4.5
def validate_util(root, min, max):
    if root is None:
        return True

    if min is not None and root.data <= min or \
       max is not None and root.data > max:
        return False

    if not validate_util(root.left, min, root.data) or \
        not validate_util(root.right, root.data, max):
        return False

    return True

def validate_bst(tree):
    result = validate_util(tree.root, None, None)
    return result

def validate_bst1(tree):
    inorder = tree.inorder()
    return all([inorder[i] <= inorder[i + 1] for i in range(len(inorder) - 1)])

#4.6
def find_successor_util(root):
    if root.right is None:
        q = node
        node = root.parent
        while node.left != q and node is not None:
            q = node
            node = node.parent
            return node

    node = root.right
    while node.left is not None:
        node = node.left
               
    return node
               
def find_successor(tree, node):
    successor = find_successor_util(node)
    return successor

#5.2
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

#5.3
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

#5.4
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

#5.6
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

#5.7
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

#5.8
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

#6
import math
def cross_off(prime, is_prime):
    i = prime * prime
    while i < len(is_prime):
        is_prime[i] = False
        i += prime
    return is_prime

def next_prime(prime, is_prime):
    next = prime + 1
    while next < len(is_prime) and is_prime[next] == False:
        next += 1
    return next
    
def sieve(maximum):
    is_prime = [True] * (maximum + 1)

    prime = 2
    while prime <= math.sqrt(maximum):
        is_prime = cross_off(prime, is_prime)
        prime = next_prime(prime, is_prime)

    result = [prime for prime in range(maximum + 1) if is_prime[prime]]
    return result

#6.7
import random

def gender_ratio(n_families):
    boys = []
    girls = n_families
    for i in range(n_families):
        current_boys = 0
        while True:
            rand_num = random.random()
            if rand_num < 0.5:
                current_boys += 1
            else:
                break
        boys.append(current_boys)
    boys = sum(boys)
    return boys / (boys + girls)

#4.7
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.vertices.update([u, v])
        
    def topological_sort_util(self, current, visited, stack, rec_stack):
        visited[current] = True
        rec_stack[current] = True
        
        for neighbor in self.graph[current]:
            if visited[neighbor] == False: #cycle exists
                if self.topological_sort_util\
                   (neighbor, visited, stack, rec_stack) == False:
                    return False
            elif rec_stack[neighbor] == True:
                return False
            
        rec_stack[current] = False
        stack.appendleft(current)
        return True
            
    def topological_sort(self):
        visited = dict.fromkeys(self.vertices, False)
        rec_stack = dict.fromkeys(self.vertices, False)
        stack = deque()

        for node in self.vertices:
            if visited[node] == False:
                if self.topological_sort_util(node, visited,\
                                                stack, rec_stack) == False:
                    return -1 #error, cycle exists

        return stack

#4.8 with parent
def lca_util(n1, n2):
    while n1 is not None:
        node = n2
        while node is not None:
            if node == n1:
                return node
            node = node.parent
        n1 = n1.parent
    return None
    
def lca(tree, node1, node2):
    result = lca_util(node1, node2)
    if result is None:
        result = lca_util(node2, node1)
    return result


#4.8 without parent
def lca1_util(root, target, path):
    if root == target:
        return path
    if root is None:
        return None

    left = lca1_util(root.left, target, path + [-1])
    right = lca1_util(root.right, target, path + [1])
    if left is None and right is None:
        return None
    elif left is not None:
        return left
    return right

def lca1(tree, node1, node2):
    path1 = lca1_util(tree.root, node1, [])
    path2 = lca1_util(tree.root, node2, [])
    result = tree.root
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            if path1[i] == -1:
                result = result.left
            else:
                result = result.right
        else:
            break
    return result

def lca2_util(root, target1, target2):
    if root is None:
        return None
    if root == target1 or root == target2:
        return root

    left = lca2_util(root.left, target1, target2)
    right = lca2_util(root.right, target1, target2)
    if left is not None and right is not None:
        print(root)
    if left is not None:
        return left
    if right is not None:
        return right
    return None

def lca2(tree, node1, node2):
    lca2_util(tree.root, node1, node2)

def find(root, node):
    if root is None:
        return False
    if root == node:
        return True
    return find(root.left, node) or find(root.right, node)

def lca3_util(root, p, q):
    if root is None or root == p or root == q:
        return root

    p_left = find(root, p)
    q_left = find(root, q)
    if p_left != q_left:
        return root

    child = root.left if p_left == True else root.right
    return lca3_util(child, p, q)
    

def lca3(tree, node1, node2):
    if find(tree.root, node1) is None or find(tree.root, node2) is None:
        return None
    return lca3_util(root, node1, node2) 

#4.10
def check_subtree_util_helper(root1, root2):
    if root1 is None and root2 is None:
        return True

    if root1 is None or root2 is None:
        return False

    if root1.data != root2.data:
        return False
    
    return check_subtree_util_helper(root1.left, root2.left) and \
           check_subtree_util_helper(root1.right, root2.right)

def check_subtree_util(root1, root2):
    if root1 is None:
        return False

    if check_subtree_util_helper(root1, root2):
        return True
    
    return check_subtree_util(root1.left, root2) or \
               check_subtree_util(root1.right, root2)

def check_subtree(t1, t2):
    return check_subtree_util(t1.root, t2.root)

def preorder(root, result):
    if root is None:
        result.append('X')
        return
    result.append(str(root.data))
    preorder(root.left, result)
    preorder(root.right, result)
        
def check_subtree1(t1, t2):
    preorder1 = []
    preorder2 = []
    preorder(t1.root, preorder1)
    preorder(t2.root, preorder2)
    str1 = "".join(preorder1)
    str2 = "".join(preorder2)
    return str2 in str1

#4.11
import random

class BinaryTree:
    def __init__(self):
        self.root = None
        self.depth = 0

    def find_util(self, root, target):
        if root is None:
            return -1

        if root.data == target:
            return root

        left = self.find_util(root.left, target)
        right = self.find_util(root.right, target)
        if left != -1:
            return left
        if right != -1:
            return right

        return -1

    def find(self, node):
        return self.find_util(self.root, node)

    def get_max_depth(self, root):
        if root is None:
            return 0

        left = self.get_max_depth(root.left)
        right = self.get_max_depth(root.right)

        return 1 + max(left, right)

    def insert(self, value, parent, position):
        new_node = Node(value)
        if parent == None:
            self.root = new_node
            return new_node
        parent = self.find(parent)
        if position == -1:
            parent.left = new_node
        else:
            parent.right = new_node
        self.depth = self.get_max_depth(self.root) - 1

    def get_random_node(self):
        random_depth = random.randint(0, self.depth)
        rand_sequence = random.sample([-1, 1], random_depth)
        rand_node = self.root
        for direction in rand_sequence:
            if direction == -1:
                rand_node = rand_node.left
            else:
                rand_node = rand_node.right
        return rand_node
        
#4.12
def n_paths_util(root, value, result, complete):
    if root is None:
        return
    
    if value - root.data == 0:
        complete = True
        return

    if complete == True:
        result[0] += 1
    
    n_paths_util(root.left, value - root.data, result, complete)
    n_paths_util(root.left, value, result, complete)
    n_paths_util(root.right, value - root.data, result, complete)
    n_paths_util(root.right, value, result, complete)

def n_paths(tree, value):
    result = [0]
    n_paths_util(tree.root, value, result, False)
    return result[0]

def n_paths1_util(root, target, current):
    if root is None:
        return 0

    current += root.data
    total = 0
    if current == target:
        total += 1

    total += n_paths1_util(root.left, target, current)
    total += n_paths1_util(root.right, target, current)
    return total

def n_paths1(root, target):
    if root is None:
        return 0

    n_paths = n_paths1_util(root, target, 0)
    n_paths_left = n_paths1(root.left, target, 0)
    n_paths_right = n_paths1(root.right, target, 0)

    return n_paths + n_paths_left + n_paths_right


#5.1
def insertion(n, m, i, j):
    mask = ~0
    mask <<= j + 1
    mask &= ((1 << i) - 1)
    n &= mask
    m <<= i
    return (n | m)
    
#6.9
def lockers():
    lockers = [False] * 101
    for i in range(1, 101):
        for j in range(i, 101, i):
            lockers[j] = not lockers[j]

    lockers = [val for val in range(1, 101) if lockers[val] == True]
    return lockers

#7.1
from enum import Enum
import random

class Sign(Enum):
    Livce = 1
    Baklava = 2
    Detelina = 3
    Srce = 4

class Card:
    def __init__(self, sign, number):
        self.sign = sign
        self.number = number
        self.available = True

    def __eq__(self, other):
        return self.sign == other.sign and self.number == other.number

    def is_available(self):
        return self.available

    def mark_unavailable(self):
        self.available = False

    def mark_uavailable(self):
        self.available = True

    def __str__(self):
        return "Number: {0}, Sign: {1}".format(self.number, self.sign.name)
    
class DeckOfCards:
    
    _all_cards = [Card(sign, num) for sign in Sign for num in range(1, 15)]

    def __init__(self):
        new_deck = DeckOfCards._all_cards.copy()
        random.shuffle(new_deck)
        self.deck = deque(new_deck)
        self.used = []

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        card = self.deck.pop()
        print(card)
        used.append(card)

class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def score(self):
        score = 0
        for card in self.cards:
            score += card.value
        return score

#7.2
class Employee:
    def __init__(self, embg, name):
        self.embg = embg
        self.name = name
        self.available = True

    def free(self):
        self.available = True

    def make_unavailable(self):
        self.available = False

class Respondent(Employee):
    def __init__(self, embg, name, department):
        super.__init__(embg, name, department)
        self.manager = department.manager
   
class Manager(Employee):
    def __init__(self, embg, name, department):
        super.__init__(embg, name, department)

class Director(Employee):
    def __init__(self, embg, name, departments = []):
        super.__init__(embg, name, departments)

import time

class CallStatus(Enum):
    pending = 1
    active = 2
    finished = 3

import uuid

class Call:
    def __init__(self):
        self.status = CallStatus.pending
        self.employee = None
        self.duration = -1
        self.id == uuid.uuid4()

    def anwer(self, employee):
        self.employee = employee
        self.status = CallStatus.active
        self.start = time.time()

    def end(self):
        self.duration = (time.time() - self.start)
        self.status = CallStatus.finished
        self.employee.free()

    def escalate(self, employee):
        employee.make_unavailable()
        self.employee = employee
        
    def __eq__(self, other):
        return self.id == other.id

class CallCenter:
    def __init__(self):
        self.employees = []
        self.finished_calls = []
        self.active_calls = []
        self.pending_calls = deque()

    def call(self, call):
        self.pending_calls.append(call)

    def answer_call(self):
        call = self.pending_calls.popleft()
        
        for employee in self.employees:
            if issubclass(employee, Respondent):
                if employee.available == True:
                    employee.available = False
                    call.anwer(employee)
                    self.active_calls.append(call)
                    break

        if call.status == CallStatus.pending:
            print("no free employees")

    def end_call(self, call):
        for call_ in self.active_calls:
            if call_ == call:
                self.active_calls.remove(call_)
                call_.end()
                call_.employee.available = True
                self.finished_calls.append(call_)
                return call_
        return None
        
    def escalate_call(self, call):
        if call.status != CallStatus.active:
            raise Exception("Call is not active")

        manager = call.employee.manager
        if manager.available == True:
            call.escalate(manager)
            return True
        elif manager.director.available == True:
            call.escalate(manager.director)
            return True

        return False
