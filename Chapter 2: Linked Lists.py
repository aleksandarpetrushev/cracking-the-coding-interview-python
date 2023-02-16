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

#2.1 Remove Duplicates
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

#2.2 Return Kth to Last
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

#2.3 Delete Middle Node
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

#2.4 Partition
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

#2.5 Sum Lists
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

#2.6 Palindrome
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

#2.7 Intersection
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
