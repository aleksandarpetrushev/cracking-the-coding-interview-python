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


#3.2 Stack Min
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

#3.3 Stack of Plates
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

#3.4 Queue via Stacks
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


#3.5 Sort Stack
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


#3.6 Animal Shelter
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