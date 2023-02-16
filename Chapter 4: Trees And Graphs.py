
#4.2 Minimal Tree

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

#4.3 List of Depths
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

#4.4 Check Balanced
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


#4.5 Validate BST
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

#4.6 Successor
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

#4.7 Build Order
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

#4.8 First common ancestor with parent
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


#4.8 First common ancestor without parent
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

#4.10 Check Subtree
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

#4.11 Random Node
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

#4.12 Paths with Sum
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

