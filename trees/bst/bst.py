
# todo
# search recursive
# search linear
# find min/max
# inerstion
# deletion

# importtant ideas and concepts:
#   traversing a BST in-order (smallest to greatest values)
#   converting a sorted list to a balanced BST

# Convert Sorted Array to Binary Search Tree LeetCode Easy
# first LC BST problem. Took like 7 mins total
def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        
    def build_tree(arr):
        if not arr: return None
        root_idx = len(arr)//2
        left = build_tree(arr[:root_idx])
        right = build_tree(arr[root_idx+1:])
        root = TreeNode(arr[root_idx], left, right)
        return root

    return build_tree(nums)

# Minimum Absolute Difference in BST LeetCode Easy
# Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.
# https://leetcode.com/problems/minimum-absolute-difference-in-bst/description/
# took like 20 mins because has the wrong approach but did in like 5 mins once I realized
# TC O(n), SC O(n)
# takes two passes and requires that you store the entire BST in a separate array
def getMinimumDifference(self, root: Optional[TreeNode]) -> int:     
    def bst_sort(root):
        if not root: return []
        return bst_sort(root.left) + [root.val] + bst_sort(root.right)
    sorted_arr = bst_sort(root)

    dif, prev = 100000, None
    for val in sorted_arr:
        if prev != None and abs(val - prev) < dif:
            dif = abs(val - prev)
        prev = val
    return dif

# same as above but single-pass TC O(n) SC O(1)
minimum, prev = 100000, None
def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
    def find_min(root):
        if not root: return min
        find_min(root.left)
        if self.prev != None:
            self.minimum = min(self.minimum, root.val - self.prev)
        self.prev = root.val
        find_min(root.right)
    find_min(root)
    return self.minimum

# Find Mode in Binary Search Tree
# https://leetcode.com/problems/find-mode-in-binary-search-tree/description/
# took under 10 mins
# TC = O(n), SC = O(n)
mode, res = 0, []
def findMode(self, root: Optional[TreeNode]) -> List[int]:
    freq, = Counter()

    def dfs(root):
        if not root: return None
        freq[root.val] += 1
        if freq[root.val] == self.mode:
            self.res.append(root.val)
        elif freq[root.val] > self.mode:
            self.res = [root.val]
            self.mode = freq[root.val]
        dfs(root.left)
        dfs(root.right)
    dfs(root)
    return self.res

# Binary Search Tree to Greater Sum Tree LeetCode Medium
# Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.
# took about 25 mins, actually had to think pretty hard but makes sense and like the clean solution
# idea: 
# 1. pass everything that is greater down to right and left (if there is a greater ancestor tree for example) for child nodes to add to their values
# 2. but do the right side first and have the right side return the largest subtree value (i.e., the leftmost value. Because in a subtree, the farthest left node will be the sum of the entire tree)
# 3. with this value returned from the right, increase the 'greater' value provided by the parent, and pass it to the left to update all of their values
# TC: O(n) SC: O(1)
def bstToGst(self, root: TreeNode) -> TreeNode:
    def dfs(root, greater):
        if not root: return greater
        greater = dfs(root.right, greater)
        root.val += greater
        return dfs(root.left, root.val) if root.left else root.val
    dfs(root, 0)
    return root

# todo come back to this because I looked at the answer
# Construct Binary Search Tree from Preorder Traversal
# https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/description/
# and come back to this idk how to do
# https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/

# Two Sum IV - Input is a BST LeetCode Easy
# https://leetcode.com/problems/two-sum-iv-input-is-a-bst/description/
# TC O(n) SC O(n)
# took like 3 mins
def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
    find = set()
    def dfs(root):
        if not root: return False
        if root.val in find: return True
        find.add(k - root.val)
        return dfs(root.left) or dfs(root.right)
    return dfs(root)

# Search in a Binary Search Tree LeetCode Easy
# https://leetcode.com/problems/search-in-a-binary-search-tree/submissions/
# took like 1 minute mbut just trying to do easy questions because BST solutions aren't coming to me very easily
# TC O(n) SC O(1)
def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    def bsearch(root):
        if not root: return None
        if root.val == val: return root
        return bsearch(root.left) if val < root.val else bsearch(root.right)
    return bsearch(root)

# Increasing Order Search Tree LeetCode Easy
# Given the root of a binary search tree, rearrange the tree in in-order so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only one right child.
# this actually took me a minute. Idk why I'm having a hard time grasping the concept of the bst in-order traversal
prev, new_head = None, None
def increasingBST(self, root: TreeNode) -> TreeNode:
    
    def dfs(root):
        if not root: return
        dfs(root.left)
        if not self.new_head: self.new_head = root
        if self.prev:
            self.prev.right = root
        self.prev = root
        dfs(root.right)
        root.left = None  
    dfs(root)
    return self.new_head

# Range Sum of BST LeetCode Easy
# https://leetcode.com/problems/range-sum-of-bst/description/
# took like 5 mins
sum = 0
def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
    
    def dfs(root):
        if not root: return

        if root.val >= low and root.val <= high:
            self.sum += root.val
        dfs(root.left)
        dfs(root.right)
    dfs(root)
    return self.sum

# notes on this question^
# again, I still struggle with conceptualizing BSTs. I attempted to use the following logic for excample, which is incorrect and wrong in thinking.
# I need to try harder to realize the patterns with BSTs and how some rules are not intuitive
# see the below code for the above dfs algo and how it is incorrect
def dfs(root):
    if not root: return

    if root.val > low and root.val < high:
        self.sum += root.val
    if root.val < low:
        dfs(root.right)
    if root.val > high:
        dfs(root.left)

# other similar thinking is that the closest relationships between node values will be between parent and children, which is not the case and I am slowly having to try harder to realize such patterns

# Construct Binary Search Tree from Preorder Traversal LeetCode Medium
# https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/description/0
# took about 10 mins but came back to it and did it myself after not being able to do it and looking at a solution
def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
    if not preorder: return None
    root = TreeNode(preorder[0])
    preorder, i = preorder[1:], 0
    while i < len(preorder) and preorder[i] < root.val:
        i += 1
    root.left = Solution.bstFromPreorder(self, preorder[:i])
    root.right = Solution.bstFromPreorder(self, preorder[i:])
    return root

# Balance a Binary Search Tree LeetCode Medium
# https://leetcode.com/problems/balance-a-binary-search-tree/submissions/
# Given the root of a binary search tree, return a balanced binary search tree with the same node values. If there is more than one answer, return any of them.
# idea: convert the BST to a list and then build the tree from that list. See 'sortedListToBST' solution below where you build a BST from a sorted list
# TC O(n) ( I think - O(n) to traverse, O(n) to build) SC O(n)
def balanceBST(self, root: TreeNode) -> TreeNode:
    arr = []
    
    def traverse(root):
        if not root: return None
        traverse(root.left)
        arr.append(root.val)
        traverse(root.right)

    def build(arr):
        if not arr: return None
        root_idx = len(arr)//2
        root = TreeNode(arr[root_idx])
        root.left = build(arr[:root_idx])
        root.right = build(arr[root_idx+1:])
        return root
    
    traverse(root)
    return build(arr)  

# there is definitely a more efficient way of doing this
# also why does this always work isn't there a way to skew the list so that alwys uging the midpoints to build the subtrees doesn't always work? nope

# All Elements in Two Binary Search Trees LeetCode Medium
# Given two binary search trees root1 and root2, return a list containing all the integers from both trees sorted in ascending order.
# https://leetcode.com/problems/all-elements-in-two-binary-search-trees/submissions/
# approach: converts trees to lists and merge the lists (could not think of a way to do it with single pass and recursion but will look at other solutions rn)
# TC = O(n) to convert the trees to lists and O(n) to merge the two lists, SC = O(n) - need to store entire duplicates of trees in lists
# (see improved version below)
def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
    def tree_to_list(root):
        if not root: return []
        return tree_to_list(root.left) + [root.val] + tree_to_list(root.right)
    arr1 = tree_to_list(root1)
    arr2 = tree_to_list(root2)
    res, p1, p2 = [], 0, 0
    while p1 < len(arr1) or p2 < len(arr2):
        if p1 == len(arr1):
            res.append(arr2[p2])
            p2 += 1
        elif p2 == len(arr2):
            res.append(arr1[p1])
            p1 += 1 
        elif arr1[p1] < arr2[p2]:
            res.append(arr1[p1])
            p1 += 1
        else:
            res.append(arr2[p2])
            p2 += 1        
    return res

# yea it seems this is the best way to approach the problem (sadly not 'clean single pass solution', but did clean up my above solution with the below using deques)
# improved from above
# and also note the time complexity is actually O(n+m) not O(n) (unless it might be assumed that n = the length of the two lists combined I guess)
def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
    def tree_to_list(root):
        if not root: return []
        return tree_to_list(root.left) + [root.val] + tree_to_list(root.right)
    list1, list2, res = deque(tree_to_list(root1)), deque(tree_to_list(root2)), []
    while list1 and list2:
        res.append(list1.popleft() if list1[0] < list2[0] else list2.popleft())
    return res + list(list1 or list2)

# Convert Sorted List to Binary Search Tree LeetCode Medium
# https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
# idea: two pointer method, split list into left, root and right lists and recursively perform this logic
# couldn't find a way to make it much cleaner
# took about 10 mins
# TC O(n) SC O(n) because we have to build an entirely new tree
def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
    def build(l):
        if not l: return None
        p1, p2, prev = l,l, None
        while p2.next and p2.next.next:
            prev = p1
            p1 = p1.next
            p2 = p2.next.next
        
        # cut off end and recall with new two sublists
        if prev:
            prev.next = None
        r = p1.next
        p1.next = None
        return TreeNode(p1.val, build(l) if prev else None, build(r))
    return build(head)

# Binary Search Tree Iterator LeetCode Medium
# https://leetcode.com/problems/binary-search-tree-iterator/description/
# approach: convert BST to queue and pop throught the queue as the user want to traverse the tree/move the pointer forward
# pointer only needs to be able to go forwards not back
# took like 5 mins
from collections import deque
class BSTIterator:
    arr = deque()

    def build(self, root):
        if not root: return None
        self.build(root.left)
        self.arr.append(root.val)
        self.build(root.right)  

    def __init__(self, root: Optional[TreeNode]):
        self.build(root)

    def next(self) -> int:
        return self.arr.popleft() if self.arr else None

    def hasNext(self) -> bool:
        return bool(self.arr)

# Kth Smallest Element in a BST LeetCode Medium
# https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
# Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
# idea: do the usual BST in-ordser traversal but keep a global k value to keep track of hwo many nodes we have visited and once at the kth element return it all the way back to the top (without exploring further)
# TC: O(n), SC: O(1)
k = 0
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    def explore(root):
        if not root: return None
        left = explore(root.left)
        if left != None: return left # if left returns a value stop early
        self.k += 1
        if self.k == k: return root.val
        return explore(root.right) # if right returns a value return right else none
    return explore(root)

# Lowest Common Ancestor of a Binary Search Tree LeetCode Medium
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/submissions/
# approach: search through the tree in-order (as in the solution above) and return from each node exploration to values 1. if a solution has been found in the node and 2. if a common ancestor has been found (if two nodes have been found) and what that common ancestor is
# TC O(n) SC O(1)
# took 13 mins
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def find(root):
        if not root: return False, None
        found_left, val_left = find(root.left)
        if found_left and val_left: return True, val_left 
        found_right, val_right = find(root.right)
        if found_right and val_right: return True, val_right
        if root.val == q.val or root.val == p.val:
            return True, root if found_left or found_right else None
        if found_left and found_right: # note this and the below return statement can be condensed but not going to do that for readability
            return True, root
        return False, None
    return find(root)[1]

# wow but look at this iterative solution so much simpler https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/solutions/1394823/explained-easy-iterative-python-solution/
# I didn't even take into account the fact that there is order within the tree and we don't need to traverse in order
# at any node, the node is either 1. greater than both values we are looking for, or 2. it's LESS than both values we are looking for. Otherwise IT is the common node
# if 1. go left if 2. go right else return root wow simple

# Delete Node in a BST LeetCode Medium
# https://leetcode.com/problems/delete-node-in-a-bst/description/
# took like 35 mins becuase of edge cases. This soluition is ugly asf
# O(logn) or O(h) and O(logn) / O(h) space
def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    def update(root):
        if not root: return None
        if root.val == key:
            if not root.left or not root.right: return root.right or root.left
            next, prev = root.right, None
            while next.left:
                prev = next
                next = next.left
            # section a start
            rightmost = next.right
            while rightmost and rightmost.right:
                rightmost = rightmost.right
            if prev:
                if rightmost:
                    rightmost.right = root.right
                else:
                    next.right = root.right
                prev.left = None
            # seciton a end
            next.left = root.left
            return next
        if key < root.val:
            root.left = update(root.left)
        else:
            root.right = update(root.right)
        return root
    return update(root)


# okay cleaned the solution up a bit by using recursion to remove the node we reaplce the deleted node with. I only changed "section a" from the above code
def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    def update(root):
        if not root: return None
        if root.val == key:
            if not root.left or not root.right: return root.right or root.left
            next, prev = root.right, None
            while next.left:
                prev = next
                next = next.left
            next.right = self.deleteNode(root.right, next.val)
            next.left = root.left
            return next
        if key < root.val:
            root.left = update(root.left)
        else:
            root.right = update(root.right)
        return root
    return update(root)

# Validate Binary Search Tree LeetCode Medium
# https://leetcode.com/problems/validate-binary-search-tree/description/
# took 13 mins but had to refactor because I originally thought it was just as simple as comparing root to children
# TC O(logn) SC O(n)
# (better solution below)
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    greater, lesser = set(), set()
    def validate(root):
        if not root: return True
        for val in greater:
            if root.val <= val: return False
        for val in lesser:
            if root.val >= val: return False
        lesser.add(root.val)
        left = validate(root.left) 
        if not left: return False
        lesser.remove(root.val)
        greater.add(root.val)
        right = validate(root.right)
        greater.remove(root.val)
        return right
    return validate(root)

# WAY BETTER
# alternative solution that uses less space, is O(1) instead of O(n)
# have two variables upper and lower and at every node, based on the path to the node, the node value must be wiothin the upper and lower bound
# and each node will accordingly update the upper and lower bound before exploring it's left and right child and then undo said changes (backtracking?)
# TC O(n) SC O(1)
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def validate(root, lower, upper):
        if not root: return True
        if lower != None and root.val <= lower: return False
        if upper != None and root.val >= upper: return False
        left = validate(root.left, lower, root.val) 
        if not left: return False
        right = validate(root.right, root.val, upper)
        return right
    return validate(root, None, None)


# Hard
# given BST returns number of ways the BST could hbave been re-arranged if the BST was generated from a list
# incorrect
def numOfWays(self, nums: List[int]) -> int:
    def combos(root):
        if not root: return 1
        if not root.left and not root.right: return 1
        if not root.left: return combos(root.right)
        if not root.right: return combos(root.left)
        return 2*combos(root.left) + 2*combos(root.right)
    return combos(root)

# given a list to be used to build a BST, determine the number of nodes in each layer without building the tree

# Minimum Depth of Binary Tree LeetCode Easy
# TC O(n) SC O(n)
def minDepth(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    def bfs():
        layer, d = deque([root]), 1
        while layer:
            next = deque()
            for node in layer:
                if node.left: next.append(node.left)
                if node.right: next.append(node.right)
                if not node.left and not node.right:
                    return d
            layer = next
            d += 1
        return d
    return bfs()

# Balanced Binary Tree LeetCode Easy
# Given a binary tree, determine if it is height-balance
# took like 5 mins after I realized how the height-balanced aspect works
# approach: dfs each path and ensure that any node has both balanced subtrees (for the sake of early stoppage) and that the subtree depths are within 1
# TC O(n), SC O(1)
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    def dfs(root, d):
        if not root: return True, d
        left_balanced, left_depth = dfs(root.left, d+1)
        if not left_balanced: return False, 0
        right_balanced, right_depth = dfs(root.right, d+1)
        return right_balanced and abs(left_depth - right_depth) < 2, max(left_depth, right_depth)
    return dfs(root, 0)[0]

# Recover Binary Search Tree LeetCode Medium
# https://leetcode.com/problems/recover-binary-search-tree/description/
# You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.
# needed to look at solution and realized I completely missed the idea of using the inorder traversal
# see solution
# this didn't make sense to me the way he did it with the start and end but it makes sense. Just think about it as if it were an array/list
def recoverTree(self, root: Optional[TreeNode]) -> None:
    prev, start, end = None, None, None
    def inorder(root):
        nonlocal prev, start, end
        if not root: return False
        found = inorder(root.left)
        if found: return True
        if prev and prev.val > root.val:
            if not start:
                start = prev
            end = root
        prev = root         
        inorder(root.right)
    inorder(root)
    if start and end: start.val, end.val = end.val, start.val
    return root

# Convert BST to Greater Tree LeetCode Medium
# https://leetcode.com/problems/convert-bst-to-greater-tree/submissions/
# Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.
# did this qesiton already but did different approach where instead of going per root->leaf path I just did a reverse inorder traversal
# see the other solution for this somewhere above
# took 7 mins
def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    total = 0
    def rev_inorder(root):
        nonlocal total
        if not root: return 0
        rev_inorder(root.right)
        root.val += total
        total = root.val
        rev_inorder(root.left)
    rev_inorder(root)
    return root

# Insert into a Binary Search Tree LeetCode Medium
# https://leetcode.com/problems/insert-into-a-binary-search-tree/description/
# took like 5 mins. Did iteratively so space complexity is O(1) because no recursive call
# TC O(h)/O(n) SC O(1)
def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if not root: return TreeNode(val, None, None)
    curr, prev = root, root
    while curr:
        prev = curr
        curr = curr.left if val < curr.val else curr.right
    if val < prev.val:
        prev.left = TreeNode(val, None, None)
    else:
        prev.right = TreeNode(val, None, None)
    return root

# Maximum Sum BST in Binary Tree LeetCode Hard
# https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/description/
# Given a binary tree root, return the maximum sum of all keys of any sub-tree which is also a Binary Search Tree (BST).
# a lot harder than I expected
# took 90 mins
def maxSumBST(self, root: Optional[TreeNode]) -> int:
    maximum, prev, defects, prev_depth = 0, None, set(), 0

    # perform single pass and find all defect nodes
    def inorder(root, depth):
        nonlocal maximum, prev, defects, prev_depth
        if not root: return 0
        inorder(root.left, depth + 1)
        # if a defect is found in the order, set the higher node to defective
        if (prev and prev.val >= root.val):
            if depth < prev_depth:
                defects.add((root.val, id(root)))
            else:
                defects.add((prev.val, id(prev)))               
        prev = root
        prev_depth = depth
        inorder(root.right, depth + 1)   

    # then do dfs to get the maximum sum while also validating that no defects appear in the subtrees
    def dfs(root):
        nonlocal maximum, defects
        if not root: return True, 0
        left_valid, left_sum = dfs(root.left)
        right_valid, right_sum = dfs(root.right)
        if left_valid and right_valid and (root.val, id(root)) not in defects:
            maximum = max(maximum, root.val + left_sum + right_sum)
        return left_valid and right_valid and (root.val, id(root)) not in defects, root.val + left_sum + right_sum
    inorder(root, 0)
    dfs(root)
    return maximum
    
# binary tree gotchas:
# nodes to left and right shouldn't be equal to the the root

# todo
# given a binary tree print the number of valid subtrees in the tree

# Pseudo-Palindromic Paths in a Binary Tree LeetCode Medium
# https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/?envType=daily-question&envId=2024-01-24
# took 10 mins
# Return the number of pseudo-palindromic paths going from the root node to leaf nodes => a path that's elements can be re-arranged into a palindrom
# approach: keep track of the count of all path nodes, there can only be one odd count for it to be palindrome. Do BFS.
# TC: O(n) SC: O(n)
def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
    res = 0

    def is_sudo_palindromic(path_node_counts):
        odd_used = False
        for val in path_node_counts.keys():
            if path_node_counts[val]%2 == 1:
                if odd_used: return False
                else: odd_used = True
        return True
    
    def dfs(root, node_counts):
        if not root: return
        nonlocal res
        node_counts[root.val] += 1

        if not root.left and not root.right:
            res += 1 if is_sudo_palindromic(node_counts) else 0
        else:
            dfs(root.left, node_counts)
            dfs(root.right, node_counts)
        node_counts[root.val] -= 1
    dfs(root, Counter())
    return res