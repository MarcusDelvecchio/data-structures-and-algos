
# todo
# search recursive
# search linear
# find min/max
# inerstion
# deletion

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
    print(sorted_arr)
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