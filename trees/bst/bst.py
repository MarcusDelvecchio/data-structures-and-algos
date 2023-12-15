
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
        return min
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
# took about 25 mins, actually had to think pretty hard but makes sense and like the clean solution
# idea: 
# 1. pass everything that is greater down to right and left (if there is a greater ancestor tree for example) for child nodes to add to their values
# 2. but do the right side first and have the right side return the largest subtree value (i.e., the leftmost value. Because in a subtree, the farthest left node will be the sum of the entire tree)
# 3. with this value returned from the right, increase the 'greater' value provided by the parent, and pass it to the left to update all of their values
def bstToGst(self, root: TreeNode) -> TreeNode:
    def dfs(root, greater):
        if not root: return greater
        greater = dfs(root.right, greater)
        root.val += greater
        return dfs(root.left, root.val) if root.left else root.val
    dfs(root, 0)
    return root