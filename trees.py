from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Given the root of a binary tree, return its maximum depth.
# as per https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/535/
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    
    left_ans = maxDepth(self, root.left)
    right_ans = maxDepth(self, root.right)
    return max(left_ans, right_ans) + 1