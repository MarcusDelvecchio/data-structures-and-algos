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

# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
# https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/536/
# ex:   Input: root = [1,2,2,3,4,4,3]
#       Output: true
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True
    
    # func to determine if items are same forward and back
    def isLevelSymmetric(level_vals):
        if len(level_vals) == 1: return True
        
        first, second = 0, len(level) - 1
        
        while first < second:
            if level_vals[second] != level_vals[first]:
                return False
            
            second -= 1
            first += 1
        return True
    
    # check symmetry per depth level, so using a queue to manage the items in the level
    level = deque()
    level.append(root)
    
    while len(level):
        # check this level first before traversing to next level
        level_vals = []
        for node in level:
            level_vals.append(node.val if node else None)
        print(level_vals)
        level_ok = isLevelSymmetric(level_vals)
        
        # if the current level isn't symmetric stop
        if not level_ok:
            return False
        
        # continue to the next level
        next_level = deque()
        for node in level:
            if not node: continue
            next_level.append(node.left)
            next_level.append(node.right)
            
        level = next_level
    
    return True

# Path Sum Tree Question
# Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.
# A leaf is a node with no children.
# https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/537/
def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    if not root: return False
    
    if not root.left and not root.right:
        return root.val == targetSum
    else:
        newTargetSum = targetSum - root.val
        return Solution.hasPathSum(self, root.left, newTargetSum) or Solution.hasPathSum(self, root.right, newTargetSum)

# Same Tree LeetCode Easy
# Given the roots of two binary trees p and q, write a function to check if they are the same or not.
# https://leetcode.com/problems/same-tree/
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p or not q:
        if not p and not q:
            return True
        return False
    
    return q.val == p.val and Solution.isSameTree(self, p.left, q.left) and Solution.isSameTree(self, p.right, q.right)

# produce inorder array
# given a binary tree return the in-order traversal in an array
def inorder(root):
    if not root: return []

    return inorder(root.left) + [root.val] + inorder(root.right)

# produce inorder array
# given a binary tree return the in-order traversal in an array
def postorder(root):
    if not root: return []

    return inorder(root.left) + inorder(root.right) + [root.val]

# simple example tree
third_left = TreeNode(15)
third_right = TreeNode(7)
second_right = TreeNode(20, third_left, third_right)
second_left = TreeNode(9)
root = TreeNode(3, second_left, second_right)

# Construct Binary Tree from Inorder and Postorder Traversal
# Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder 
# is the postorder traversal of the same tree, construct and return the binary tree.
# https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/942/
def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    root = postorder[len(postorder) -1]
    right, left = None, None
    
    # if the root has items to the right of it, recursively handle that tree
    inorder_root_index = inorder.index(root)
    if inorder_root_index < len(inorder) - 1:
        postorder_cut = postorder[:len(postorder)-1]
        inorder_right = inorder[inorder_root_index + 1:]
        right = Solution.buildTree(self, inorder_right, postorder_cut)
        
    # check for left
    if inorder_root_index > 0:
        inorder_left = inorder[:inorder_root_index]
        postorder_left = postorder.copy()
        while postorder_left[-1] not in inorder_left:
            postorder_left.pop()
    
        left = Solution.buildTree(self, inorder_left, postorder_left)
        
    return TreeNode(root, left, right)

# using a few properties of inorder and postorder
# 1. the last node of the postorder list is the root
# 2. everything to the left of a node in the inorder list will be to its left, and everything after will be in it's right tree
# much simpler solution below
def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    if not inorder:
        return None
    
    root = postorder.pop()
    
    # if the root has items to the right of it, recursively handle that tree
    inorder_root_index = inorder.index(root)
    right = Solution.buildTree(self, inorder[inorder_root_index+1:], postorder)
        
    # check for left    
    left = Solution.buildTree(self, inorder[:inorder_root_index], postorder)
        
    return TreeNode(root, left, right)
            