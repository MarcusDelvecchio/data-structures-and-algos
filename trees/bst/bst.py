
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