
# todo
# search recursive
# search linear
# find min/max
# inerstion
# deletion

# Convert Sorted Array to Binary Search Tree LeetCode Easy
def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        
        def build_tree(arr):
            if not arr: return None
            root_idx = len(arr)//2
            left = build_tree(arr[:root_idx])
            right = build_tree(arr[root_idx+1:])
            root = TreeNode(arr[root_idx], left, right)
            return root

        return build_tree(nums)