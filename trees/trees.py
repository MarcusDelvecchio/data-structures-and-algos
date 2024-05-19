from typing import Optional, List

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

# same thing as above but using DFS
# Tree: Height of a Binary Tree HackerRank Easy
# https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem
def height(root):
    nodes = deque([root])
    depth = 0
    
    while nodes:
        size = len(nodes)
        for _ in range(size):
            node = nodes.popleft()
            
            # add nodes children
            if node.right:
                nodes.append(node.right)
            if node.left:
                nodes.append(node.left)
        depth += 1
    return depth-1

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

# Closest Binary Search Tree Value LeetCode Easy
# https://leetcode.com/problems/closest-binary-search-tree-value/description/
# Interesting question for understanding intuition of BSTs. Go traverse left/right as we find values closer to our target, but when we cannot go
# left or right anymore, you don't simply return the current node; nodes we have traversed already could be closer, so we return the closest we have seen
# example: tree [5, 1, 15] with target 7, we traver right to 15, but don't return 15, we return 6
# TC: O(n), SC: O(1)
def closestValue(self, root: Optional[TreeNode], target: float) -> int:
    closest = float('inf')
    while root:
        # update the closest value
        if abs(root.val - target) < abs(closest - target):
            closest = root.val
        elif abs(root.val - target) == abs(closest - target):
            closest = min(root.val, closest)
        
        # go left or right to a closer value if we can, else return the closest value we've seen
        if target > root.val and root.right:
            root = root.right
        elif target < root.val and root.left:
            root = root.left
        else:
            return closest

# Path Sum 2 LeetCode Medium
# https://leetcode.com/problems/path-sum-ii/submissions/
# took almost 40 but still picking up the tree ideas
# this one was just weird bc needing to pass data back up the tree
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    if not root:
        return []
    
    # check if a leaf node and if value is correct value to finish path 
    if not root.left and not root.right:
        if root.val == targetSum:
            return [[root.val]]
        else:
            return []
        
    left_solutions = Solution.pathSum(self, root.left, targetSum - root.val)
    right_solutions = Solution.pathSum(self, root.right, targetSum - root.val)
    
    res = []
    # if there are solutions from the left branch add our value to them and return upwards
    for solution in left_solutions:
        sol = [root.val] + solution
        res.append(sol)
        
    # if there are solutions from the right branch add our value to them and return upwards
    for solution in right_solutions:
        sol = [root.val] + solution
        res.append(sol) 
    return res

# Binary Tree Sum Root to Leaf Numbers
# https://leetcode.com/problems/sum-root-to-leaf-numbers/description/
# took 11 mins
res = 0 # defining outside because if we do definition inside nested func we will get 'variable res referneced before assignment' - also can workarounf by doing res[0] but this is fine
def sumNumbers(self, root: Optional[TreeNode]) -> int:
    def get_sum(root, num):
        if not root: return num
        if root.left: get_sum(root.left, num + str(root.val))
        if root.right: get_sum(root.right, num + str(root.val))
        if not root.left and not root.right:
            self.res += int(num + str(root.val))
    
    get_sum(root, "")
    return self.res

# Delete Leaves With a Given Value LeetCode Medium
# https://leetcode.com/problems/delete-leaves-with-a-given-value/description/?envType=daily-question&envId=2024-05-17
# : Given a binary tree root and an integer target, delete all the leaf nodes with value target.
# : Note that once you delete a leaf node with value target, if its parent node becomes a leaf node and has the value target, it should also be deleted (you need to continue doing that until you cannot).
# approach: do dfs and have a node return true if it is in the "deleted" state, i.e., deletd or non-existient
# if a child's two children (if any) are deleted, then that node itself is also elligible for deletion if it matches the target
# but regardless of whether or not it is to be deleted, if either child is returns True to indicate is is to be deleted, the relationship to the child is removed
# TC: O(n), SC: O(1), beats 98% in runtime
def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
    
    def dfs(root):
        if not root: return True

        left_deleted = dfs(root.left)
        right_deleted = dfs(root.right)
        if left_deleted: root.left = None
        if right_deleted: root.right = None
        elligible_for_deletion = left_deleted and right_deleted

        if elligible_for_deletion and root.val == target:
            return True
        return False
    
    delete_root = dfs(root)
    return None if delete_root else root

# Populating Next Right Pointers in Each Node
# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/
# took 28 but I call it 25 becuase of stupid issues
# combination of both iterative and recursive approach I guess?
def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root: return
    
    def con(n):
        if not n: return
        next_l = n.left
        next_r = n.right
        while next_l:
            next_l.next = next_r
            next_l = next_l.right
            next_r = next_r.left
        con(n.left)
        con(n.right)

    con(root)
    return root

# alternative solution to connect solution
# using DFS like mine, but instead of going all the way to thte bottom is just one node at a time depth first
# utilizes the fact that if node.next we can set node.left.next = node.next.left which I didn't even realize
# here https://leetcode.com/problems/populating-next-right-pointers-in-each-node/solutions/379177/python3-bfs-and-dfs/
def connect_2(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root: return
    
    ## (1). left child -> right child
    ## (2). right child -> next.left child
    def dfs(self,root):
        if root == None or root.left == None:
            return
        root.left.next = root.right
        if root.next != None: 
            root.right.next = root.next.left
        self.dfs(root.left)
        self.dfs(root.right)

    dfs(root)
    return root

# Lowest Common Ancestor of a Binary Tree Medium
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/
# took 50 mins but did for 30 and got stuck on issues with testcases (seemingly) merging but they weren't
def lowestCommonAncestor(self, root, p, q):
    if p is q: return p
    res, P = [], []
    
    def dfs(root):
        if not root: return
        P.append(root)
        if root is p or root is q:
            res.append(list(P))
            if len(res) == 2:
                return True
        if dfs(root.left) or dfs(root.right):
            return True
        P.pop()

    dfs(root)
    for i in range(len(res[0])):
        if i == len(res[0]) or i == len(res[1]):
            return recent
        if res[0][i] is res[1][i]:
            recent = res[0][i]
    return recent

# Lowest Common Ancestor of a Binary Tree Medium alternative solution
# wow so short so simple
# explanation here https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/solutions/3231708/236-solution-with-step-by-step-explanation/
def lowestCommonAncestor(self, root, p, q):
    if not root or root == p or root == q:
      return root

    l = lowestCommonAncestor(root.left, p, q)
    r = lowestCommonAncestor(root.right, p, q)

    if l and r:
      return root
    return l or r

# Sum Root to Leaf Numbers LeetCode Medium
# https://leetcode.com/problems/sum-root-to-leaf-numbers/description/
# TC: O(n), SC: O(1)
# took 6 mins
def sumNumbers(self, root: Optional[TreeNode]) -> int:
    ans = 0

    def dfs(root, path):
        if not root: return
        nonlocal ans
        path.append(str(root.val))

        # traverse left
        dfs(root.left, path)

        # traverse right
        dfs(root.right, path)

        # if it's a leaf node, add the path number to the sum
        if not root.right and not root.left:
            path_val = int("".join(path))
            ans += path_val
        path.pop()
    dfs(root, [])
    return ans

# Add One Row to Tree LeetCode Medium
# took 33 mins and 12 mins to shorten
# https://leetcode.com/problems/add-one-row-to-tree/description/
# Given the root of a binary tree and two integers val and depth, add a row of nodes with value val at the given depth depth.
# (see full explanation, a bit complicated)
def addOneRow(self, root, val, depth):
    q, d = deque([root]), 1

    if depth == 1:
        return TreeNode(val, root)

    while d < depth:
        next_row = deque()
        if d + 1 == depth:
            for n in q:
                n.left = TreeNode(val, n.left)
                n.right = TreeNode(val, None, n.right)
        else:
            for node in q:
                if node.left: next_row.append(node.left)
                if node.right: next_row.append(node.right)
            q = next_row
        d += 1
    return root

# sombody else's BFS solution for Add One Row. Adding here because I it makes a lot of sense
# also follows that pattern of filling queue level-by-level with nested loop
# https://leetcode.com/problems/add-one-row-to-tree/solutions/2664284/python-two-solutions-using-dfs-and-bfs/
def addOneRow(self, root, val, depth):
    if depth == 1: return TreeNode(val, root)
    
    queue = deque([root])
    while depth - 1 != 1:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        depth -= 1
            
    while queue:
        node = queue.popleft()
        node.left  = TreeNode(val, left  = node.left)
        node.right = TreeNode(val, right = node.right)
        
    return root

# recursive solution (depth first rather then breadth first) for above
# took literally 5 mins to re-write nice
def addOneRow(self, root, val, depth):
    if depth == 1:
        return TreeNode(val, root)

    def insert(root, dep):
        if not root: return
        if dep + 1 == depth:
            root.left = TreeNode(val, root.left)
            root.right = TreeNode(val, None, root.right)
        else:
            insert(root.left, dep+1)
            insert(root.right, dep+1)
    insert(root, 1)
    return root

# Find Duplicate Subtrees LeetCode Medium
# https://leetcode.com/problems/find-duplicate-subtrees/description/
# did in 23 mins noice
# weirdest issue where swtiching the order of nodes returned makes the answer incorrect - see the comment
# very nice solution one of the top Python solutions is the EXACT same https://leetcode.com/problems/find-duplicate-subtrees/solutions/1178526/easy-clean-straightforward-python-recursive/
def findDuplicateSubtrees(self, root):
    trees, res = Counter(), []

    def dfs(root):
        if not root: return [None]
        tree = [root.val] + dfs(root.left) + dfs(root.right)
        # tree = dfs(root.left) + [root.val] + dfs(root.right) is incorrect - why?
        if tuple(tree) and trees[tuple(tree)] == 1:
            res.append(root)
        trees[tuple(tree)] += 1
        return tree
    dfs(root)
    return res

# Construct Binary Tree from Preorder and Inorder Traversal
# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
# took 35 mins because of small issues but could have done in like 25-30 if ideal
# still have a hard time wrapping my head around this problem and dealing with the small edge cases with the recusvie calls and poping items from the queue
def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if len(preorder) == 0:
        return None
    
    preorder_glo = deque(preorder)
    def construct(preorder, inorder):
        if len(inorder) == 0: return None
        if len(inorder) == 1:
            return TreeNode(preorder_glo.popleft())

        root_index = inorder.index(preorder[0])
        current = preorder_glo.popleft()

        left = construct(preorder_glo, inorder[:root_index])
        right = construct(preorder_glo, inorder[root_index + 1:])
        return TreeNode(current, left, right)

    return construct(preorder_glo, inorder)


# yet again, another vastly simpler solution done in 6 lines from some other guy
# study this. Its so simple
# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solutions/401124/python-easy-solution-with-comments/
# note that also here list.index(n) is O(n) for every input in n so it inefficient, so a hashmap could have simply been used to improve effeciency but good otherwise
def buildTree(self, preorder, inorder):
        # Recursive solution
        if inorder:
            # Find index of root node within in-order traversal
            index = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[index])
            
            # Recursively generate left subtree starting from 
            # 0th index to root index within in-order traversal
            root.left = self.buildTree(preorder, inorder[:index])
            
            # Recursively generate right subtree starting from 
            # next of root index till last index
            root.right = self.buildTree(preorder, inorder[index+1:])
            return root

# Binary Tree Right Side View LeetCode Medium
# https://leetcode.com/problems/binary-tree-right-side-view/description/
# this took 8:41 seconds new record for medium noice  
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root: return []
    nodes = defaultdict(TreeNode)

    def dfs(root, d):
        if not root: return
        nodes[d] = root
        dfs(root.left, d+1)
        dfs(root.right, d+1)
    dfs(root, 0)
    return [nodes[d].val for d in nodes.keys()]

# Path Sum III LeetCode Medium
# https://leetcode.com/problems/path-sum-iii/submissions/
# took about 40 mins but could have had it in like 20 but was having issues with 1 off error
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
    if not root: return 0

    res = [0]
    def find_paths(root, targets):
        if not root: return
        
        res[0] += targets.count(root.val)
        for i in range(len(targets)):
            targets[i] -= root.val
        targets.append(targetSum)
        
        find_paths(root.left, targets.copy())
        find_paths(root.right, targets.copy())
    find_paths(root, [targetSum])
    return res[0]

# my cleaner solution with comments
res = 0
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
    def find_paths(root, targets):
        if not root: return

        # 1. increment res if root is equal to any of the values in targets ("completes" any of the paths in targets)
        self.res += targets.count(root.val)
        
        # 2. subtract root.val from every target in targets and add target to targets again ("update" the paths in targets)
        targets = [targets[i] - root.val for i in range(len(targets))] + [targetSum]

        # 4. call function on left and right
        find_paths(root.left, targets)
        find_paths(root.right, targets)
        
    find_paths(root, [targetSum])
    return self.res

# note that at 4. targets.copy() is no longer needed to be used because the targets array is duplicated/copied in the above line anyways

# Most Frequent Subtree Sum LeetCode Medium
# https://leetcode.com/problems/most-frequent-subtree-sum/submissions/
# took 12 mins
from collections import Counter
class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        sums = Counter()

        def dfs(root):
            if not root: return 0
            val = root.val + dfs(root.right) + dfs(root.left)
            sums[val] += 1
            return val
        
        dfs(root)
        
        # return largest vals in res
        freq, res = 0, []
        for sum in sums.keys():
            if sums[sum] == freq:
                res.append(sum)
            if sums[sum] > freq:
                res = [sum]
                freq = sums[sum]
        
        return res

# Binary Tree Maximum Path Sum Hard
# https://leetcode.com/problems/binary-tree-maximum-path-sum/description/
# did in 13 minutes say word
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    m = [root.val]

    def find_paths(root):
        if not root: return 0

        # 1. find max sub in right and left subtrees 
        right = max(find_paths(root.right), 0)
        left = max(find_paths(root.left), 0)

        # 2. update max if entire subtree (left, root, right) is greater or right subtree is greater or left subtree is greater
        if right + left + root.val > m[0]:
            m[0] = right + left + root.val

        # 3. return the max from the side of the tree that has the larger sum (or the node itself if both children are negative)
        return max(left, right) + root.val


    find_paths(root)
    return m[0]

# Vertical Order Traversal of a Binary Tree Hard
# https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/description/
# took 40 flippin minutes because of the issue with nodes being able to have the same damn coordinates
# but definitely could have figured out the sorting better and faster, realized it was top-to-bottom for column values 2 mins before at the end as well so dfs needed dict logic to support that
# will definitely do a BFS implementation soon for practice and bc I think it can be much cleaner
def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root: return []
    cols = defaultdict(list)

    def traverse(root, row, col):
        if not root: return

        # add value to column
        col_vals = [item for item in cols[col] if item[1] == row]
        new_col = []
        for node in cols[col]:
            if node[1] < row:
                new_col.append(node)
            else:
                break
        new_col.extend(sorted(col_vals + [(root.val, row)]))
        for node in cols[col]:
            if node[1] > row:
                new_col.append(node)
        cols[col] = new_col

        # traverse children
        traverse(root.left, row + 1, col - 1)
        traverse(root.right, row + 1, col + 1)
    
    traverse(root, 0, 0)
    vals = []
    for col in sorted(cols.keys()):
        col_vals = []
        for node in cols[col]:
            col_vals.append(node[0])
        vals.append(col_vals)
    return vals

# Binary Tree Cameras LeetCode Hard
# https://leetcode.com/problems/binary-tree-cameras/description/
# took 43 mins but was pretty ugly
cams = 0
def minCameraCover(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    if not root.left and not root.right:
        return 1

    def calc_cams(root, has_parent):
        if not root: return False, False
        if root.left or root.right:
            left_needs, left_is = calc_cams(root.left, True)
            right_needs, right_is = calc_cams(root.right, True)
            if left_needs or right_needs:
                self.cams += 1
                return False, True
            elif left_is or right_is:
                return False, False
        if not has_parent:
            self.cams += 1
        return True, False
    calc_cams(root, False)
    return self.cams

# Height of Binary Tree After Subtree Removal Queries LeetCode Hard
# https://leetcode.com/problems/height-of-binary-tree-after-subtree-removal-queries/
# this shit took me 6:15:00 ...
# I tried every method but kept gettings issues with edge case scnaraios where the tree was as wide or as tall as possible
# the solution is to use bfs and store the 'nodes under' for all nodes along each depth
# then when you remove the specified node you can simply get sibling node to that node with the largest number of nodes under it
# about 2 or 3 hours into this problem I tried the bfs solution exactly like this, but ran into memory problems when I tried to store
# the number of nodes under EVERY node in every layer. So I moved on to try different solutions
# a day later I cam  back after hints and realized that I could have reduced the memory issue by only storing the top TWO nodes
# with the most number of nodes under them in each layer and this worked.
# need to try to think of these edge case scaarios more where the tree is as large or is as wide as possible
def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
    # aka "nodes under this node". Map of { node.val: number of nodes under said node } hash table
    node_depths = Counter()

    # map containing the two nodes in each depth with the most nodes under it
    nodes_at_depths = defaultdict(list)

    # map for node to it's depth (distance from top)
    depth_of_nodes = defaultdict(int)

    def get_depths(root, d):
        if not root: return 0
        depth_of_nodes[root.val] = d                  

        max_depth_under = 0
        if root.right or root.left:
            max_depth_under = max(get_depths(root.left, d+1) + 1, get_depths(root.right, d+1) + 1)
        node_depths[root.val] = max_depth_under
    
        # keep the two largest 'nodes under' values at each depth
        if len(nodes_at_depths[d]) < 2:
            nodes_at_depths[d].append(root.val)
        else:
            if node_depths[root.val] > node_depths[nodes_at_depths[d][0]] or node_depths[root.val] > node_depths[nodes_at_depths[d][1]]:
                index_to_replace = 0 if node_depths[nodes_at_depths[d][1]] > node_depths[nodes_at_depths[d][0]] else 1
                nodes_at_depths[d][index_to_replace] = root.val
        return max_depth_under

    res = []
    get_depths(root, 0)
    for query in queries:
        # get the depth of the node
        d = depth_of_nodes[query]

        # get the other nodes at the same depth
        layer = nodes_at_depths[d]

        # return the sibling value
        if len(layer) == 1:
            d_new = d
            d_new -= 1
            added = False
            while d_new > 0 and not added:
                layer = nodes_at_depths[d_new]
                largest = 0
                for node in layer:
                    if node_depths[node] != node_depths[query] + d - d_new:
                        if not largest:
                            largest = node_depths[node]
                        elif node_depths[node] > largest:
                            largest = node_depths[node]
                        added = True
                        res.append(max(largest + d_new, d - 1))
                d_new -= 1
            else:
                res.append(d - 1)
        else:
            index = None
            if query in layer:
                index = 0 if layer[1] == query else 1
            else:
                index = 0 if node_depths[layer[0]] > node_depths[layer[1]] else 1
            res.append(node_depths[layer[index]] + d)
    return res

# Recover a Tree From Preorder Traversal LeetCode Hard
# https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/description/
# took 58 mins
# wasn't bad. I tried fiddling with the string logic and wasn't sure whether to do it iteratively or recursively for a while
# but figured it out after a bit. Also just converted the string bs to a queue of (node_val, node_depth) tuples
# which made the problem a lot easier to grasp
def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
    if not traversal: return None

    # convert the string traversal into a queue
    q, depth, num = deque(), 0, ""
    for i in range(len(traversal)):
        if traversal[i] == '-':
            depth += 1
        else:
            num += traversal[i]

            if i+1 == len(traversal) or traversal[i+1] == '-':
                q.append((int(num), depth))
                num, depth = "", 0

    def build_tree(nodes):
        if not nodes: return None

        root = nodes.popleft()

        # collect nodes in left and right trees
        nodes_left = deque()
        nodes_right = deque()
        i = 0

        # loop through tuples until you find a second node with depth d+1 from root. When you find this you know you have hit the right sub tree (nodes on the right)
        for node in nodes:
            if node[1] == root[1] + 1:
                i += 1
            
            if i < 2:
                nodes_left.append(node)
            else:
                nodes_right.append(node)
        
        # create node and recursively build left and right trees
        return TreeNode(int(root[0]), build_tree(nodes_left), build_tree(nodes_right))

    return build_tree(q)

# similar and simpler solution here
# https://leetcode.com/problems/recover-the-original-array/discuss/1647452/Python-Short-solution-explained
# rather than checking if a potential k value exists from every element to some other element he just checks every possible k value from the lowest value to every other val
# and instead of duplicating the sorted(nums) array and removing items as we move up (as we select those items as a higher[i] to some lower[i]), he uses a Counter
# because we only care about the number of items remaining not the order so why use an array and array.remove this is inefficient
# and as soon as a single solution works the guy returns it (because there are multiplereco)

# Cycle Length Queries in a Tree LeetCode Hard
# https://leetcode.com/problems/cycle-length-queries-in-a-tree/description/
# took 13 mins but was reading for like 8 and complete in 5
# Given the properties of the tree in question, node with value n has parent value floor(n/2). This can be used to traverse upwards in the tree from both nodes 
# (while keeping track of distance traveled) until a common ancestor is found. This distance will represent the length of the cycle less the single edge connecting the two nodes.
def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
    # given the properties of the tree, any node n has parent value floor(n/2)
    res = []
    for q in queries:
        a = q[0]
        b = q[1]

        # find common parent between a and b counting distance
        d = 0
        while a != b:
            if a > b:
                a = floor(a/2)
            else:
                b = floor(b/2)
            d += 1
        res.append(d + 1)
    return res

# Binary Tree Zigzag Level Order Traversal LeetCode Medium
# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/submissions/
# took 13 because of issues with configuring start/stop indices for looping through row items forwards/back
def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root: return

    res = []
    q = deque([root])
    left_first = True

    while q:
        next_level = deque()
        for node in q:
            if node.left: next_level.append(node.left)
            if node.right: next_level.append(node.right)
        
        start = 0 if left_first else len(q) - 1
        end = len(q) if left_first else -1
        row = []
        for i in range(start, end, 1 if left_first else -1):
            row.append(q[i].val)
        res.append(row)
        left_first = not left_first
        q = next_level
    return res

# Serialize and Deserialize Binary Tree Hard
# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
# took 1:01:00
# was getting sticky with the deserialize logic but fgiured it out
# solution isn't clean and was too lazy to set out time to find a clever way to make it clean
def serialize(self, root):
    """Encodes a tree to a single string.
    
    :type root: TreeNode
    :rtype: str
    """
    res = []
    q = deque([root])
    while q:
        next_level = deque()
        for node in q:
            if not node: continue
            next_level.append(node.left)
            next_level.append(node.right)
        res.append([node.val if node else None for node in q])
        q = next_level

    # convert res to a string
    string = ""
    for level in res:
        string += ("[")
        for node in level:
            string += str(node)
            string += (" ")
        string += ("]")
    return string

def deserialize(self, data):
    """Decodes your encoded data to tree.
    
    :type data: str
    :rtype: TreeNode
    """
    # convert string to list
    val = []
    levels = data.split("[")
    for level in levels:
        inside = level.split("]")
        for character in inside:
            val.append(character.split(" "))
    new = []
    for value in val:
        items = []
        for i in value:
            if i != "":
                items.append(None if i == "None" else int(i))
        if items:
            new.append(items)
    val = new
    tree = deque(val)

    root = TreeNode(tree.popleft()[0])
    if root.val == None:
        return None

    previous = [root]
    while tree:
        level = tree.popleft()
        i = 0
        for node in previous:
            if not node:
                i += 2
            else:
                break
        new_previous = []
        for node in level:
            node = TreeNode(node) if node != None else None
            while previous[floor(i/2)] == None:
                i+=1
            if i%2 == 0:
                previous[floor(i/2)].left = node
            else:
                previous[floor(i/2)].right = node
            i+=1
            new_previous.append(node)
        previous = new_previous
    return root

# Leetcode daily January 9th - 872. Leaf-Similar Trees Easy
# return true/false if two trees have the same 'leaf sequence' i.e, all of their leaves are the same values in the same order
# took 4:30 seconds and ran AND accepted first try
# TC: O(n) SC: O(1)
def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    
    def dfs(root):
        if not root: return []

        if not root.left and not root.right:
            return [root.val]

        return dfs(root.left) + dfs(root.right)

    return dfs(root1) == dfs(root2)

# LeetCode daily Jan 10th - Maximum Difference Between Node and Ancestor Medium
# Given the root of a binary tree, return the maximium differenc between any node and an ancestor node (a node above it)
# took about 12 mins
# TC: O(n), SC: O(n)
def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
    maximum = [0]
    
    def dfs(root):
        if not root: return None, None
        if not root.left and not root.right: return root.val, root.val
        
        left_min, left_max, right_min, right_max = None, None, None, None
        if root.left:
            left_min, left_max = dfs(root.left)
            maximum[0] = max(maximum[0], abs(root.val - left_max), abs(root.val - left_min))
        
        if root.right:
            right_min, right_max = dfs(root.right)
            maximum[0] = max(maximum[0], abs(root.val - right_min), abs(root.val - right_max))
        
        if root.left and root.right:
            return min(left_min, right_min, root.val), max(left_max, right_max, root.val)
        elif not root.left:
            return min(right_min, root.val), max(right_max, root.val)
        else:
            return min(left_min, root.val), max(left_max, root.val)
    dfs(root)
    return maximum[0]

# LeetCode daily January 11th - Amount of Time for Binary Tree to Be Infected LeetCode Medium
# https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/description/
# This actually took a bit because I was having a hard time coming up with a O(n) DFS solution.
# Even couldn't conceptualize it very well until I started setting it up and had to tweak it to get a solution
# took like 15 mins programming but spend a while thinking abt it while I was driving
# TC: O(n) SC: O(n)
# and I still think it's a pretty 'muddy' solution. Not very clear or simple
def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
    maximum = [0]
    
    def dfs(root):
        if not root: return 0, False
        
        left, found_left = dfs(root.left)
        right, found_right = dfs(root.right)
        
        if found_left or found_right:
            maximum[0] = max(maximum[0], left + right)
        else:
            maximum[0] = max(maximum[0], left, right)
        
        if root.val == start:
            return 1, True
        else:
            if found_left:
                return left + 1, True
            elif found_right:
                return right + 1, True
            else:
                return max(left, right) + 1, found_right or found_left
    dfs(root)
    return maximum[0]

# All Possible Full Binary Trees LeetCode Medium
# Given an integer n, return a list of all possible full binary trees with n nodes
# https://leetcode.com/problems/all-possible-full-binary-trees/
# Took over an hour because initially tried to do it with DP or bottom-up
# couldn't figure it out (got half-working solution that folded to edge cases)
# TC: O(n), SC: O(n)
# posted solution/explanation: https://leetcode.com/problems/all-possible-full-binary-trees/solutions/4647383/python-recursive-solution-simple-explanation-on-time-on-space/
memo = {}
def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
    if n == 1: return [TreeNode(0)]
    res = []

    # check if we already calculated this solution
    if n in Solution.memo: return Solution.memo[n]

    # loop from 1 to n-2 to determine the number of nodes to assign to the left and right subtrees
    # the left/right nodes will always add up to n-1 (+root == n)
    for i in range(1, n-1):
        left_nodes = i
        right_nodes = n - 1 - i

        # get the all of the possible left and right subtrees with the designed number of nodes for each side
        right_subtrees = Solution.allPossibleFBT(self, left_nodes)
        left_subtrees = Solution.allPossibleFBT(self, right_nodes)

        # add a tree to res for every combination of possible trees generated for left and right
        for j in range(len(right_subtrees)):
            for k in range(len(left_subtrees)):
                res.append(TreeNode(0, right_subtrees[j], left_subtrees[k]))
    
    # memoize and return the result
    Solution.memo[n] = res
    return res

# Find Bottom Left Tree Value LeetCode Medium
# https://leetcode.com/problems/find-bottom-left-tree-value/description/?envType=daily-question&envId=2024-02-28
# took like 6 mins
def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
    curr = [root]
    while curr:
        next_row = []
        for node in curr:
            if node.left:
                next_row.append(node.left)
            if node.right:
                next_row.append(node.right)
        if not next_row:
            break
        curr = next_row
    return curr[0].val

# Even Odd Tree LeetCode Medium
# https://leetcode.com/problems/even-odd-tree/?envType=daily-question&envId=2024-02-29
# Given the root of a binary tree, return true if the binary tree is Even-Odd, otherwise return false. (see question desc)
# took 6 mins
# TC: O(n), SC: O(n)
def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
    curr, level = [root], 0

    while curr:
        next_layer, prev = [], -float("inf") if level%2 == 0 else float("inf")
        for node in curr:
            if node.left: next_layer.append(node.left)
            if node.right: next_layer.append(node.right)
            if level%2 == 0 and (node.val%2 != 1 or node.val <= prev):
                return False
            elif level%2 == 1 and (node.val%2 != 0 or node.val >= prev):
                return False
            prev = node.val
        level += 1
        curr = next_layer
    return True

# Smallest String Starting From Leaf LeetCode Medium
# https://leetcode.com/problems/smallest-string-starting-from-leaf/description/
# took like 25
# TC: O(n), SC: O(n)
# approach: perform DFS and add .parent prop to all nodes (O(n) space) and also collect all leaf nodes
# then perform BFS from bottom up, collecting the minimum paths
# do this while there are one or more equal minimum paths or stop and return parth to top if there is only a single one
def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
    leaves = []
    
    def dfs(root, parent):
        if not root: return
        root.parent = parent
        dfs(root.right, root)
        dfs(root.left, root)
        if not root.right and not root.left:
            leaves.append(root)
        
    dfs(root, None)

    # perform reverse BFS from bottom to top
    ans = []
    while leaves:
        minn, new_leaves = 27, []
        for node in leaves:
            if node.val < minn:
                minn = node.val
                if node.parent:
                    new_leaves = [node.parent]
                else:
                    new_leaves = []
            elif node.val == minn and node.parent:
                new_leaves.append(node.parent)

        # add the minimum to the ans
        ans.append(chr(ord('`')+minn+1))

        # if any leaf in the queue is equal to the current minimum and doesn't have a parent, we can stop
        for leaf in leaves:
            if leaf.val == minn and not leaf.parent:
                return "".join(ans)
        leaves = new_leaves

        # if there is only one item in the leaves, we can return from this leaf to top
        if len(leaves) == 1:
            leaf = leaves[0]
            while leaf:
                ans.append(chr(ord('`')+leaf.val+1))
                leaf = leaf.parent
            return "".join(ans)
    return "".join(ans)

# Lowest Common Ancestor of a Binary Tree IV LeetCode Medium
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# Given the root of a binary tree and an array of TreeNode objects nodes, return the lowest common ancestor (LCA) of all the nodes in nodes. All the nodes will exist in the tree, and all values of the tree's nodes are unique.
# Note that I initially decided to return an array from every node reprresenting which of the nodes we have found out of the entire list of nodes
# but got a TLE (even though that was also an O(n) solution, it is technically O(n*nodes) solution which caused TLE)
# note that we didn't need to do this because why return a list? if we found *any* of the nodes in the child subtree then we must include that entire subtree
def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
    nodes = {node.val: True for node in nodes}

    def dfs(root):
        if not root: return False, None

        # check left and right
        found_in_left, left_LCA = dfs(root.left)
        found_in_right, right_LCA = dfs(root.right)

        if root.val in nodes or found_in_left and found_in_right: # if this node is in the list to find or both left and right subtrees include a node, this node could be the LCA
            return True, root
        elif found_in_right: # if a required is found only in right, return the LCA of the right
            return True, right_LCA
        elif found_in_left: # if a required is found only in left, return the LCA of the left
            return True, left_LCA
        else: # else, this node does not need to be included and there is no potential LCA here
            return False, None
    
    return dfs(root)[1]

# You are given a tree of n nodes where nodes are indexed from [1..n] and it is rooted at 1. You have to perform t swap operations on it, and after each swap operation print the in-order traversal of the current state of the tree.


# Distribute Coins in Binary Tree LeetCode Medium
# pretty tough and unintuitive. Definitely one of the hardest medium trees problems I've seen
# https://leetcode.com/problems/distribute-coins-in-binary-tree/
# this took like 40 mins
# : You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins in total throughout the whole tree.
# : In one move, we may choose two adjacent nodes and move one coin from one node to another. A move may be from parent to child, or from child to parent.
# : Return the minimum number of moves required to make every node have exactly one coin.
# TC: O(n), SC: O(n)
# notes: this is not a standard/simple recursive problem. The initial problem is to find the number of moves requires to re-distribute n coins in a tree with n nodes
# but note that the sub problem (the recursive call to the left or right child of a node) is not nessessarily n nodes and n coins.
# So having nodes == coins is not a precondition to the problem
# approach: we must see the problem through the perspective that since all nodes require a single coin, subtrees can end up with less or more coins that needed
# we consider the lack/excess of coins in a subtree as 'positive or nagative' vancancies. Andfor every node, we consider it's 'vacancy' value (whether it has extra coins to pass along to childrent or up to parent)
# and realize that for every vacancy, whether or positive or negative, the corresponding number of coins must be redistributed upwards (if pos) or downwards (if neg)
# so for every node, we check determine the vacancy value based on it's two children and it's own value, and add that vacancy to the total ans
# if my node is of value 4 and it has 2 child nodes (that have no further children), those two child nodes will have a vacancy value of -1, meaning that they require 1 coin to be sent downwards to them
# those nodes will already add their vacnacy values (1 each) to the total ans, but our parent node will have a total vacancy value of -1 (left) + -1 (right) + (4-1) (self - 1 coin required for self) = 1
# so that means this node must redictribute it's 1 excess coin upwards, so we add that 1 extra to the ans and return 1 upwards
# parameters and base case
def distributeCoins(self, root: Optional[TreeNode]) -> int:
    ans = [0]

    def dfs(root):
        if not root: return 0
        
        # check left and right subtrees for positive and negative vacancies
        vacancies_left = dfs(root.left)
        vacancies_right = dfs(root.right)
        node_quantity = root.val - 1 # the vacancy value of the current node

        # if we have positive or negative vacancies, these coins will need to be . So we acount for the moves in res
        ans[0] += abs(vacancies_left + vacancies_right + node_quantity)

        return vacancies_left + vacancies_right + node_quantity

    dfs(root)
    return ans[0]

# ^^TODO this similar HARD problem https://leetcode.com/problems/sum-of-distances-in-tree/


# Maximum Score After Applying Operations on a Tree LeetCode Medium
# https://leetcode.com/problems/maximum-score-after-applying-operations-on-a-tree/description/
# Pretty tough, not sure if it is just because I haven't been doing too many tree problems recently
# : There is an undirected tree with n nodes labeled from 0 to n - 1, and rooted at node 0. You are given a 2D integer array edges of length n - 1, where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.
# : You are also given a 0-indexed integer array values of length n, where values[i] is the value associated with the ith node.
# : You start with a score of 0. In one operation, you can:
# : Pick any node i.
# : Add values[i] to your score.
# : Set values[i] to 0.
# : A tree is healthy if the sum of values on the path from the root to any leaf node is different than zero.
# : Return the maximum score you can obtain after performing these operations on the tree any number of times so that it remains healthy.
# approach: for every path from root to node, at least a single node value must be preserved and not used in the total
# so for every node, we should traverse it's children (if any) and calculate the minimum nodes that can be preserved (see child_total_min_preserved) for each subtree such that the
# property is maintained. We should then compare the SUM of these minimum preserved nodes in each subtree to the current node itself
# if the node's value is less than all of the indivdual nodes summed, then we can simply preserve THIS node instead of the individual ones
# since this node is a common ancestor to all the child trees, it will be sufficient for all subtree paths if this one is preserved
# and we do this comparison for uevery node, gonig up the tree, and returning two calues 1. the tree accumulative total and 2. the sub of the minimum nodes ot preserve. 
# we then subtract the minimum preserved from the total as our result
# TC: O(n), SC: O(n)
# beats 93% runtime
# a unrelatred complication that I faced here was the fact that the tree was assumed to be rooted at 0 and the edges array contained one-directionsl edges
# so a case arose where, even though the tree was rooted at zero, the single edge away from zero was in the edges list like [[7,0]]
# so initially I added 0 to children[7] but not the other way around. And we never traversed past the route
# because of this, I had to account for edges both ways, and implement a second param called 'parent' to ensure we do not tarverse back up to the parent of a child
# because it's parent will also exist in it's children
def maximumScoreAfterOperations(self, edges: List[List[int]], values: List[int]) -> int:
    children = collections.defaultdict(list)

    for src, tar in edges:
        children[src].append(tar)
        children[tar].append(src)
    
    # returns: sum, min_deducible
    def dfs(node_num, parent):
        val = values[node_num]

        subtree_sums = child_total_min_preserved = 0
        for child in children[node_num]:
            if child == parent: continue
            child_sum, subtree_min = dfs(child, node_num)
            subtree_sums += child_sum
            child_total_min_preserved += subtree_min
        
        # if our node does not have any subtrees then the minimum node to preserve is this node itself
        if not subtree_sums:
            child_total_min_preserved = val

        # if keeping this node is better than resetting the min node from every child tree than we should consider keeping this node now
        if val < child_total_min_preserved:
            return subtree_sums + val, val
        else:
            return subtree_sums + val, child_total_min_preserved

    tree_sum, minimum_to_keep = dfs(0, None)
    return tree_sum - minimum_to_keep

# Find the Maximum Sum of Node Values LeetCode Hard
# https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/
# : (see entire description this is simplified) Given an undirected tree (not binary though) and an integer k. You can take any edge on the tree and XOR the node's values with k, that are attached to that edge.
# : return the largest possible sum of tree nodes that can be achieved by applying this operation any number of times to any number of edges.
# approach: tough question until w principles are understood
# 1. (a XOR b) XOR b = a => XORing a single number with the same number twice returns back to the same number
# 2. if we have a tree like: a -- b -- c -- d
# and we XOR nodes on edge c-d, and then XOR nodes on edge b-c, then c node has returned to it's initial state
# likewise, if we then XOR nodes on edge a-b, then a is flipped and b returns to it's initial state
# so based on the abaility to reverse these operations, we can esentially XOR any two nodes in the tree *regardless* of whether or not they are adjacent
# by applying those operations to 'propagate' the XOR operation through the tree while also resetting nodes as we do so
# so any two nodes can be XORed at a time any number of times
# so the question then boils down to: given a tree, what is the maximum total sum that can be made with the nodes of the tree, given that we cna XOR *any* two nodes at at time with k, any number of times
# approach: for every node in the tree, XOR it with k and compare it's value to the original
# if it is greater than the original, use that value and increment the number of "flips", otherwise keep the original
# at the end, if the number of flips is even, return the sum of all nodes
# else, we must either unflip one value that has been flipped or flip one value has not been flipped, as to take the smallest loss to the total
# we keep track of these with variables min_flip_loss and min_unflip_loss, the minimum losses to take if unflipped on at the end we had wanted to flip or flipping one we didn't. Whichever is smaller
# TC: O(n), SC: O(1)
# beats 98% runtime
def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
    # (A XOR B) XOR B = A
    # (A XOR B) XOR A = B

    flips = total = 0
    min_flip_loss = min_unflip_loss = float('inf')
    for i in range(len(nums)):
        flipped = nums[i] ^ k
        if flipped > nums[i]:
            flips += 1
            total += flipped
            min_unflip_loss = min(min_unflip_loss, flipped - nums[i])
        else:
            total += nums[i]
            min_flip_loss = min(min_flip_loss, nums[i] - flipped)

    if flips % 2 == 0:
        return total
    
    return total - min(min_flip_loss, min_unflip_loss)

# Find Number of Coins to Place in Tree Nodes LeetCode Hard
# https://leetcode.com/problems/find-number-of-coins-to-place-in-tree-nodes/description/
# TC: O(n), SC: O(n)
# : (see desc)
# a subproblem here is finding the maximum product of positive and negative numbers of values in the subtrees
# I initially oversimplified to returning the largest values, but large negative values can also be included such that they are multiplied together to get a large positive
# solution is to maintain the two largest negatives as well as the 3 largest positives so the follwing combinations can be used to get the largest product from 3 of them:
# a: 2 largest negative and one largest positive
# b: 3 largest positives
def placedCoins(self, edges: List[List[int]], cost: List[int]) -> List[int]:
    children = collections.defaultdict(list)
    for from_, to_ in edges:
        children[from_].append(to_)
        children[to_].append(from_)

    ans = [0]*len(cost)

    # we use this function to traverse the largest and smallest items returned froim the children. We traverse largest 3 times to get the 3 largest and smallest twice to get the 2 largest negatives
    # these is an O(3n) and O(3n) runtimes = O(n)
    # rather than sorting all of the largest and smallests returned by the children, which is O(nlogn) for both operations
    def updateLargestAndSmallest(all_largest, all_smallest):
        largest_3, smallest_2 = [-float('inf'), -float('inf'), -float('inf')], [float('inf'), float('inf')]
        for num in all_largest:
            if num > largest_3[0]:
                largest_3[2] = largest_3[1]
                largest_3[1] = largest_3[0]
                largest_3[0] = num
            elif num > largest_3[1]:
                largest_3[2] = largest_3[1]
                largest_3[1] = num
            elif num > largest_3[2]:
                largest_3[2] = num
        
        for num in all_smallest:
            if num < smallest_2[0]:
                smallest_2[1] = smallest_2[0]
                smallest_2[0] = num
            elif num < smallest_2[1]:
                smallest_2[1] = num
        
        return [num for num in largest_3 if num != -float('inf')], [num for num in smallest_2 if num != float('inf')]


    def dfs(node_num, parent):

        # explore the children and determine the size of the subtrees
        tree_size = 0
        tree_largest_costs = [cost[node_num]]
        tree_largest_neg_costs = [cost[node_num]]
        for child in children[node_num]:
            if child == parent: continue
            subtree_size, subtree_largest_3_pos, subtree_largest_two_neg = dfs(child, node_num)
            tree_size += subtree_size
            tree_largest_costs += subtree_largest_3_pos
            tree_largest_neg_costs += subtree_largest_two_neg


        # sorts and trim the largest costs array
        largest_pos, largest_neg = updateLargestAndSmallest(tree_largest_costs, tree_largest_neg_costs)

        # note the above funciton call is equivalent to the following two lines
        # largest_pos = sorted(tree_largest_costs, reverse=True)[:4]
        # largest_neg = sorted(tree_largest_neg_costs)[:3]

        # calculate the cost of the node and add it to ans
        # note I have hbere if subtree > 1 (assuming that the curren't node is counted in it's own subtree)
        node_coins = 1
        if tree_size > 1:
            node_coins = max(0, largest_pos[0]*largest_pos[1]*largest_pos[2], largest_pos[0]*largest_neg[0]*largest_neg[1])
        ans[node_num] = node_coins

        # return the number of nodes in this overall subtree and the 3 maximum cost values to the parent
        return tree_size + 1, largest_pos, largest_neg

    dfs(0, None)
    return ans