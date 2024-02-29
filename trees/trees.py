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
        print([node.val for node in curr])
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