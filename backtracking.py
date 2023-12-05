#  Binary Tree Paths
# https://leetcode.com/problems/binary-tree-paths/description/
# took like 4 mins idk how it's backtracking though
def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    paths = []
    
    def dfs(root, p):
        if not root: return None

        p = p + ("->" if len(p) else "") + str(root.val)
        if not root.left and not root.right:
            paths.append(p)
        else:
            dfs(root.left, p)
            dfs(root.right, p)
    dfs(root, "")
    return paths

# Combination Sum LeetCode Medium 
# https://leetcode.com/problems/combination-sum/solutions/429538/general-backtracking-questions-solutions-in-python-for-reference/
# took like 13 mins
# still not sure where or what makes it specifically backtracking
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    solutions = {}
    
    def get_combos(target, combination, idx):
        for i in range(idx, len(candidates)):
            if target - candidates[i] > 0:
                target -= candidates[i]
                combination.append(candidates[i])
                get_combos(target, combination, i)

                # reset values after returning
                target += candidates[i]
                combination.pop()
            elif target - candidates[i] == 0:
                solutions[tuple(sorted(combination + [candidates[i]]))] = True
        # if have gone through all possible solutions and no more to be explored: backtrack
        return
    get_combos(target, [], 0)
    return [list(combination) for combination in solutions.keys()]

# time complexity is ??

# Combinations LeetCode Medium
# https://leetcode.com/problems/combinations/description/
# took 18 mins bc took a bit to think about the solution
# after looking into other solutions I updated the solution with the following:
# 1. at first I thought I would have to use a hash map for the res and curr values. When I would find a solution to the backtracking problem I would then sort the array of values and insert
# it into the res hash map. But I relaized there is no need to do this as all of the solutions will automatically be unique and sorted, since the indx value prevents duplicate values from being selected anyways
# see that solution here https://leetcode.com/problems/combinations/submissions/1112553976/ - but this one still uses hashmap for curr, which is also unnecessary
def combine(self, n: int, k: int) -> List[List[int]]:
    res = []

    def combos(l, curr, idx):
        # backtrack if we get a solution
        if l == k:
            res.append(curr.copy())
            return
            
        for i in range(idx, n + 1):
            curr.append(i)
            combos(l + 1, curr, i + 1)
            curr.pop()
    combos(0, [], 1)
    return res