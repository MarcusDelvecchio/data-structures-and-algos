# Open the Lock LeetCode Medium
# give 4 digit lock where you can change one digit one value at a time, and certain combinations are 'deadends' that will cause the lock to get stuck,
# return the minimum number of combination changes that can be applied to get to the target combo, or -1 if it is impossible
# https://leetcode.com/problems/open-the-lock/description/
# TC: O(n), SC: O(n)
# tried dfs for 25 mins, then restarted and complete with bfs in 5
def openLock(self, deadends: List[str], target: str) -> int:
    q = collections.deque(["0000"])
    deadends = set(deadends)

    depth = 0
    seen = set()
    while q:
        size = len(q)
        for _ in range(size):
            lock = q.popleft()
            if lock in deadends: continue
            if lock == target:
                return depth
            for i in range(4):
                # increasing
                increased = int(lock[i]) + 1
                increased_lock = lock[:i] + str(increased if increased != 10 else 0) + lock[i+1:]
                if increased_lock not in seen:
                    seen.add(increased_lock)
                    q.append(increased_lock)

                # decreasing
                decreased = int(lock[i]) - 1
                decreased_lock = lock[:i] + str(decreased if decreased != -1 else 9) + lock[i+1:]
                if decreased_lock not in seen:
                    seen.add(decreased_lock)
                    q.append(decreased_lock)
        depth += 1
    return -1