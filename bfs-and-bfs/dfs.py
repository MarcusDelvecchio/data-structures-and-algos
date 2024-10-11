from collections import defaultdict

# Keys and Rooms LeetCode Medium
# https://leetcode.com/problems/keys-and-rooms/description/
# TC: O(n) SC: O(n)
def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
    seen = set()

    def dfs(roomIdx):
        if roomIdx in seen: return
        seen.add(roomIdx)
        for key in rooms[roomIdx]:
            dfs(key)

    dfs(0)
    for i in range(len(rooms)):
        if i not in seen: return False
    return True

# Detonate the Maximum Bombs LeetCode Medium
# https://leetcode.com/problems/detonate-the-maximum-bombs/description/
# TC: O(n^2) SC: O(n)
def maximumDetonation(self, bombs: List[List[int]]) -> int:
    can_detonate = defaultdict(list)

    # for each bomb, determine the bombs that it can explode
    for b1 in range(len(bombs)):
        for b2 in range(len(bombs)):
            if b1 == b2: continue
            x1, y1, r1 = bombs[b1]
            x2, y2, r2 = bombs[b2]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= r1:
                can_detonate[b1].append(b2)

    memo = defaultdict(int)
    def dfs(idx, seen):
        if idx in seen: return 0
        seen.add(idx)
        count = 1
        for b2 in can_detonate[idx]:
            count += dfs(b2, seen)
        return count

    # perform a BFS to determine which bomb can detonate the most bombs
    best = 1
    for b in range(len(bombs)): # RuntimeError: dictionary changed size during iteration
        best = max(best, dfs(b, set()))
    return best