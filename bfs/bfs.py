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

# Freedom Trail LeetCode Hard
# You are given a string 'ring' representing a circular ring lock/door and a target key. You want to spell out the key by rotating the ring counter/clockwise
# letter by letter, where each turn counts as a step and you must also press a button to submit a letter, counting as a step
# calculate the minimum number of steps to spell the entire key
# https://leetcode.com/problems/freedom-trail/description/
# TC: O(key*ring), SC: O(ring)
# tried to do a greedy solution for like 45 mins and realized greedy wouldn't work (going to closest next letter isn't always favourable)
# then did BFS approach took 5 mins but got TLE because the queue could have have duplicate possibilities where we could have different total steps
# while being at the same ring idx, so used a hash map to continuously take the minimum number of steps - see logic below
# took 15 mins for BFS
# approach: perform BFS starting from the initial index, adding all of the possible next indices to the queue, (to reach the next key character) along with the associated minimum steps to get to that index
# the queue size will have a max size of len(ring)
def findRotateSteps(self, ring: str, key: str) -> int:
    # create a map of of locations of all letters in format { letter: [sorted, indices, where, letter, appears]}
    chars = collections.defaultdict(list)
    for idx, char in enumerate(ring):
        chars[char].append(idx)

    # queue representing the possible current ring index for the current key character, along with the minimum associated (accumulative path) of getting there
    q = collections.deque([(0,0)]) # queue values represent (total steps, ring index)
    key_idx = 0
    while key_idx != len(key): # iterate through the key characters, adding possible next index/paths total to the queue
        size = len(q)
        next_q = collections.defaultdict(lambda:float('inf')) # compile the next values to be added to the queue. using a hasmap here because we always want the minimum path for each possible next index. If we end up at idx 2 for our next target value and this could be with a path total of 3 vs 6, we can ignore the path total 6
        for _ in range(size):
            steps, ring_idx = q.popleft()
            for target_char_idx in chars[key[key_idx]]:
                dist_direct = abs(target_char_idx - ring_idx) # distance directly to the possible next target item
                dist_wrapped = len(ring) - dist_direct # distance to the possible next target item by wrapping around
                next_q[target_char_idx] = min(steps + min(dist_direct, dist_wrapped), next_q[target_char_idx]) # update the next value with the minimum value
        for i in next_q:
            q.append((next_q[i], i))
        key_idx += 1
    return min(q)[0] + len(key)