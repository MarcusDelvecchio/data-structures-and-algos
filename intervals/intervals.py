
# Insert Interval LeetCode Medium
# https://leetcode.com/problems/insert-interval/description/
# TC: O(n), SC: O(n) -> we could def do a O(1) solution but fine for now
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    i = 0
    res = []

    # add intervals that have start times coming before the interval being added
    while i < len(intervals) and intervals[i][0] < newInterval[0]:
        res.append(intervals[i])
        i+= 1
    
    # add the new interval or merge it with the most recent one
    if not res or res[-1][1] < newInterval[0]:   # if recent one ends before new one starts, just add
        res.append(newInterval)
    else:
        res[-1][1] = max(newInterval[1], res[-1][1]) # if new one starts before recent one ends, update end interval (be be later of the two end dates)

    # then while our recent interval's end date extends past further intervals's start date, extend that end date
    while i < len(intervals) and res[-1][1] >= intervals[i][0]:
        res[-1][1] = max(intervals[i][1], res[-1][1]) #  to the later end date of either the recent one or the one being added
        i += 1

    # then just add any remaining intervals
    while i < len(intervals):
        res.append(intervals[i])
        i += 1
    
    return res