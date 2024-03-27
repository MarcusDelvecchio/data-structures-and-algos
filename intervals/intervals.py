
# given a list of SORTED intervals, this function returns whether or not there are overlapping intervals
def canAttendMeetings(self, intervals: List[Interval]) -> bool:
    prevEnd = intervals[0].end
    for interval in intervals[1:]:
        if interval.start < prevEnd:
            return False
        else:
            prevEnd = end
    return True

# Meeting Schedule LeetCode Easy
# https://neetcode.io/problems/meeting-schedule
# given a list of UNSORTED meeting intervals determine if a person could add all meetings to their schedule without any conflicts.
# (are there any overlapping intervals)
# TC: O(nlogn) -> there is no better solution than just simply sorting it
# SC: O(1)
def canAttendMeetings(self, intervals: List[Interval]) -> bool:
    if not intervals: return True
    intervals.sort(key=lambda x: x.start)
    prevEnd = intervals[0].end
    for interval in intervals[1:]:
        if interval.start < prevEnd:
            return False
        prevEnd = interval.end
    return True

# Insert Interval LeetCode Medium
# https://leetcode.com/problems/insert-interval/description/
# TC: O(n), SC: O(n) -> we could def do a O(1) solution but fine for now
# took 17 mins
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
        res[-1][1] = max(intervals[i][1], res[-1][1])  # extend to the later end date of either the recent one or the one being added
        i += 1

    # then just add any remaining intervals
    while i < len(intervals):
        res.append(intervals[i])
        i += 1
    
    return res

# Merge Intervals LeetCode Medium
# https://leetcode.com/problems/merge-intervals/
# TC: O(nlogn), SC: O(n) -> SC can definitely be reduced to O(1)
# took 9:15
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    i = 0
    res = []
    while i < len(intervals):
        res.append(intervals[i])
        j = i+1
        while j < len(intervals) and intervals[i][1] >= intervals[j][0]:
            res[-1][1] = max(res[-1][1], intervals[j][1])
            j += 1
        i = j
    return res

# Non-overlapping Intervals LeetCode Medium
# https://leetcode.com/problems/non-overlapping-intervals/description/
# looked at solution after being stuck trying to implement recursive brute force solution for a while
# didn't realize the trick with the end dates
# TC: O(nlogn), SC: O(1)
# todo review
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    #  approach: sort the intervals, and iterate forwards in ascending start time order
    #  when two intervals over lap, delete the one that has a later end date
    #  then compare the next value with the end date of the recent item
    intervals.sort()

    curr_end = intervals[0][1]
    res = 0
    for start, end in intervals[1:]:
        if start < curr_end:
            res += 1
            curr_end = min(curr_end, end)
        else:
            curr_end = end
    return res

# Meeting Schedule II LeetCode Medium
# https://neetcode.io/problems/meeting-schedule-ii
# Given an array of meeting time interval objects consisting of start and end times, find the minimum number of days required to schedule all meetings without any conflicts.
# TC: O(nlogn), SC: O(n)
def minMeetingRooms(self, intervals: List[Interval]) -> int:
    # problem essentially asking "whats the most number of meetings occurring at a single moment?"
    # convert meetings to list of times of type start or end
    # iterate forward in the list and tracking the number of 'open' intervals: 
    # every time a start time comes add 1 to the num of open intervals and every time an end time comes subtract 1
    times = []
    for interval in intervals:
        times.append([interval.start, 1])
        times.append([interval.end, -1])
    times.sort()
    open_intervals, maxx = 0, 0
    for time in times:
        open_intervals += time[1]
        maxx = max(maxx, open_intervals)
    return maxx