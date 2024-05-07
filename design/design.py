# Design Hit Counter LeetCode Medium
# https://leetcode.com/problems/design-hit-counter/description/
# Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).
# Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive roughly at the same time.
# * Implement the HitCounter class:
# : HitCounter() Initializes the object of the hit counter system.
# : void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
# : int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., the past 300 seconds).
# TC: O(1) for hit, O(logn) for getHits()
# approach: create array of items representing hits across time, when we look for hits (300 seconds) before a timestamp, we:
# 1. do binary search to find the index of the last hit before timestamp - represnting total number of hits up to timetstamp
# 2. do binary search to find the index of the last hit before 300s before timestamp - represnting total number of hits out of the 300s window
# 3. subtract total - expired = hits in range
class HitCounter:

    def __init__(self):
        self.hits = []

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        before = self.getHitsBeforeTimeStamp(timestamp) # get the number of hits that take place up to and including timestamp
        expired = self.getHitsBeforeTimeStamp(timestamp - 300) # get the number of hits that take place up to (and including) 5 mins before timestamp
        return before - expired # return difference

    # binary search looking for the index of the greatest value that is less than or equal to timestamp
    # and there could be duplicate items so we have to find the one with the larger index
    def getHitsBeforeTimeStamp(self, timestamp):
        left, right = 0, len(self.hits)-1
        closest = float('inf')
        closest_idx = -1
        while left <= right:
            mid_idx = (left+right)//2
            # update closest value
            if self.hits[mid_idx] <= timestamp and timestamp-self.hits[mid_idx] <= closest and mid_idx > closest_idx:
                closest_idx = mid_idx
                closest = timestamp-self.hits[mid_idx]
            
            # shift
            if self.hits[mid_idx] > timestamp:
                right = mid_idx - 1
            else: # mid_idx < len(hits) - 1 and hits[mid_idx] < timestamp: not we shift right if they are equal as well because there could be a later item
                left = mid_idx + 1
        return closest_idx