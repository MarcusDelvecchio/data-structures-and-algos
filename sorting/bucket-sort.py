# Contains Duplicate III LeetCode Hard
# bucket sort / bucketing approach
# each bucket represents a range of values. we then iterate over the array and check if there's a nearby element in the same or adjacent buckets that satisfies the condition.
# https://leetcode.com/problems/contains-duplicate-iii/description/
# simplified: given an array of elements, return true if within a window size w, there exists two elements with k or less absolute difference
# TC: O(n), SC: O(n)
# note: had difficulty determining the formula for the bucket divisor
# in the bucket sort wiki it says that bucketdivisor (M) is max(nums) + 1, but here we do it differently
# "By bucketing numbers properly, this problem becomes almost identical to Contains Duplicate II except that numbers in adjacent buckets need to be check as well."
def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    bucketDivisor = valueDiff + 1 # determine the bucket index for each elemment. Why is this not max(nums) + 1 
    buckets = {}
    for idx, num in enumerate(nums):
        bucket = num // bucketDivisor # math.floor(num * len(nums) / bucketDivisor)
        if bucket in buckets:
            return True

        # add R to it's bucket and ensure there is not already a value there
        if bucket+1 in buckets and abs(buckets[bucket+1] - num) <= valueDiff:
            return True
        if bucket-1 in buckets and abs(buckets[bucket-1] - num) <= valueDiff:
            return True
        buckets[bucket] = num
            
        # if needed, remove the element at L from it's bucket before moving R forward
        if idx >= indexDiff:
            L_bucket = nums[idx - indexDiff] // bucketDivisor # math.floor(nums[R - indexDiff] * len(nums) / bucketDivisor)
            del buckets[L_bucket]
    return False 

