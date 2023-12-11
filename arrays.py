# took about an hour but O(n+m) not not true Hard solution
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    p1, p2 = 0, 0 

    new = []
    length = 0
    even = (len(nums1) + len(nums2))%2 == 0
    while length < floor((len(nums1) + len(nums2))/2 + 1):
        if p1 == len(nums1):
            if p2 == len(nums2):
                break
            new.append(nums2[p2])
            p2 += 1
        elif p2 == len(nums2) or nums1[p1] < nums2[p2]:
            new.append(nums1[p1])
            p1 += 1
        else:
            new.append(nums2[p2])
            p2 += 1
        length += 1
        
    if even:
        return (new[len(new) - 2] + new[len(new) - 1]) / 2
    else:
        return new[-1] 

# Trapping Rain Water LeetCode Hard
# https://leetcode.com/problems/trapping-rain-water/description/
# took 34:00 mins
# wish the solution was cleaner but is what it is
def trap(self, height: List[int]) -> int:
    w, current, temp = 0, 0, 0
    cont = False

    # put all vals into a dict
    vals = defaultdict(int)
    for value in height:
        vals[value] += 1

    for i in range(len(height)):
        if cont or height[i] >= current:
            cont = False
            w += temp

            # look ahead to find if a value exists that is greater or equal to the current val
            j, found = i+1, False
            while j < len(height):
                if height[j] >= height[i]:
                    found = True
                    break
                j += 1
            if found:
                current = height[i]
            else:
                t = height[i] - 1
                while t > 0:
                    if t in vals and vals[t]:
                        current = t
                        break
                    t -= 1
                if t == 0:
                    cont = True
            temp = 0
        else:
            temp += current - height[i]
        vals[height[i]] -= 1
    return w

# note on "amortized" complexity of appending to array
# That means the whole operation of pushing n objects onto the list is O(n). If we amortize that per element, it's O(n)/n = O(1).
# https://stackoverflow.com/questions/33044883/why-is-the-time-complexity-of-pythons-list-append-method-o1