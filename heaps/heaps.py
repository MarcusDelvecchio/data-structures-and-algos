import heapq as hq
from collections import Counter
from heapq import heappush, heappop, heapify

# iterates through a given list and returns true if it is a valid max-heap else false
def is_max_heap(l):
    for i in range(len(l)//2): 
        curr = l[i]
        left = l[(i+1)*2-1] if (i+1)*2-1 < len(l) else 0
        right = l[(i+1)*2] if (i+1)*2 < len(l) else 0
        if left > curr or right > curr: 
            return False
    return True

# iterates through a given list and returns true if it is valid min-heap format
def is_min_heap(l):
    for i in range(len(l)//2):
        curr = l[i]
        left = l[(i+1)*2-1] if (i+1)*2-1 < len(l) else None
        right = l[(i+2)*2] if (i+2)*2 < len(l) else None
        if right != right < curr or left and left < curr:
            return False
    return True

# iterates through a give list and returns true if it is a min heap or a max heap *with one pass*
def is_min_or_max(l):
    is_min = True
    is_max = True
    for i in range(len(l)//2):
        curr = l[i]
        left = l[(i+1)*2-1] if (i+1)*2-1 < len(l) else None
        right = l[(i+1)*2] if (i+1)*2 < len(l) else None
        if right != None and right < curr or left != None and left < curr:
            is_min = False
        if right != None and right > curr or left != None and left > curr:
            is_max = False
    return is_min or is_max

# takes a max heap and inserts a given item into it
def max_heap_insert(l, n):
    # add element to the end of the list
    l.append(n)

    # continuously compare n with its parent and switch if needed
    idx = len(l) - 1
    while l[idx] > l[idx//2]:
        temp = l[idx//2]
        l[idx//2] = l[idx]
        l[idx] = temp
        idx = idx//2
    return l

# deletes (extracts) the value from the root of a given max heap l
# and returns the extracted root
def max_heap_extract(l):
    # replace root with end
    root = l[0]
    l[0] = l[-1]
    del l[-1]

    # continuously sift down while parent is smaller than smaller child
    curr = 0
    idx = 1 if l[1] > l[2] else 2
    while l[idx] > l[curr]:
        temp = l[idx]
        l[idx] = l[curr]
        l[curr] = temp
        curr = idx
        left_idx = (curr+1)*2-1
        right_idx = (curr+1)*2
        
        if left_idx >= len(l):
            break

        idx = left_idx if right_idx >= len(l) else max(left_idx, right_idx, key=lambda x: l[x])
        # idx = left_idx if right_idx >= len(l) or l[left_idx] > l[right_idx] else right_idx
    return root

# heap insert then extract operation
# more efficient then doing heap insert and then heap extract separately
# TC = O(logn)
def max_heap_push_pop(l, val):
    if val > l[0]: return val
    root = l[0]
    l[0] = val

    # down sift (down-heap) the node until it is in the correct location
    curr = 0
    idx = max(1,2, key=lambda x: l[x])
    while l[idx] > l[curr]:
        temp = l[curr]
        l[curr] = l[idx]
        l[idx] = temp
        curr = idx

        # get the new node to compare to (left or right)
        left_idx = (curr*2)+1
        right_idx = (curr*2)+2

        if left_idx > len(l):
            break
        
        idx = left_idx if right_idx > len(l) else max(left_idx, right_idx, key=lambda x: l[x])
    return root

def max_heap_del_by_val(l, val):
    # get the array index of the value we want to delete
    for curr, node in enumerate(l):
        if node == val: break
    
    # swap this element with the last element and down heap until max-heap property is restored
    l[curr] = l[-1]
    del l[-1]

    # ensure heap is not small enough that we can simply return now
    if len(l) < 2: return
    left, right = get_children(curr)

    if left > len(l):
        return

    idx = left if right > len(l) or l[left] > l[right] else right
    while l[idx] > l[curr]:
        temp = l[idx]
        l[idx] = l[curr]
        l[curr] = temp
        curr = idx

        # determine which side we should sift to next
        left_idx, right_idx = get_children(curr)


        if left_idx >= len(l):
            break

        idx = left_idx if right_idx >= len(l) or max(left_idx, right_idx, key=lambda x: l[x]) else right_idx
    return


# Kth Largest Element in an Array
# maxheap, put all numbers in the maxheap and pop k times.
# https://leetcode.com/problems/kth-largest-element-in-an-array/description/
# gets the kth largest element in a list using a heap
# TC O(n+klogn) (O(n) to build heap from unsorted array and O(logn) to heapify after every pop) | SC O(n) (list is transformed into a heap in-place)
# see this answer for interesting description of how to make the solution more and more efficient https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/762174/4-python-solutions-with-step-by-step-optimization-plus-time-and-space-analysis/
def findKthLargest(self, nums: List[int], k: int) -> int:
    nums_max = [-num for num in nums]
    heapq.heapify(nums_max)
    val = None
    for _ in range(k):
        val = heapq.heappop(nums_max)
    return -val

# Last Stone Weight LeetCode Easy
# https://leetcode.com/problems/last-stone-weight/description/
# TC O(n) (building) + O(n(logn+logn)) = O(n) + O(nlogn) = O(nlogn)
#    ^building           ^for every 2 inputs we pop 2 (which takes logn to heapify)a and insert 1
def lastStoneWeight(self, stones: List[int]) -> int:
    stones = [-num for num in stones]
    heapq.heapify(stones)
    while len(stones) > 1:
        stone_1 = -heapq.heappop(stones)
        stone_2 = -heapq.heappop(stones)

        if stone_1 > stone_2:
            heapq.heappush(stones, stone_2-stone_1)
    return -stones[0] if stones else 0

# Largest Number After Digit Swaps by Parity LeetCode Easy
# https://leetcode.com/problems/largest-number-after-digit-swaps-by-parity/description/
# You are given a positive integer num. You may swap any two digits of num that have the same parity (i.e. both odd digits or both even digits). Return the largest possible value of num after any number of swaps.
# solution: create two heaps with even and odd values, iterate throught the original number pop from corresponding heap to get the greatest values
# TC O(n) to split into even/odd lists, O(n) to convert the two lists to heaps, O(nlogn) to pop from either heap for every n
# TC = O(nlogn) SC = O(n) (two new lists are created of size n which are then heapified along with a string)
def largestInteger(self, num: int) -> int:
    num, res, num_odd, num_even = str(num), "", [], []
    for digit in num:
        num_odd.append(-int(digit)) if int(digit)%2 == 1 else num_even.append(-int(digit))
    heapq.heapify(num_even)
    heapq.heapify(num_odd)
    for digit in num:
        res += str(-heapq.heappop(num_even)) if int(digit)%2 == 0 else str(-heapq.heappop(num_odd))
    return int(res)


# Top K Frequent Elements LeetCode Medium
# https://leetcode.com/problems/top-k-frequent-elements/description/
# TC: O(n) + O(n) + O(n) + O(klogn) = O(n + klogn) SC: O(n)
# TC = O(n) to iterate and convert nums to a dict, O(n) to convert dict to list of (freq, num) pairs, O(n) to build heap from this list and O(klogn) to pop k items
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    freq, res = defaultdict(int), []
    # create dict with all frequencies. O(n)
    for num in nums:
        freq[num] += 1

    # convert dict to a list. O(n)
    n = [(-freq[num], num) for num in freq.keys()]

    # heapify this list. O(n)
    heapq.heapify(n)

    # extract the largest item from the heap k times. O(klogn)
    for _ in range(k):
        res.append(heapq.heappop(n)[1])
    return res

# more efficient solution to above
# instead of creating a heap with all n items and then popping the top k, we can simply create a heap with only k items, and maintain this heap. Then once we have gone through all the items we can simply pop the remaining k items from the heap
# this solution is faster than the one above especially if k is significantly less than n
# TC = O(n) to create dict + O(n) to convert to lists + O(k) to heapify k items + O((n-k)logk) to pushpop the remaining k items + O(klogk) to pop the final k items
# TC = O(n + (n-k)logk + klogk)
# can be simplified because k is assumed to be smaller than n so the second term is dominant giving us
# TC = O((n-k)logk) = O(nlogk)
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    freq, res, first_k, remaining = defaultdict(int), [], [], []
    # create dict of frequencies of items O(n)
    for num in nums:
        freq[num] += 1
    
    # split items into first k and remaining items O(n)
    for idx, num in enumerate(freq.keys()):
        first_k.append((freq[num], num)) if idx < k else remaining.append((freq[num], num))

    # heapify first k items O(k)
    heapq.heapify(first_k)

    # pushpop the remaining items O((n-k)logk)
    while remaining:
        heapq.heappushpop(first_k, remaining.pop())
    
    # pop the remaining k items O(klogk)
    # note we do not need to do this because it can be any order
    while first_k:
        res.append(heapq.heappop(first_k)[1])
    return res

# the above solution can also be simplified using pythons heapq.nlargest function just to clean things up
# this nlargest function does exactly what we have done above. Creating a heap of k (where k = n in 'nlargest') elements and then pushpoping the remaining items to maintain the n/k largest
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    freq, first_k, remaining = Counter(nums), [], []

    # get nlargest values using heapq function. O(nlogk)
    largest = heapq.nlargest(k, [(freq[num], num) for num in freq.keys()])
    return [item[1] for item in largest]

# greedy solution
# adds k pairs to a tree and then attempts to pushpop all remaining possible items into the tree
# problem with though is there are too many items and the lists are ascending - we do not need to do this
# we can add pairs as we go along - but 
def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    h, res = [], []
    for i in range(len(nums1)):
        for j in range(len(nums2)):
            if len(h) < k:
                h.append((-(nums1[i]+nums2[j]),nums1[i],nums2[j]))
            elif len(h) == k:
                heapq.heapify(h)
                heapq.heappushpop(h, (-(nums1[i]+nums2[j]), nums1[i],nums2[j]))
            else:
                heapq.heappushpop(h, (-(nums1[i]+nums2[j]), nums1[i],nums2[j]))
    while h:
        pair = heapq.heappop(h)
        res.append([pair[1], pair[2]])
    return reversed(res)

# Sort Characters By Frequency LeetCode Medium
# https://leetcode.com/problems/sort-characters-by-frequency/description/
# took 6 mins
# TC: O(n) to convert to list, O(n) to convert to dict, O(n) to heapify, O(nlogn) to empty heap, O(n) to build res array and O(n) to join it.
# TC: O(nlogn)
# note that doing res += char*-count where res is a string instead of a list would be O(n^2) it seems because string concatenation is O(N+M) where N and M are the lengths of the two strings being concatenated. So this would be O(n+m) for EVERY char in n. So .join is much faster
class Solution:
    def frequencySort(self, s: str) -> str:
        freq, res = Counter(list(s)), []
        h = [(-freq[c], c) for c in freq.keys()]
        heapify(h)
        while h:
            count, char = heappop(h)
            res.append(char*-count)
        return "".join(res)

# Task Scheduler LeetCode Medium
# https://leetcode.com/problems/task-scheduler/description/
# this is a medium btu took me over an hour
# had so many issues with the 'waiting' queue implementation and had a hard time conceptualizing the units of time and 'one off' issues (ex how the waiting array is length wait time + 1)
def leastInterval(self, tasks: List[str], n: int) -> int:
    counts, ans, waiting = Counter(tasks), 0, deque([None]*(n+1))
    h = [(-counts[task], task) for task in counts]
    heapify(h)
    num_waiting = 0

    while h or num_waiting:
        next = waiting.pop()
        if next:
            heappush(h, (next[0], next[1]))
            num_waiting -= 1
        if h:
            freq, task = heappop(h)          
            if -freq > 1:
                waiting.appendleft((freq+1, task))
                num_waiting += 1
            else:
                waiting.appendleft(None)
        else:
            waiting.appendleft(None)
        ans += 1
    return ans  

# todo do this solution with just a counter
# get n+1 most common items at a time, increment units of time by n+1, decrement count of items actually popped (ex only 4 popped but n = 5) and then repeat

# K Closest Points to Origin LeetCode Medium
# took 6 mins
# using nsmallest but could also do manually if I feel like it
# could come back to practice
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    distances = [(sqrt(p[0]*p[0]+p[1]*p[1]), p[0], p[1]) for p in points]
    return [[p[1], p[2]] for p in nsmallest(k, distances)]

# Find K Closest Elements LeetCode Medium
# https://leetcode.com/problems/find-k-closest-elements/description/
# same with the abopve solution, did very inefficiently because the heap questions are all becoming the same
# should implement nsmallest at least for better efficiency and find a cleaner way to do it than sorting it again at the end (Its 3am and I'm nearly at 9 hours today so getting off)
# took 6 mins
def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    distances, res = [(abs(val - x), val) for val in arr], []
    heapify(distances)
    for _ in range(k):
        res.append(heappop(distances)[1])
    return sorted(res)


# import heapq as hq
# Least Number of Unique Integers after K Removals LeetCode Medium
# Given an array of integers arr and an integer k. Find the least number of unique integers after removing exactly k elements.
# https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/description/?envType=daily-question&envId=2024-02-16
# TC: O(n), SC: O(n)
def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
    freq, res = defaultdict(int), []

    # gather number frequencies - O(n)
    for num in arr:
        freq[num] += 1
    
    # convert frequency dict to list - O(n)
    n  = [(freq[num], num) for num in freq]
    uniques = len(freq)

    # heapify list - O(n)
    hq.heapify(n)

    # heappop the items with the lowest frequency for the k items being removed
    # O(klogn)
    while k > 0:
        smallest = hq.heappop(n)
        if smallest[0] <= k:
            k -= smallest[0]
            uniques -= 1
        else:
            break
    return uniques

# optimization from above #1 - only add k elements to the heap so that heapify is O(nlogk) rather than O(n)
# THIS WRITE UP IS PROBABLY WRONG - todo go down rabbit role and look into dif between O(n) and O(klogn) and O(n + klogn) and O(nlogn) (see https://www.google.com/search?client=safari&sca_esv=a2b78866b809fff6&rls=en&sxsrf=ACQVn0_hVw3LtI08E70Cz3Ux0Jk7Cnor-Q:1708121466761&q=O(n)+vs+%22klogn%22+OR+%22klog(n)%22&nfpr=1&sa=X&ved=2ahUKEwistImq8LCEAxV4kIkEHYKWAC4QvgUoAXoECAgQAw&biw=1314&bih=751&dpr=2
# (heapifying the k least frequent elements is O(k) to heapify the first k elements (any k elements just to create the heap) and then O(nlogk) to get the k most infrequent of all n elements (O(logk) to pushpop elements for all n to get a heap of size k = O(nlogk)))
# but to do this, the initial array also needs to be sorted
# to sort the array would be nlogn again, which defeats the purpose of this optimization
# so instead, we can get the k least frequent elements in O(n) rather than sorting
# since the largest number of items we will remove is k, we don't need all n items to be in the heap
# Since, K <= N, NlogK will always be more than or equal to KlogN
def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
    freq, res = defaultdict(int), []

    # gather number frequencies - O(n)
    for num in arr:
        freq[num] += 1
    
    # convert frequency dict to list of the k most infrequent elements in O(n)
    

    # heapify the first k items of the list - O(k)
    hq.heapify(n)

    # heappop the items with the lowest frequency for the k items being removed
    # O(klogk)
    while k > 0:
        smallest = hq.heappop(n)
        if smallest[0] <= k:
            k -= smallest[0]
            uniques -= 1
        else:
            break
    return uniques

# NOT FINISHED IMPLEMENTING
# also couldn't we do this in O(n) somehow - put into dict of freq: val pairs and then remove number of items that many times?
# ^ yes we can see below
# see https://www.google.com/search?client=safari&sca_esv=a2b78866b809fff6&rls=en&sxsrf=ACQVn0_hVw3LtI08E70Cz3Ux0Jk7Cnor-Q:1708121466761&q=O(n)+vs+%22klogn%22+OR+%22klog(n)%22&nfpr=1&sa=X&ved=2ahUKEwistImq8LCEAxV4kIkEHYKWAC4QvgUoAXoECAgQAw&biw=1314&bih=751&dpr=2
# TC: O(n), SC: O(n)
# beats 90%
# approach: 
#   1. create dict of {number: frequency} key value pairs 
#   2. convert this dict to {frequency: [list, of nums, with, this freq]} key/value pairs
#   3. iterate from 1 to n for all possible frequencieies (from low to high), removing elements with lowest frequencies if elements exist with said possible frequency
def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
    num_freq, frequencies = Counter(arr), defaultdict(list)
    uniques = len(num_freq)

    # add all nums to the frequency: [nums, with, this, frequency] dict
    for num in num_freq:
        frequencies[num_freq[num]].append(num)
    
    # remove nums iterating up from 1 to n, for all possible frequencies
    f = 1
    while f < 10**5:
        if f in frequencies:
            for _ in range(len(frequencies[f])):
                if k < f:
                    return uniques
                else:
                    k-=f
                    uniques -= 1
        f+=1
    return uniques

# Furthest Building You Can Reach LeetCode Medium
# https://leetcode.com/problems/furthest-building-you-can-reach/submissions/1178418076/?envType=daily-question&envId=2024-02-17
# took like 35
# greedy and heap solution
# TC: O(nlogn) or O(nlogb) SC: O(n)
# rudest heap question I've seen
# SHORTENED SOLUTION BELOW
def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
    if len(heights) == 1 or (not ladders and heights[1] - heights[0] > bricks): return 0
    first_n_buildings = []
    
    # get as far as we can using only bricks and saving the buildings we traversed in a list - O(n)
    i = 1
    while i < len(heights):
        if heights[i] - heights[i-1] > bricks: break
        if heights[i] > heights[i-1]:
            first_n_buildings.append(-(heights[i] - heights[i-1]))
            bricks -= (heights[i] - heights[i-1])
        i += 1

    # max-heapify this lift - this is O(n) but the largest size of the list is bricks, so this is O(bricks)
    hq.heapify(first_n_buildings)

    # continuously use our ladders on the greatest build now (greatest in the heap or the next item if it is greater)
    # O(nlogn) - because for every n in the list of heights (worst case) we will either heap-push or heap-pop, which is an O(logn) operation for a heap of max size n
    # note that this is not  O(nlog(bricks)) because although at first the max heap size is bricks, it can become larger than bricks
    # wait nvm it cannot become larger than the value of bricks so is it O(nlog(bricks)) ?
    while i < len(heights):
        if heights[i] <= heights[i-1]:
            i+= 1
            continue
        if heights[i] - heights[i-1] <= bricks:
            bricks -= heights[i] - heights[i-1]
            hq.heappush(first_n_buildings, -(heights[i] - heights[i-1]))
            i += 1
            continue
        if ladders == 0: break
        if first_n_buildings:
            largest = -(first_n_buildings[0])
            if largest > (heights[i] - heights[i-1]):
                hq.heappop(first_n_buildings)
                bricks += largest - (heights[i] - heights[i-1])
                hq.heappush(first_n_buildings, -(heights[i] - heights[i-1]))
        ladders -= 1
        i += 1

    return i-1

# Furthest Building You Can Reach LeetCode Medium (Above)
# realized we can just cut out the entire first part and simply start with an empty heap
def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
    if len(heights) == 1 or (not ladders and heights[1] - heights[0] > bricks): return 0
    prev_transactions, i = [], 1

    # 1. use any bricks we have if we can use them on the next building
    # 2. continuously use our ladders on the greatest building gap (greatest gap the heap or gap to the next item if it is greater) 
    while i < len(heights):
        if heights[i] <= heights[i-1]: # if the next building is lower it is free
            i += 1
            continue
        if heights[i] - heights[i-1] <= bricks: # use our bricks to get to the next building if we can
            bricks -= heights[i] - heights[i-1]
            hq.heappush(prev_transactions, -(heights[i] - heights[i-1]))
            i += 1
            continue
        
        # if we cannot use any bricks, we must use ladders on the gaps that costed the most bricks
        if ladders == 0: break # if we have no ladders we're done
        if prev_transactions: # if we have not spent any gaps on buildings yet, we must use the ladder on the next building (we cannot exchange)
            largest = -(prev_transactions[0]) 
            if largest > (heights[i] - heights[i-1]): # compare the largest transaction to the next building and use our ladder on whichever is more costly
                hq.heappop(prev_transactions)
                bricks += largest - (heights[i] - heights[i-1]) # if we're using a ladder on a previous transaction we get the bricks back
                hq.heappush(prev_transactions, -(heights[i] - heights[i-1]))
        ladders -= 1
        i += 1 # move to the next building
    return i-1

# https://leetcode.com/problems/meeting-rooms-iii/description/?envType=daily-question&envId=2024-02-18
# stuck on 79/82 testcases and I have no idea why
# but bad idea to iterate over values of time
def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
    
    # get all start values into a array of start_times[start] = end
    start_times = {}
    for meeting in meetings:
        start_times[meeting[0]] = meeting

    waiting = [] # minheap of meetings waiting (based on start time)
    available = [i for i in range(n)] # minheap of rooms available
    end_times = defaultdict(list) # dict of running meetings by endtime: rooms ending at that time
    room_counts = [0]*n # list of number of meetings that took place in each room
    hq.heapify(available)

    # iterate upwards in time
    for time in range(5*(10**5)+1):
        # if a meeting is waiting 
        while waiting and available:
            meeting = hq.heappop(waiting)
            room = hq.heappop(available)
            delay = time - meeting[0]
            end_times[meeting[1] + delay].append(room)

        # if a meeting is supposed to be starting now check for availability
        if time in start_times:
            meeting = start_times[time]
            if available:
                room = hq.heappop(available)
                end_times[meeting[1]].append(room)
            else:
                hq.heappush(waiting, meeting)
        
        # if a meeting is ending at this time
        if time+1 in end_times:
            for room in end_times[time+1]:
                hq.heappush(available, room) # add room to available again
                room_counts[room] += 1
            end_times[time+1] = []
    
    # return the room with the largest number of meetings
    greatest, rooms = 0, []
    for i in range(len(room_counts)):
        if room_counts[i] > greatest:
            greatest = room_counts[i]
            rooms = [i]
        elif room_counts[i] == greatest:
            rooms.append(i) 
    return min(rooms)

# solution - two heaps
# Meeting Rooms III LeetCode Hard
# todo come back to this ans review it
def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
    booked, free = [], list(range(n))
    meetings.sort()
    freq = defaultdict(int)

    for start, end in meetings:
        while booked and booked[0][0] <= start:
            _, room = heappop(booked)
            heappush(free, room)
        
        if free:
            room = heapq.heappop(free)
            heappush(booked, [end, room])
        else:
            release, room = heappop(booked)
            heappush(booked, [release + end - start, room])
        
        freq[room] += 1

    return max(freq, key=freq.get)

# Top K Frequent Elements LC Medium again ik
# best solution yet
# TC: O((n-k)logk) = O(nlogk) ??
# SC: O(n)
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    freq = Counter(nums)
    
    # split the items into the first k and then remaining items
    # TC O(n)
    first_k, remaining = [], []
    for idx, key in enumerate(freq.keys()):
        if idx < k:
            first_k.append((freq[key], key))
        else:
            remaining.append((freq[key], key))
    
    # heapify the first k items 
    # TC O(k)
    hq.heapify(first_k)

    # now continuously pushpop the remaining n-k items (this is a min heap so minimum items will be pushed)
    # TC O((n-k)logk)
    for item in remaining:
        hq.heappushpop(first_k, item)
    
    # then return the remaining k items in the heap (any order, just extract value)
    # TC O(k)
    return [item for _, item in first_k]