from types import List

# Design Snake Game LeetCode Medium
# https://leetcode.com/problems/design-snake-game/submissions/1252198125/
# TC: O(1) for all operations but spawnNextFood is O(n)
# note this question has ambiguities and not well documented + limited test cases
class SnakeGame:

    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.grid = [[0]*width for _ in range(height)]
        self.score = 0
        self.path = collections.deque([[0,0]])
        self.occupied = set() # provides O(1) lookup time so we don't have to traverse entire snake path every time 
        self.food = collections.deque(food)
        self.food_reminaing = len(food)

        # add initial food to the grid
        self.occupied.add((0,0))
        self.spawnNextFood()          

    def move(self, direction: str) -> int:
        rows, cols = len(self.grid), len(self.grid[0])
        # check next move, if invalid, end game
        head = self.path[-1]
        row, col = head[0], head[1]
        if direction == "R":
            col += 1
        elif direction == "L":
            col -= 1
        elif direction == "U":
            row -= 1
        else:# direction == "D":
            row += 1
        
        # validate the snake is still in the grid
        if row < 0 or row == rows or col < 0 or col == cols:
            return -1
        
        # valid the snake has not run into itself
        if self.grid[row][col] == 1 and (self.path[0][0] != row or self.path[0][1] != col):
            return -1

        # else if next move location is food, generate next food
        self.path.append([row, col])
        self.occupied.add((row, col))
        if self.grid[row][col] == 2:
            self.score += 1
            self.spawnNextFood()
        else:

            # remove the tail item
            tail = self.path.popleft()
            if tail[0] != row or tail[1] != col: # don't set the tail value to 0 again if we are oging back to the square the tail is on
                self.grid[tail[0]][tail[1]] = 0
                self.occupied.remove((tail[0], tail[1]))
        
        self.grid[row][col] = 1
        return self.score

    def spawnNextFood(self):
        # if all food is collected, end game
        if not self.food_reminaing:
            return

        # find next food item that isn't under the snake
        # or don't show next food at all
        i = 0
        while i < len(self.food) and (not self.food[i] or tuple(self.food[i]) in self.occupied):
            i += 1
        if i < len(self.food) and self.food[i] and tuple(self.food[i]) not in self.occupied:
            self.food_reminaing -= 1
            self.grid[self.food[i][0]][self.food[i][1]] = 2
            self.food[i] = None

# Moving Average from Data Stream LeetCode Easy
# https://leetcode.com/problems/moving-average-from-data-stream/description/
# uses chaining to handle collisions
# much more interesting problem
# we group keys in into hashkeys by dividing keys by 100 and putting them into a linked list at that index of the array
# and then maintaining that linked list of nodes (keys and values) as we add, remove and update them
# SC comes out to O(10,000), where the maximum number of collitions is 100
# so SC is 100 times smaller for possible TC of O(100) for worset case operations (still O(n) and SC: O(1))
class Node:

    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None

# when doing this question I didn't realize keys were all integers, so was so confused how we would possible implement this
# but since keys are ints we can simply use an array
class MyHashMap:

    def __init__(self):
        self.map = [None]*(10000+1) # max 100 collisions (100*10,000 = 10^6 which is limit)

    def put(self, key: int, value: int) -> None:
        hashKey = key//100
        # iterate through nodes with mapped to that key and look for node with exact key
        if self.map[hashKey] != None:
            node = self.map[hashKey]
            while node.key != key and node.next:
                node = node.next

            # now we have either found the node or gone through all nodes mapped to this key, so either update the node value or create the node
            if node.key == key:
                node.val = value
            else:
                node.next = Node(key, value)
        else:
            self.map[hashKey] = Node(key, value)

    def get(self, key: int) -> int:
        hashKey = key//100
        # iterate across keys mapped to the key and return the value associated with the specific key
        if self.map[hashKey] != None:
            node = self.map[hashKey]
            while node.key != key and node.next:
                node = node.next
            if node.key == key:
                return node.val
        return -1

    def remove(self, key: int) -> None:
        hashKey = key//100
        # iterate across keys mapped to the key and delete the node and update the relationship
        if self.map[hashKey] != None:
            node = self.map[hashKey]

            # check if first node is the item
            if node.key == key:
                self.map[hashKey] = node.next

            # else traverse forward until item may be found
            while node.next and node.next.key != key:
                node = node.next
            if node.next and node.next.key == key:
                node.next = node.next.next
                # old node.next is garbage collected  

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

# Design Bitset LeetCode Medium
# https://leetcode.com/problems/design-bitset/description/
# see system requirements^
# All operations TC: O(1)
# SC: O(n)
# the challenging part is having to implement the flip() function in O(1) time, by using a global 'flipped' flag to represent
# whether all of the actual bit values are flipped
# in doing this, we also need to meticulously organize getters and setters (see getBits() and setBits()) for the bit values to fetch the correct
# values corresponding to thwe 'flipped' state and update the values corresponding to the 'flipped' state
# we can think of the logic in the getters/setters as an interface to the correct system/class state
class Bitset:
    def __init__(self, size: int):
        self.bits = [0]*size        
        self.flipped = False
        self.ones_count = 0

    def getBit(self, idx):
        return int(self.bits[idx] and not self.flipped or not self.bits[idx] and self.flipped)

    def setBit(self, idx, val):
        if idx < len(self.bits) and self.getBit(idx) != val:
            new_val = int(not self.bits[idx])
            self.bits[idx] = new_val
            self.ones_count += 1 if self.getBit(idx) else -1

    def fix(self, idx: int) -> None:
        self.setBit(idx, 1)

    def unfix(self, idx: int) -> None:
        self.setBit(idx, 0)

    def flip(self) -> None:
        self.flipped = not self.flipped
        self.ones_count = len(self.bits) - self.ones_count

    def all(self) -> bool: return self.ones_count == len(self.bits)  
    def one(self) -> bool: return self.ones_count > 0        
    def count(self) -> int: return self.ones_count
    def toString(self) -> str: return "".join([str(self.getBit(idx)) for idx in range(len(self.bits))])

# Design Twitter LeetCode Medium
# https://leetcode.com/problems/design-twitter/description/
# see system requirements^
# TC: 
# getNewsFeed: O(10) = O(1)
# follow: O(n+m) where n is followee tweets length and m is follower tweets length becuase we have to add all of the followee's tweets to the follower's feed in chronological order
# unfollow: O(n+m) where n is followee tweets length and m is follower tweets length becuase we have to remove all of the followee's tweets from the follower's feed
# postTweet: O(n) where n is the length of the poster's followers, since we need to add the tweet to the end of all of their follower's feeds
# I decided to create supplemental User and Tweet classes
# took ~30 mins and 40 after debugging edge cases
# edge cases to be aware of: users following/unfollowing themselves, users following/unfollowing users they already follow / don't follow, not yet existing users creating tweets
class Tweet:

    def __init__(self, id, timestamp, creator):
        self.id = id
        self.timestamp = timestamp
        self.creator = creator

class User:
    
    def __init__(self, id):
        self.following = set()
        self.followers = set()
        self.tweets = []
        self.feed = []
        self.id = id

    def postTweet(self, tweetId, timestamp):
        tweet = Tweet(tweetId, timestamp, self.id)
        self.tweets.append(tweet)
        return tweet

class Twitter:

    def __init__(self):
        self.users = {}
        self.time = 0

    # gets a user from the self.users dict or creates one (issues with defaultdict(User) and initializing it with an id)
    def getUser(self, userId):
        if userId not in self.users:
            user = User(userId)
            self.users[userId] = user
        return self.users[userId]

    def postTweet(self, userId: int, tweetId: int) -> None:
        user = self.getUser(userId)
        tweet = user.postTweet(tweetId, self.time)

        # add the tweet to all of the user's followers feeds
        for followerId in user.followers:
            follower = self.getUser(followerId)
            follower.feed.append(tweet)

        # add the tweet to the user's own feed
        user.feed.append(tweet)

        self.time += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        user = self.getUser(userId)
        if len(user.feed) > 10:
            return reversed([tweet.id for tweet in user.feed[-10:]])
        else:
            return reversed([tweet.id for tweet in user.feed])

    def follow(self, followerId: int, followeeId: int) -> None:
        follower, followee = self.getUser(followerId), self.getUser(followeeId)
        if followerId == followeeId or followeeId in follower.following: return

        # update following / followers
        follower.following.add(followeeId)
        followee.followers.add(followerId)
        
        # add all followee tweets to follower feed
        new_feed = []
        followee_tweet_no = 0
        for tweet in follower.feed:
            while followee_tweet_no < len(followee.tweets) and followee.tweets[followee_tweet_no].timestamp < tweet.timestamp:
                new_feed.append(followee.tweets[followee_tweet_no])
                followee_tweet_no += 1
            new_feed.append(tweet)
        if followee_tweet_no < len(followee.tweets):
            new_feed.extend(followee.tweets[followee_tweet_no:])

        # update the tweets
        follower.feed = new_feed

    def unfollow(self, followerId: int, followeeId: int) -> None:
        follower, followee = self.getUser(followerId), self.getUser(followeeId)
        if followerId == followeeId or followeeId not in follower.following: return

        # update following / followers
        follower.following.remove(followeeId)
        followee.followers.remove(followerId)

        # remove all of the (ex) followee's tweets from the unfollower's feed
        new_feed = []
        for tweet in follower.feed:
            if tweet.creator != followeeId:
                new_feed.append(tweet)
        follower.feed = new_feed


class Data:

    def __init__(self, key, val):
        self.val = val
        self.key = key # need the key so it can delete itself
        self.next = None
        self.prev = None

# LRU Cache LeetCode Medium
# https://leetcode.com/problems/lru-cache/description/
# Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
# : Implement the LRUCache class:
# : LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
# : int get(int key) Return the value of the key if the key exists, otherwise return -1.
# : void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
# : The functions get and put must each run in O(1) average time complexity.
# TC: O(1) for all operations and O(n) where n is capacity
# so many edge cases and had issues because all of the relationships in the DLL
# NOTE this question should be redone by having simple logic as follows:
# define a method for adding a new node to the end, as the new head
# on GET if the item exists, delete it and re-add it to the end
# on PUT if the item exists, delete it and re-add it to the end with the updated value
# a simpler approach
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys = {}
        self.head = None
        self.tail = None

    def get(self, key: int) -> int:
        res = None
        if key in self.keys:
            self.moveToTop(key)
            res = self.keys[key].val
        else:
            res = -1

        return res

    def moveToTop(self, key):
        if self.head == self.keys[key]: return

        # if the item is the tail, update the tail to the next item
        if self.keys[key] == self.tail and self.tail.next:
            self.tail.next.prev = None
            self.tail = self.tail.next
        
        # update the item before this item point at this items next item
        # and update the item after this item to point at the previous item
        if self.keys[key].prev:
            self.keys[key].prev.next = self.keys[key].next
            self.keys[key].next.prev = self.keys[key].prev

        # point old head to this node and set the node to the head
        self.head.next = self.keys[key]   
        self.keys[key].prev = self.head
        self.head = self.keys[key]
        self.head.next = None

    def put(self, key: int, value: int) -> None:

        if key in self.keys:
            self.keys[key].val = value
            self.moveToTop(key)
        else:
            # create a new item, point the head to it and set it to the head
            self.keys[key] = Data(key, value)
            if self.head:
                self.head.next = self.keys[key]
                self.keys[key].prev = self.head
            self.head = self.keys[key]

            # update the self.tail
            if len(self.keys) == 1:
                self.tail = self.head
            elif len(self.keys) > self.capacity:
                temp = self.tail.next

                # 'remove' the data
                del self.keys[self.tail.key]
                del self.tail
                self.tail = temp
                self.tail.prev = None

# LFU Cache LeetCode Hard
# https://leetcode.com/problems/lfu-cache/description/
# took over an hour becuase edge case hell and issues with DLLs
# LRU vs LFU: given the higher expected chance of requesting a certain element in the future, we should not evict it even if it was least recently used.
# approach: maintain a hash map of keys (similar to LRU cache) but this problem is more difficult as we are not always adding/popping to end front/end
# here we also keep a second dict 'frequencies' each containing it's own doubly linked lists from which we add and remove items from add we get/put them and their frequency/use changes
# this question took a minute. But used good separation of methods and tried to re-use code
# had a bunch of issues with maintaining the DLLs in each of the frequency values. Also note that these DLLs loop around so from the tail of the DLL of a certain frequency we can also easily get the head (most recently used item of that frequency)
# I also made a terrible assumption that if cache is full and no nodes with freq = 1 then you can never add a new node. My assumption here was that basically you alsways * add the new item * and the pop the least frequently used
# which in certain cases could end up always being the new item. This is not the case. If the cache is at capcity first we pop the LFU item and add the new one
# also had to be careful with checking "if node.next" to check if the node was the head. often node.next is itself (if single node and wrapping around). same goes for node.prev
# we also have to maintain the 'LFU' item (variable min_freq) at all times, so when a new node is added it will always be set to 1 and when a node in the min_frequency group is promoted to a higher frequency, if no more nodes exist in the old frequency, we increase the minimum as well
class Node:

    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = self
        self.prev = self
        self.freq = 1

class LFUCache:

    def __init__(self, capacity: int):
        self.keys = {}
        self.capacity = capacity
        self.frequencies = {}
        self.count = 0
        self.min_freq = 1

    # removes a key from it's frequency list
    def __remove(self, node):
        # delete the node from the dict
        del self.keys[node.key]
        self.count -= 1

        # if it is the tail remove it and delete the frequency entry if needed
        if node == self.frequencies[node.freq]:
            if node.next != node:
                self.frequencies[node.freq] = node.next
            else:
                del self.frequencies[node.freq]
        
        # update relationships
        # this can be moved into a function in node called node.deleteSelf() but fine here
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        node.prev = node
        node.next = node

    # add the item to the end of it it's corresponding frequency list
    # the frequency list could be empty or not
    def __add(self, node):
        self.keys[node.key] = node
        if node.freq in self.frequencies:
            self.frequencies[node.freq].prev.next = node
            old_head = self.frequencies[node.freq].prev
            self.frequencies[node.freq].prev = node
            node.prev = old_head
            node.next = self.frequencies[node.freq]
        else:
            self.frequencies[node.freq] = node
            # note.prev and node.next are self by default
        self.count += 1

    def increaseNodeFreq(self, node):
        # update lists
        self.__remove(node)

        # update min frequency if there are no more nodes in that frequency
        if node.freq not in self.frequencies and self.min_freq == node.freq:
            self.min_freq += 1

        # update the node
        node.freq += 1
        self.__add(node)
    
    # check if the value is in the dict, if so return it
    # and increment it's frequency (and update lists)
    def get(self, key: int) -> int:

        if key in self.keys:
            self.increaseNodeFreq(self.keys[key])
            return self.keys[key].val
        else:
            return -1

    # check if the value is in the dict, if so update it
    # and increment it's frequency (and update the lists)
    def put(self, key: int, value: int) -> None:

        if key in self.keys:
            self.increaseNodeFreq(self.keys[key])
            self.keys[key].val = value
        else:
            # if at capacity remove the node with the lowest frequency
            if self.count == self.capacity:
                self.__remove(self.frequencies[self.min_freq])

            # add the new node
            new_node = Node(key, value)
            self.__add(new_node)
            self.min_freq = 1 # min frequency will now be 1



# Max Stack LeetCode Hard
# https://leetcode.com/problems/max-stack/
# design/implement a stack that allows you to maintain a stack you can push, pop, peek (as usual)
# but you can also peekMax and popMax, which peeks and pops the maximum item in the stack. NOTE that a major constraing is O(logn) for all operations
# this question is freaky and not much documentation / communicty solutions, many people creating solutions with sub O(logn) operation time
# but came across Python SortedList data structure that lets you basically have a list that is sorted that you can add/remove items from in O(logn) time and it doesn't need to be shifted, causing O(n) worst case
# it leverages trees under the hood. Very similar to heap but with heap (for this question) you cannot remove a specific index (without difficult logic)
# :
# also see notes in dsa.txt about the SortedList
# below are other notes about this question I had as I was thinking
# considered:
# 1.
# linked list representing stack and then doing binary search to find/remove max
# but realized you can't do BS on a LL
# :
# 2.
# LL stack + heap so we could use the heap to get the max at any time and pop the max at any time (heap val would have pointer to stack LL node) while also getting top value in the stack
# but realized if we pop from the stack we cannot remove that item value from the heap
# at this point binary search is out of the picture because an array would take O(n) time for shifting
# and a LL cannot be indexed
# so has to be something heap related
# used a SortedList but apparently you can also find a way to remove items from a heap by keeping track of the indices of items and
# every time any item is moved around, update it index in the index tracker (such as a dict), so that you can pop ANY item from the heap (not just min/max)
# in O(logn) time
# this would have worked but would probably have to build heap from scratch
# so used SortedList which provides same functionality as list that remains sorted but requires no shift when you add
# TC: O(logn) for all operations except top() which is O(1)
# also see this solution https://leetcode.com/problems/max-stack/solutions/309621/python-using-stack-heap-set-with-explanation-and-discussion-of-performance/
from sortedcontainers import SortedList
class Node:
    
    def __init__(self, val, prev):
        self.val = val
        self.prev = prev
        self.next = None

class MaxStack:

    def __init__(self):
        self.head = None
        self.sortedItems = SortedList(key=lambda x: x.val)

    def push(self, x: int) -> None:
        node = Node(x, self.head)
        if self.head:
            self.head.next = node
        self.head = node

        # add item to the sorted list with pointer to original node
        self.sortedItems.add(node)

    def pop(self) -> int:
        node = self.head
        self.head = node.prev

        # remove the item from the sorted list
        self.sortedItems.remove(node)
        return node.val

    def top(self) -> int:
        return self.head.val

    def peekMax(self) -> int:
        return self.sortedItems[-1].val

    def popMax(self) -> int:
        maxx = self.sortedItems.pop(-1)
        if maxx.prev:
            maxx.prev.next = maxx.next
        if maxx.next:
            maxx.next.prev = maxx.prev
        if maxx == self.head:
            self.head = maxx.prev
        return maxx.val
            # contest notes
            # put obstacle using binary search
            # solution would be to do a sorted list but every time we add an item to thelist we maintaint the distances between all items (and the max distance between two items)
            # if we add a new item we update the distances corresponding the removing the distance between the previous left <--> right by addingthe new itrm
            # then when we query to place a block we cna just determine the max distance in O(1)
            # but here we would have to use 2 heaps or somt or build sorted heap from scratch
            # NVM ALSO sortedlist add is O(nlogn) so merked