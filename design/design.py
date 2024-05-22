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