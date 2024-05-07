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

# Design Twitter LeetCode Medium
# https://leetcode.com/problems/design-twitter/description/
# see system requirements^
# TC: 
# getNewsFeed: O(10) = O(1)
# follow: O(n+m) where n is followee tweets length and m is follower tweets length becuase we have to add all of the followee's tweets to the follower's feed in chronological order
# unfollow: O(n+m) where n is followee tweets length and m is follower tweets length becuase we have to remove all of the followee's tweets from the follower's feed
# postTweet: O(n) where n is the length of the poster's followers, since we need to add the tweet to the end of all of their follower's feeds
# I decided to create supplemental User and Tweet classes
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
