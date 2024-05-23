# Design Parking System LeetCode Easy
# https://leetcode.com/problems/design-parking-system/description/
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.big = big
        self.medium = medium
        self.small = small  

    def addCar(self, carType: int) -> bool:
        if carType == 1 and self.big != 0:
            self.big -= 1
            return True
        elif carType == 2 and self.medium != 0:
            self.medium -= 1
            return True
        elif carType == 3 and self.small != 0:
            self.small -= 1
            return True
        return False

# Moving Average from Data Stream LeetCode Easy
# https://leetcode.com/problems/moving-average-from-data-stream/description/
# TC: O(1) for next
# TC: O(n) where n = size (using a queue and clearing old items boosted SC from beats 9% to beats 60%)
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.items = collections.deque([]) # allows for O(size) space rather than O(n) space where n it the total number of items inserted over time

    # adds the provided value to the list of items and returns the new average (and removes an item if it goes over size)
    def next(self, val: int) -> float:
        self.items.append(val)

        # pop if we have gone over our size limit
        if len(self.items) > self.size:
            self.items.popleft()

        total = sum(self.items)
        return total/len(self.items)

# this is a cheap solution that uses O(10^6) space by default - see better solution below
# when doing this question I didn't realize keys were all integers, so was so confused how we would possible implement this
# but since keys are ints we can simply use an array
class MyHashMap:

    def __init__(self):
        self.map = [None]*((10**6)+1)

    def put(self, key: int, value: int) -> None:
        if self.map[key] != None:
            self.map[key] = value
        else:
            self.map[key] = value

    def get(self, key: int) -> int:
        if self.map[key] != None:
            return self.map[key]
        else:
            return -1

    def remove(self, key: int) -> None:
        if self.map[key] != None:
            self.map[key] = None


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

# Logger Rate Limiter LeetCode Easy
# https://leetcode.com/problems/logger-rate-limiter/description/
# TC: O(1), SC: O(n) where n is the total number of operations
# this design never deletes old message timestamps
class Logger:

    def __init__(self):
        self.recent_prints = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message in self.recent_prints and self.recent_prints[message] + 10 > timestamp:
            return False
        self.recent_prints[message] = timestamp
        return True