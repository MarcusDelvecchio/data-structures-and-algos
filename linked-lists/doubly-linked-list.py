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