class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
        
b = Node(3)
a = Node(2, b)
head = Node(1, a)

# say we want to save the current node and then increment current, we can do this
current = head # just pretending we are at some "current node" in the LL
saved = current # and say we want to save that node before moving current forward to the next node
current = current.next # then if we increment current, saved will not increment with it

print(current.data) # 2
print(saved.data) # 1