class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

# prints the contents of the provided linked list
def printList(head):
    res = ""
    current = head
    while(current):
        res += str(current.data)
        current = current.next
    
    print(res)


# function to remove all nodes with a certain value
def removeByValue(head, data):
    current = head

    while(True):
        next = current.next
        while(next.data == data):
            if(not next.next):
                current.next = None
                return head
            else:
                next = next.next
        current.next = next
        current = next

        if(not current.next):
            break
    return head

# 2.1 Remove Dups 
# function that removes all duplicates from a linked list
def removeDups(head):
    previous = head
    current = head.next
    # head.next = None ? 
    included = [head.data]

    while(True):
        if(included.count(current.data) == 0):
            included.append(current.data)
            previous.next = current
            previous = current
        
        if(not current.next):
            previous.next = None
            break
        else:
            current = current.next
    return head

# 2.2 remove kth to last
# find the kth last element of the singly linked list
def getKthLast(head, k):
    p1 = head
    p2 = head if k == 1 else head.next
    # if less than k elements in the array return null

    # move second pointer so it is k elements ahead of p1
    for i in range(0, k - 2):
        if(not p2.next):
            return None
        p2 = p2.next

    # then move both p1 and p2 forward until p2 hits then end of the list
    while(True):
        if(not p2.next):
            return p1
        else:
            p1 = p1.next
            p2 = p2.next

# 2.3 delete a given internal node
# given an internal node (node that isn't first or last) of a LL, delete said node without having access to the head
def deleteInternalNode(n):
    # copy data from next node to the current node
    n.data = n.next.data

    # delete next node but this doesn't need to be performed in Python

# 2.4 Partition based on value
# given a value in a LL, parition the LL so that all nodes with value less than the give one come before all nodes that have a value greater
# also note that the partition value itself can appear anywhere on the upper side of the LL; it doesn't need to appear between
def partitionOnVal(head, val):
    current = head
    lessStart = None
    moreStart = None

    lessCurrent = None
    moreCurrent = None

    # go through list and copy items into less and more lists
    while(True):
        if(current.data >= val):
            if(not moreStart):
                moreStart = current
            else:
                moreCurrent.next = current
            moreCurrent = current
        else:
            if(not lessStart):
                lessStart = current
            else:
                lessCurrent.next = current
            lessCurrent = current
        
        # if there are no more items, connect the two lists and return
        if(not current.next):
            # cut off end because the last item in more will still have a next (we never removed it)
            if(moreCurrent):
                moreCurrent.next = None

            # check first if there are even any lesser values at all
            if(lessStart):
                lessCurrent.next = moreStart
                return lessStart
            else:
                return moreStart
        else:
            current = current.next

# prints various solutions for problem 2.4
def displaySolution_2_4(head):
    for i in range(0, 10):
        print("partitionOnVal(head,", i, ") returns:")
        printList(partitionOnVal(head, i))

# 2.5 Sum Lists
# takes two linked lists that represents numbers, where the heads indicate the 1's digits and add the numbers, returning the sum as a linked list
def addLinkedListNumbers(head1, head2):
    resHead = None
    resCurrent = resHead
    resPrevious = None

    current1 = head1
    current2 = head2

    isCarry = False

    while(True):
        if(not current1 and not current2):
            return resHead
        elif not current1:
            resCurrent.next = current2.next
            return resHead
        elif not current2:
            resCurrent.next = current1.next
            return resHead
        else: # current1 and current2 exist
            resCurrent.data = Node((current1.data + current2.data)%10, resPrevious)
            resPrevious = resCurrent
            current1 = current1.next
            current2 = current2.next



fifth = Node(2)
fourth = Node(5, fifth)
third = Node(3, fourth)
second = Node(4, third)
first = Node(3, second)
head2 = Node(5, first)
head = Node(5, head2)
# print(first.next.data)
# removeByValue(head, 3)
# removeDups(head)
# kthLast = getKthLast(head, 3)
# print(kthLast.data)

# 2.4 testing
# printList(partitionOnVal(head, 1))

# uncomment this to display various solutions where we partition the LL on values from ranged 0 - 10
# displaySolution_2_4(head)



