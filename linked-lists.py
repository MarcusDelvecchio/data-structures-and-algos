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
# completed on Leetcode see https://leetcode.com/problems/add-two-numbers/submissions/
def addLinkedListNumbers(l1, l2):
    current1 = l1
    current2 = l2

    resHead = None
    resCurrent = None

    prev = None
    carry = 0
    while(True):
        if current1 and current2:
            value = (current1.val + current2.val + carry)
            carry = (1 if value >= 10 else 0)
            print("value: ", value, ", carry: ", carry)

            resCurrent = ListNode(value%10)
            if prev:
                prev.next = resCurrent
            prev = resCurrent
        elif not current1 and not current2:
            if carry:
                prev.next = ListNode(carry)
            return resHead
        elif not current1:
            prev.next = addLinkedListNumbers(self, current2, ListNode(carry)) if carry else current2
            return resHead
        else: # not current2:
            prev.next = addLinkedListNumbers(self, current1, ListNode(carry)) if carry else current1
            return resHead
        
        if not resHead:
            resHead = resCurrent

        current1 = current1.next
        current2 = current2.next

# 2.5b Sum Lists in Forward order
# takes two linked lists that represents numbers, where the tails indicate the 1's digits and add the numbers, returning the sum as a linked list
# does this by getting kth last of each list and then adding them every iteration
def addLinkedListNumbersForwardOrder(l1, l2):
    resHead = None
    resCurrent = None

    prev = None
    carry = 0
    currentDigit = 1 # start with "first last"
    while(True):
        if current1 and current2:
            # get the kth last of each of the LLs and add them
            value = getKthLast(l1, currentDigit) + getKthLast(l1, currentDigit) + carry
            carry = (1 if value >= 10 else 0)

            resCurrent = ListNode(value%10)
            if prev:
                prev.next = resCurrent
            prev = resCurrent
        elif not current1 and not current2:
            if carry:
                prev.next = ListNode(carry)
            return resHead
        elif not current1:
            prev.next = addLinkedListNumbers(self, current2, ListNode(carry)) if carry else current2
            return resHead
        else: # not current2:
            prev.next = addLinkedListNumbers(self, current1, ListNode(carry)) if carry else current1
            return resHead
        
        if not resHead:
            resHead = resCurrent

        currentDigit += 1





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



