class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
    
    def copy(self):
        return Node(self.data, self.next)

# same LL node structure but different name / property naming (Leetcoded uses this)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
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
    # note here ListNode was used instead of node
    # and ListNode.val === Node.data
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


# 2.6 Palindrome
# returns true if a given linked list is a Palindrome (symmetrical)
# works by finding the mid point, splitting into left and right sub LLs, and comparing left and right LL values
def isPalindrome(head):

    #  if only one item return true
    if head.data and not head.next:
        return True

    front = head
    back = head

    headRight = None
    # headLeft = head
    len = 0

    # split the LL into two LLs
    while(True):
        if front.next and front.next.next:
            len += 1
            back = back.next
            front = front.next.next
        else: # front is at the end or one before the end (even)
            if not front.next: # odd - easy
                headRight = back.next
            else: # even - not front.next.next but front.next exists
                headRight = back.next
                len += 1
            break
    
    printList(head)
    printList(headRight)
    print(len)
    # compare the left and right LLs. Compare the kth last element of the left with the 
    k = 1
    leftCurrent = head
    while(True):
        if not leftCurrent.data == getKthLast(headRight, k).data:
            return False
        if k == len:
            return True

        leftCurrent = leftCurrent.next
        k += 1
        
# 2.7 intersection
# given two linked lists, returns true if the two of them have an intersecting node
def haveIntersection(head1, head2):

    # convert first LL to an array of ids and compare with all items of the second list
    ids = []
    current = head1
    while(True):
        ids.append(id(current))

        if not current.next:
            break
        else:
            current = current.next

    # go through second LL and see if any nodes have the same id as those collected from the first
    current = head2
    while(True):
        if ids.count(id(current)):
            return True
        elif not current.next:
            return False
        else:
            current = current.next

# 2.8 Loop Detection
# given a linked lists, determines if the list contains a loop and returns the item at the beginning of the loop
def detectLoop(head):
    ids = []

    current = head
    while(True):
        if ids.count(id(current)):
            return current
        else:
            ids.append(id(current))

        if not current.next:
            return False # no loop
        else:
            current = current.next

# Leetcode Merge In Between Linked Lists
# takes a linked list, a start and end index, and a second LL and connects the start of the second LL to index a-1 and b+1
def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    current = list1
    currentl2 = list2

    currentIndex = 0
    tempCurrent = None
    while(True):
        if currentIndex == a - 1:
            tempCurrent = current
            current = current.next
            tempCurrent.next = list2
            currentIndex += 1
            continue
        elif currentIndex == b + 1:

            # go all the way to the end of l1
            while(True):
                if currentl2.next:
                    currentl2 = currentl2.next
                else:
                    break
            currentl2.next = current
            return list1

        current = current.next
        currentIndex += 1

# Leetcode Rotate List Medium Problem
# Given the head of a linked list, rotate the list to the right by k places.
# https://leetcode.com/problems/rotate-list/description/
def RotateList(head, k):
    if k == 0 or not head or not head.next:
            return head

    p1 = head
    p2 = head

    # mod k by length of the LL
    current = head
    len = 1
    while(True):
        if current.next:
            current = current.next
            len += 1
        else:
            break
    
    print(len)
    k = k%len if k >= len else k

    if k == 0:
        return head

    # move p2 k indices ahead of p1
    for i in range(k):
        p2 = p2.next

    # now move p1 and p2 until p2 hits the end
    while(True):
        if p2.next:
            p2 = p2.next
            p1 = p1.next
        else:
            break

    # p1 is the new end and p1.next is the new head
    newHead = p1.next
    p1.next = None

    # set p2.next to the old head (move p2 to front)
    p2.next = head

    return newHead

# reverses a linked list given it's head
def reverseLinkedList(head):
    previous = None
    next = None
    current = head
    
    while True:
        if previous:
            if not current.next:
                current.next = previous
                return current
            else:
                next = current.next
                current.next = previous
                previous = current
                current = next
        else:
            previous = current
            current = current.next
            previous.next = None

# cleaner way to reverse linked list
def reverseLinkedListShort(head):
    current = head
    previous = None
    next = head.next

    while current:
        current.next = previous
        previous = current
        current = next # current = current.next
        if next:
            next = next.next

    return current
# even SHORTER way to reverse a LL that I thought up later. I'm sure it's the most effective way
# remember there are FOUR primary steps (inside that while loop)
# consider the probelm from the 'perspective' of a single node at a time
def reverseLinkedListShort_2(head):
    current = head
    prev = None
    while current:
        temp = current.next
        current.next = prev
        prev = current
        current = temp

        if not current:
            return prev

    return current

# merge two sorted lists Leetcode problem
# https://leetcode.com/problems/merge-two-sorted-lists/
# given two sorted lists, merge the two sorted lists into one sorted list, returning ites head
def mergeTwoLists(self, list1: [ListNode], list2: [ListNode]) -> [ListNode]:

    # note that these two pointers can be removed and simply replaced wiuth 'list1' and list2'
    current1 = list1
    current2 = list2

    # check for empty list
    if not list1 or not list2:
        if not list1:
            return list2
        else:
            return list1

    currentNew = None
    newHead = None
    while current1 and current2:
        # get lesser value
        lesser = current1 if current1.val <= current2.val else current2

        # move that lesser node to our new list (leave its pointer pointiung back at whatever list it is from though)
        if newHead:
            currentNew.next = lesser # change the previous items pointer to this new item
        else:
            newHead = lesser
        currentNew = lesser

        # increment the current in the list that we took it from
        if current1.val <= current2.val:
            current1 = current1.next
        else:
            current2 = current2.next
    
    if not current1 and current2:
        currentNew.next = current2
    else: #if not current2 and current1:
        currentNew.next = current1

    return newHead

# Merge K Sorted Lists Leetcode problem
# https://leetcode.com/problems/merge-k-sorted-lists/submissions/
# Given a list of K ordered lists, merge them all into 1 list and return it's head
# Took about 20/30 mins to setup the logic and spent 30 mins thinking I was having Python ref isssues but in reality it was my logic. Took exactly 1 hour
def mergeKLists(self, lists):
    newHead = None
    current = None

    while True:
        # find the smallest node from all of the lists
        smallest = None
        index = 0
        selectedIndex = 0 # index of list that the nex node is selected from
        for list in lists:
            if (not smallest) and list:
                smallest = list
                selectedIndex = index
            elif list and list.val < smallest.val:
                smallest = list
                selectedIndex = index
            index += 1

        # if there is no smallest, then there must be no items left so we can break
        if not smallest:
            return newHead
        
        # add the smallest to the new array
        if current:
            current.next = smallest
            current = smallest
        else:
            newHead = smallest
            current = smallest
        
        # shift the list that the smallest was selected from
        if lists[selectedIndex]:
            lists[selectedIndex] = lists[selectedIndex].next
    
    return newHead

# Swap Pairs Leetcode Medium
# given a LL, swap every adjuscent pair of nodes (sap 1&2, 3&4, 5&6)
# https://leetcode.com/problems/swap-nodes-in-pairs/submissions/
# also see this solution, which is exactly the same solution https://leetcode.com/problems/swap-nodes-in-pairs/solutions/1774318/python3-i-hate-linked-lists-not-explained/
def swapPairs(self, head):
    current = head

    if not head or not head.next:
        return head

    newHead = head.next
    prev = None
    while current.next:
        temp = current.next.next
        current.next.next = current
        if prev:
            prev.next = current.next
        current.next = temp
        prev = current

        # increment current
        current = temp

        if not current or not current.next:
            return newHead
    
    return newHead

# alternative solution to the Swap Pairs solution above
# this seems to be twice as fast (see https://leetcode.com/problems/swap-nodes-in-pairs/submissions/)
# made this solution because I realized that you are really pointing the first node of a pair FOUR nodes ahead (because that 4th node ahead will swap with its pair and become the 3rd node ahead)
# so this solution solved the problem without using the 'prev' pointer, but deal with a lot of small edge case scenarios as well
def swapPairs_alternative(self, head):
        current = head

        if not head or not head.next:
            return head

        newHead = head.next
        while current.next:
            temp = current.next.next
            current.next.next = current

            # point current 4 forward
            if temp:
                if temp.next:
                    current.next = temp.next
                else:
                    current.next = temp
            else:
                current.next = None
            current = temp

            if not current or not current.next:
                return newHead
        
        return newHead

seventh = Node(9)
sixth = Node(7, seventh)
fifth = Node(7, sixth)
fourth = Node(5, fifth)
third = Node(3, fourth)
second = Node(20, third)
first = Node(3, second)
head3 = Node(5, first)
head2 = Node(1, head3)
head = Node(7, head2)

printList(reverseLinkedListShort_2(head))

# headSecondList = Node(5, second)
# print(haveIntersection(headSecondList, head))

# uncomment below two lined to test loop detection
# seventh.next = second
# print(detectLoop(head).data)

# print(first.next.data)
# removeByValue(head, 3)
# removeDups(head)
# kthLast = getKthLast(head, 2)
# print(kthLast.data)

# print(isPalindrome(head))

# 2.4 testing
# printList(partitionOnVal(head, 1))

# uncomment this to display various solutions where we partition the LL on values from ranged 0 - 10
# displaySolution_2_4(head)



