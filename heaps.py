import heapq as hq

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

# heap delete by index
# heap delete by value
# heap delete by value, multiple
# converts max to to a min heap
# merge to heaps

h = [1,2,3]
h_1 = [3,2,1]
h_2 = [5,4,3,2,1,0, 1, 1,1,1,0,0,0,-1,-2, 0] #
h_3 = [5,4,3,2,1,9]
min_1 = [0,1,2,3,4,6,7,9,9]

print(max_heap_del_by_val(h_2, 5))
print(h_2)