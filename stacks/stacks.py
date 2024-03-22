from collections import defaultdict

# Valid Parentheses LeetCode Easy
# https://leetcode.com/problems/valid-parentheses/submissions/1210371468/
# took 4 mins
def isValid(self, s: str) -> bool:
    opening = {")":"(", "}": "{", "]": "["}
    chars = []
    for c in s:
        if c == "[" or c == "(" or c == "{": # if opening, add to stack
            chars.append(c)
        elif not chars or opening[c] != chars[-1]: # elif closing and the item at the top of the stack isn't the corresponding opening bracket for this bracket, invalid
            return False
        else:   # else, item at top of stack does match this closing brace and we can pop it
            chars.pop()
    return not bool(chars)


# 71. Simplify Path
# https://leetcode.com/problems/simplify-path/description/
# complicated description see link
# see shorter solution below
def simplifyPath(self, path):
        path = path + " "
        newPath = []
        word = ""

        # note we do "path + x" because we wait the loop to run one extra time to add the last characters to the path
        # we could also create a function and call the loop contents inside the loop and one more time after the loop but this minor work around seemed fine
        for index, char in enumerate(path):
            if char == "/" or index == len(path) - 1:
                # apply the word
                if word == "..":
                    newPath = newPath[:-1]
                elif word != "" and word != ".":
                    newPath.append(word)
                
                # reset word
                word = ""
            else:
                word += char

        
        # combine all word separated by slashes into final path string
        res = ""
        for word in newPath:
            res += "/" + word
        return res or "/"

# MUCH simpler solution to above
def simplifyPath(self, path):
    stack = []
    for elem in path.split("/"):
        if stack and elem == "..":
            stack.pop()
        elif elem not in [".", "", ".."]:
            stack.append(elem)
            
    return "/" + "/".join(stack)

# Given a string s, if a characters contains duplicats, remove the correct duplicate so that the string takes the least
# completely incorrect todo review the question and solution
# 
def removeDuplicateLetters(self, s: str) -> str:
    # sort the list
    s = sorted(s)

    # remove duplicates
    newLen = len(s)
    for i in range(len(s)):
        if i + 1 < newLen and s[i] == s[i+1]:
            s = s[0:i+1] + s[i+2: len(s)] if i + 2 < newLen else s[0:i+1]
            newLen -= 1
    return ''.join(s)

# Implement Queue using Two Stacks LeetCode Easy - problem of the day Jan 28th
# https://leetcode.com/problems/implement-queue-using-stacks/?envType=daily-question&envId=2024-01-29
# took like 7 mins cuz minor class issues
class MyQueue:

    def __init__(self):
        self.stack_1 = []
        self.stack_2 = []

    def push(self, x: int) -> None:
        self.stack_1.append(x)

    def pop(self) -> int:
        while len(self.stack_1) > 1:
            recent = self.stack_2.append(self.stack_1.pop())
        res = self.stack_1.pop()
        while self.stack_2:
            self.stack_1.append(self.stack_2.pop())
        return res

    def peek(self) -> int:
        while len(self.stack_1) > 1:
            recent = self.stack_2.append(self.stack_1.pop())
        res = self.stack_1[0]
        while self.stack_2:
            self.stack_1.append(self.stack_2.pop())
        return res

    def empty(self) -> bool:
        return not bool(self.stack_1)


# Daily Temperatures LeetCode Medium - Daily problem
# Requires Monotonic Stack
# https://leetcode.com/problems/daily-temperatures/description/?envType=daily-question&envId=2024-01-31
# TC: O(n), SC: O(n)
# took like 20 mins because didn't know what a monotonic stack was (and didn't want to do the O(n^2) solution
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    indices = defaultdict(list)
    for i in range(len(temperatures)):
        indices[temperatures[i]].append(i)
    mono_stack = []
    res = [0]*len(temperatures)
    
    for i in range(len(temperatures)):
        if not mono_stack or mono_stack[-1] > temperatures[i]:
            mono_stack.append(temperatures[i])
        else:
            while mono_stack and mono_stack[-1] < temperatures[i]:
                val = mono_stack.pop()
                idx = indices[val][0]
                indices[val] = indices[val][1:]
                res[idx] = i - idx
            mono_stack.append(temperatures[i])
    return res

# Min Stack LeetCode Medium
# https://leetcode.com/problems/min-stack/
# took like 15 mins. Thought about monotonic stack but pop would be O(n), and realized you could do prefix-max second stack solution
# approach: maintain a second stack, largest_before, in which we keep track of the largest value *before* each element in the primary stack, so that if that item is the largest and is popped, we know the largest value before that item
# TC: O(1) for all operations, SC: O(n)
class MinStack:

    def __init__(self):
        self.minimum = []
        self.stack = [] 
        self.largest_before = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.largest_before.append(min(val, self.largest_before[-1] if self.largest_before else float('inf')))

    def pop(self) -> None:
        self.stack.pop()
        self.largest_before.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.largest_before[-1]

# Evaluate Reverse Polish Notation LeetCode Medium
# https://leetcode.com/problems/evaluate-reverse-polish-notation/description/
# TC: O(n), SC: O(n)
# took 3:45
def evalRPN(self, tokens: List[str]) -> int:
    s = []
    for t in tokens:
        if t in ["*", "-", "/", "+"]:
            second = int(s.pop())
            first = int(s.pop())
            if t == "*":
                s.append(first*second)
            elif t == "-":
                s.append(first-second)
            elif t == "+":
                s.append(first+second)
            else: # t == "/":
                s.append(first/second)
        else:
            s.append(t)
    return int(s[0])

# Daily Temperatures LeetCode Medium
# https://leetcode.com/problems/daily-temperatures/
# TC: O(n), SC: O(n)
# took 7:25
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    res = [0]*len(temperatures)
    stack = []  # stack of indexes of items
    for i in range(len(temperatures)):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res