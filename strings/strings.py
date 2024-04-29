
# Maximum Nesting Depth of the Parentheses LeetCode Easy
# https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/
# TC: O(n), SC: O(1)
# took 1 minute 45 seconds
def maxDepth(self, s: str) -> int:
    open_braces = depth = 0
    for c in s:
        if c == "(":
            open_braces += 1
        elif c == ")":
            open_braces -= 1
        depth = max(depth, open_braces)
    return depth

# Make The String Great LeetCode Medium
# https://leetcode.com/problems/make-the-string-great/description/
# TC: O(n), SC:O(n)
# took like 5 mins but then realized I should use stack then took 70 seconds
def makeGood(self, s: str) -> str:
    res = []
    for i in range(len(s)):
        if res and res[-1] != s[i] and res[-1].lower() == s[i].lower():
            res.pop()
        else:
            res.append(s[i])
    return "".join(res)

# this is so easy don't even read
# Leetcode daily Jan 12th - Determine if String Halves Are Alike Easy
# https://leetcode.com/problems/determine-if-string-halves-are-alike/description/?envType=daily-question&envId=2024-01-12
# TC: O(n) SC: O(n)(we can also do a 2ptr method for O(1) but idc)
def halvesAreAlike(self, s: str) -> bool:
    a, b = s[:len(s)//2], s[len(s)//2:]
    count_a, count_b = 0,0

    for i in range(len(a)):
        if a[i] in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
            count_a += 1
        if b[i] in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
            count_b += 1
    return count_a == count_b

# Make Three Strings Equal LeetCode Easy
# https://leetcode.com/problems/make-three-strings-equal/
# took 8 mins
# TC: O(str1+str2+str3) = O(n), SC: O(1)
def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
    # idea: they should all be cut to the same length and compared as we cut them down
    # approach: while not the same, loop. If all same length and not equal, we must pop from all. If some are larger, pop form them
    ops, s1, s2, s3 = 0, list(s1), list(s2), list(s3)
    while s1 and s2 and s3 and (s1 != s2 or s1 != s3):
        if len(s1) == len(s2) and len(s1) == len(s3): # if all equal pop from all
            s1.pop()
            s2.pop()
            s3.pop()
            ops += 3
        else:
            maxx = max(len(s1), len(s2), len(s3))
            for s in [s1, s2, s3]:
                if len(s) == maxx:
                    s.pop()
                    ops += 1
    return ops if s1 and s2 and s3 else -1

# Isomorphic Strings LeetCode Easy
# https://leetcode.com/problems/isomorphic-strings
# actually took like 8 mins
# TC: O(n), SC: O(n)
def isIsomorphic(self, s: str, t: str) -> bool:
    s_counts = collections.Counter(s)
    t_counts = collections.Counter(t)
    mappings = {}

    # ensure all letters are consistiently mapped to the same letter
    for i in range(len(t)):
        # validate mapping matches if one already exists and validate frequency of the chars at the same idx are the same
        if t[i] in mappings and s[i] != mappings[t[i]] or t_counts[t[i]] != s_counts[s[i]]:
            return False
        else: # if mapping doesn't already exist add one
            mappings[t[i]] = s[i]
    return True

# Length of Last Word LeetCode Easy
# https://leetcode.com/problems/length-of-last-word/description/
# Given a string s consisting of words and spaces, return the length of the last word in the string. A word is a maximal substring consisting of non-space characters only.
# TC: O(n), SC: O(1)
# took 1 minute
def lengthOfLastWord(self, s: str) -> int:
    prev, length = 0, 0
    for c in s:
        if c == " ":
            if length:
                prev = length
            length = 0
        else:
            length += 1
    return length or prev

# EASY
# Is Subsequence Easy
# Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
# https://leetcode.com/problems/is-subsequence/description/
# took like 5
# TC: O(n), SC: O(1)
def isSubsequence(self, s: str, t: str) -> bool:
    if not s: return True
    s_pointer = 0
    for c in t:
        if c == s[s_pointer]:
            s_pointer += 1
            if s_pointer == len(s):
                return True
    return False


# given two strings, returns true if they are one edit (insert, remove, replace) away
# untested because 
def areOneAway(str1, str2):
    remainingEdits = 1

    # if more than 1 character difference return - 1
    lengthDif = len(str1) - len(str2)
    if abs(lengthDif) > 1:
        return False

    index1 = 0
    index2 = 0
    while True:
        if index1 == len(str1) and index2 == len(str2):
            return True
        if (index1 == len(str1) or index2 == len(str2)):
            if remainingEdits:
                return True
            else:
                return False
        if str1[index1] == str2[index2]:
            index1 += 1
            index2 += 1
        elif remainingEdits > 0:
            if str1[index1 + 1] == str2[index2]:
                index1 += 1
                remainingEdits -= 1
            elif str1[index1] == str2[index2 + 1]:
                index2 += 1
                remainingEdits -= 1
            elif str1[index1 + 1] == str2[index2 + 1]:
                index2 += 1
                index1 += 1
                remainingEdits -= 1
        else:
            return False
            


# test are one away function
def areOneAway_test():
    test0_1 = "the same string"
    test0_2 = "the same string"
    # print(areOneAway(test0_1, test0_2)) # false

    test1_1 = "abcdef"
    test1_2 = "abcdxf"
    # print(areOneAway(test1_1, test1_2)) # true

    test2_1 = "xbcdef"
    test2_2 = "abcdxf"
    # print(areOneAway(test2_1, test2_2)) # false

    test3_1 = "abcd"
    test3_2 = "abcde"
    # print(areOneAway(test3_1, test3_2)) # true

    test4_1 = "abcdde"
    test4_2 = "abcdef"
    # print(areOneAway(test4_1, test4_2)) # false
    
# Leetcode Determine is two strings are close question
# https://leetcode.com/problems/determine-if-two-strings-are-close/
# this question is basically asking: can the strings be convertted so that they have the same number of each character? same number of a's same number of b's same number of c's etc
# and if so then operation 2 (index swapping chars) can simply be applied to attain the same string
def closeStrings(self, word1: str, word2: str) -> bool:
    # NOTE THAT THIS IMPLEMENTATION IS INEFFICIENT
    # although it works, this can easily be solved by doing the following:
        # 1. check that string lengths are the same
        # 2. check that the number of various different characters are the same
        # 3. check that neither string contains a 'frequency' of a character that the other doesn't
        # see here https://youtu.be/0Nt8t75dFl0?si=XVrqsbRKL1u9ucgo&t=128

    if len(word1) != len(word2):
            return False

    # collect the occurances of each char in both strings - creates dict with key: char, value: occurances of those chars
    charsWord1 = {}
    charsWord2 = {}
    for i in range(len(word1)):
        charsWord1[word1[i]] = charsWord1[word1[i]] + 1 if word1[i] in charsWord1 else 0
        charsWord2[word2[i]] = charsWord2[word2[i]] + 1 if word2[i] in charsWord2 else 0

    keysWord1 = list(charsWord1.keys())
    keysWord2 = list(charsWord2.keys())

    # verify same number of different chars
    if len(keysWord1) != len(keysWord2):
        return False

    # determine which chars need to be swapped for which
    word1Char, word2Char = None, None
    inWord1notWord2, inWord2notWord1 = [], []
    for i in keysWord1:
        if not keysWord2.count(i):
            return False

        # else if it is in word2, check that the same amount are
        elif charsWord1[i] != charsWord2[i]:
            found = False
            for key, value in charsWord2.items():
                if charsWord2[key] == charsWord1[i]:
                    found = True
                    break
            if not found:
                return False
        
    # find what char exists in word2 that doesn't exist in word1
    for j in keysWord2:
        if not keysWord1.count(j):
            inWord2notWord1.append(j)

        # else if it is in word2, check that the same amount are
        elif charsWord2[j] != charsWord1[j]:
            found = False
            for key, value in charsWord1.items():
                if charsWord1[key] == charsWord2[j]:
                    found = True
                    break
            if not found:
                return False

    # do we even need this
    if len(inWord2notWord1) or len(inWord1notWord2):
        return False

    index = 0
    while index < len(keysWord1):
        word1key = keysWord1[index]
        found = False
        for key in list(charsWord2.keys()):
            if charsWord1[word1key] == charsWord2[key]:
                charsWord2.pop(key)
                index += 1
                found = True
                break
        if not found:
            return False
    return True



areOneAway_test()

# LeetCode Daily Jan 13th - Minimum Number of Steps to Make Two Strings Anagram Medium
# took just over 20 because I had the wrong approach
# https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/?envType=daily-question&envId=2024-01-13
# TC: O(n) SC: O(n)
def minSteps(self, s: str, t: str) -> int:
    s_chars, t_chars = Counter(s), Counter(t)
    excess = 0

    # count the excess characters in s that AREN'T in t
    # of the characters that there are more of in s than t
    for char in s_chars.keys():
        if char not in t_chars:
            excess += s_chars[char]
        elif s_chars[char] > t_chars[char]:
            excess += s_chars[char] - t_chars[char]
    return excess

# Sort Characters By Frequency LeetCode Medium
# https://leetcode.com/problems/sort-characters-by-frequency/description/
# TC: O(n^2) becuase appending strings is inefficient. SC: O(n)
# took like 6 mins because iterating and sorting across the counter items was a cheese
from collections import Counter
class Solution:
    def frequencySort(self, s: str) -> str:
        counts, res = Counter(s), ""
        for item in sorted([(key, counts[key]) for key in counts], key=lambda x: x[1], reverse=True):
            res += item[0]*item[1]
        return res

# https://leetcode.com/problems/custom-sort-string/?envType=daily-question&envId=2024-03-11
# Custom Sort String LeetCode Medium
# Took 7 mins
# TC: O(nlogn) (sorting) SC: O(1)
def customSortString(self, order: str, s: str) -> str:
    order = {order[i]: i for i in range(len(order))}
    s = sorted(list(s), key=lambda x: order[x] if x in order else -1)
    return "".join(s)

# String Compression LeetCode Medium
# TC: O(n), SC: O(n)
# https://leetcode.com/problems/string-compression/description/
# compress a list of characters by groupsing characters into char + count of items in group like: [a,a,a,b,c,c,c] => a3bc3 (3 a's, 1 b, 3 c's)
def compress(self, chars: List[str]) -> int:
    ans = []
    idx = 0
    while idx < len(chars):
        count = 1
        next_idx = idx + 1
        while next_idx < len(chars) and chars[next_idx] == chars[idx]:
            next_idx += 1
            count += 1
        if count == 1:
            ans.append(chars[idx])
        else:
            ans.append(chars[idx] + str(count))
        idx = next_idx

    curr = 0
    for part in ans:
        for c in part:
            chars[curr] = c
            curr += 1
    chars = chars[:curr]
    return curr

# Count and Say LeetCode Medium
# https://leetcode.com/problems/count-and-say/description/
# TC: O(2^n), SC: O(n)
def countAndSay(self, n: int) -> str:
    curr = "1"
    for _ in range(n-1):
        idx = 0
        ans = [] # the next sequence
        while idx < len(curr):
            next_idx = idx + 1
            count = 1
            while next_idx < len(curr) and curr[next_idx] == curr[idx]:
                next_idx += 1
                count += 1
            ans.append(str(count) + curr[idx])
            idx = next_idx
        curr = "".join(ans)
    return curr