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
    print(areOneAway(test0_1, test0_2)) # false

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