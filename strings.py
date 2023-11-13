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
    if len(word1) != len(word2):
        return False

    # collect the occurances of each char in both strings
    charsWord1 = {}
    charsWord2 = {}
    for i in range(len(word1)):
        occurancesWord1 = charsWord1[word1[i]]
        occurancesWord2 = charsWord2[word2[i]]
        charsWord1[word1[i]] = occurancesWord1 += 1 if occurancesWord1 else 0
        charsWord1[word2[i]] = occurancesWord2 += 1 if occurancesWord2 else 0

    # loop through occurances
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
            inWord1notWord2.append(i)
                
    # find what char exists in word2 that doesn't exist in word1
    for j in keysWord2:
        if not keysWord1.count(j):
            inWord2notWord1.append(j)

    # do we even need this
    if len(inWord2notWord1) != len(inWord1notWord2):
        return False

    for excessWord1 in inWord1notWord2:
        found = False
        for excessWord2 in inWord2notWord1:
            if charsWord1[excessWord1] == charsWord2[excessWord2]:
                word1 = word1.replace(word1Char, word2Char)
                found = True
                break
        if not found:
            return False
    return True



areOneAway_test()