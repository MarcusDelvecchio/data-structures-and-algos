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
def closeStrings(self, word1: str, word2: str) -> bool:
    if len(word1) != len(word2):
        return False

    index = 0
    remainingSingleSwaps = 1
    swaps = {} # dictionary in the format 'word1Char':'word2Char' when two chars are incompatible
    while index < len(word1):
        if word1[index] != word2[index]:
            keys = list(swap.keys())
            if len(keys):
                if keys[0] == word1[index] and swap[keys[0]] == word2[index]:
                    continue
                elif keys[0] == word2[index] and swap[keys[0]] == word1[index] and remainingSingleSwaps:
                    remainingSingleSwaps -= 1
                    continue
                else:
                    return False
            # no keys added to swaps yet
            else:
                swaps[word1[index]] = word2[index]
        else:
            index += 1



areOneAway_test()