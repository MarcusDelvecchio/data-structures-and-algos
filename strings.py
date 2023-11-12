# given two strings, returns true if they are one edit (insert, remove, replace) away
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



    


areOneAway_test()