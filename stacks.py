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