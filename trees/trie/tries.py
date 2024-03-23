# Implement Trie (Prefix Tree) LeetCode Medium
# implement the Trie class: 
#   Trie() Initializes the trie object
#   insert(String word) Inserts the string word into the trie
#   search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
#   startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
# https://leetcode.com/problems/implement-trie-prefix-tree/description/
# TC: O(n) for all operations, SC: O(n)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        idx = 0
        while idx < len(word) and word[idx] in curr.children:
            curr = curr.children[word[idx]]
            idx += 1
        while idx < len(word):
            child = TrieNode()
            curr.children[word[idx]] = child
            curr = child
            idx += 1
        curr.endOfWord = True

    def search(self, word: str) -> bool:
        curr = self.root
        idx = 0
        while idx < len(word):
            if word[idx] not in curr.children:
                return False
            curr = curr.children[word[idx]]
            idx += 1
        return curr.endOfWord

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        idx = 0
        while idx < len(prefix):
            if prefix[idx] not in curr.children:
                return False
            curr = curr.children[prefix[idx]]
            idx += 1
        return True


# Design Add and Search Words Data Structure LeetCode Medium
# https://leetcode.com/problems/design-add-and-search-words-data-structure
# TC: O(n^2) for search and O(n) for add
# todo: how to do in linear time?
class DictNode:

    def __init__(self):
        self.children = {}
        self.endOfWord = False

class WordDictionary:

    def __init__(self):
        self.root = DictNode()

    def addWord(self, word: str) -> None:
        curr = self.root
        idx = 0
        while idx < len(word) and word[idx] in curr.children:
            curr = curr.children[word[idx]]
            idx += 1
        while idx < len(word):
            child = DictNode()
            curr.children[word[idx]] = child
            curr = child
            idx += 1
        curr.endOfWord = True     

    def search(self, word: str) -> bool:
        idx = 0
        curr = self.root
        while idx < len(word):
            if word[idx] == ".":
                for child in curr.children:
                    if self.search(word[:idx] + child + word[idx+1:]):
                        return True
                return False
            elif word[idx] in curr.children:
                curr = curr.children[word[idx]]
                idx += 1
            else:
                return False
        return curr.endOfWord

# Word Search II LeetCode Hard
# https://leetcode.com/problems/word-search-ii
# TC: O(nm4^nm) (see 4:50: https://youtu.be/asbcE9mZz_U?si=3Iuy-_ul8UB9uvQF&t=173) SC: O(w+mn)
# I initially implemented a search function inside of the Trie class, and planned on calling it with every new character in a path (see commented 'search' method), but realized
# this was highly inefficient and I basically needed to implement the Trie search/traversal inside of my dfs function (starting at the root)
# took like 50 mins without help
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    rows, cols = len(board), len(board[0])
    res = set()
    trie = Trie()
    memo = {}

    # build trie
    for word in words:
        trie.insert(word)

    def solve(visited, prefix, root, row, col):
        if board[row][col] in root.children:
            word = prefix + board[row][col]
            child = root.children[board[row][col]]
            if child.endOfWord:
                res.add(word)

            # try all neighbors
            visited.add((row, col))
            if row < rows - 1 and (row+1, col) not in visited:
                solve(visited, word, child, row+1, col)
            if row > 0 and (row-1, col) not in visited:
                solve(visited, word, child, row-1, col)
            if col < cols - 1 and (row, col+1) not in visited:
                solve(visited, word, child, row, col+1)
            if col > 0 and (row, col-1) not in visited:
                solve(visited, word, child, row, col-1)
            visited.remove((row, col))

    for row in range(rows):
        for col in range(cols):
            solve(set(), "", trie.root, row, col)
    return res

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        idx = 0
        curr = self.root
        while idx < len(word) and word[idx] in curr.children:
            curr = curr.children[word[idx]]
            idx += 1
        while idx < len(word):
            child = TrieNode()
            curr.children[word[idx]] = child
            curr = child
            idx += 1
        curr.endOfWord = True

    # def search(self, node):
    #     idx = 0
    #     curr = self.root
    #     while idx < len(word):
    #         if word[idx] in curr.children:
    #             curr = curr.children[word[idx]]
    #             idx += 1
    #         else:
    #             return None
    #     return curr

class TrieNode:

    def __init__(self):
        self.children = {}
        self.endOfWord = False