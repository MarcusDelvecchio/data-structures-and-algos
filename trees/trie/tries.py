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