# Leetcode Roman to Integer Easy
# https://leetcode.com/problems/roman-to-integer/description/
# Given a roman numeral, convert it to an integer.
def romanToInt(self, s: str) -> int:

        def convert(c):
            if c == "I":
                return 1
            elif c == "V":
                return 5
            elif c == "X":
                return 10
            elif c == "L":
                return 50
            elif c == "C":
                return 100
            elif c == "D":
                return 500
            elif c == "M":
                return 1000

        res = 0
        i = 0
        while i < len(s):
            if i+1 < len(s) and s[i] == "I" and s[i+1] == "V":
                res += 4
                i += 2
            elif i+1 < len(s) and s[i] == "I" and s[i+1] == "X":
                res += 9
                i += 2
            elif i+1 < len(s) and s[i] == "X" and s[i+1] == "L":
                res += 40
                i += 2
            elif i+1 < len(s) and s[i] == "X" and s[i+1] == "C":
                res += 90
                i += 2
            elif i+1 < len(s) and s[i] == "C" and s[i+1] == "D":
                res += 400
                i += 2
            elif i+1 < len(s) and s[i] == "C" and s[i+1] == "M":
                res += 900
                i += 2
            else:
                res += convert(s[i])
                i += 1

        return res