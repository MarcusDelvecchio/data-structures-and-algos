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

# Leetcode Integer to Roman Medium
# https://leetcode.com/problems/integer-to-roman/submissions/
# Given an integer convert it to a roman numeral.
# took 19 mins
# but there is a way simpler and hash map way to do it see https://leetcode.com/problems/integer-to-roman/solutions/6274/simple-solution/
def intToRoman(self, num: int) -> str:
        res = []

        # divide by 1000s to get number of Ms
        M = floor(num/1000)
        num -= M*1000
        res.extend(["M"]*M)

        CM = floor(num/900)
        num -= CM*900
        res.extend(["CM"]*CM)
        

        # divide by 500s to get number of Ds
        D = floor(num/500)
        num -= D*500
        res.extend(["D"]*D)

        CD = floor(num/400)
        num -= CD*400
        res.extend(["CD"]*CD)

        # divide by 100s to get number of Cs
        C = floor(num/100)
        num -= C*100
        res.extend(["C"]*C)

        XC = floor(num/90)
        num -= XC*90
        res.extend(["XC"]*XC)

        # divide by 50 to get number of Ls
        L = floor(num/50)
        num -= L*50
        res.extend(["L"]*L)

        XL = floor(num/40)
        num -= XL*40
        res.extend(["XL"]*XL)

        # divide by 10 to get number of Xs
        X = floor(num/10)
        num -= X*10
        res.extend(["X"]*X)

        IX = floor(num/9)
        num -= IX*9
        res.extend(["IX"]*IX)

        # divide by 5 to get number of Vs
        V = floor(num/5)
        num -= V*5
        res.extend(["V"]*V)

        IV = floor(num/4)
        num -= IV*4
        res.extend(["IV"]*IV)

        # divide by 1 to get number of Is
        I = floor(num/1)
        num -= I*1
        res.extend(["I"]*I)
    
        return ''.join(res)
