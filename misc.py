# Divisor Game LeetCode Easy
# https://leetcode.com/problems/divisor-game/description/
# this question is a brain fork until you understand the maths behind it
# good explanation https://leetcode.com/problems/divisor-game/solutions/274727/python-dp/
# there is actually a DP way to solve this problem but I assume that
# it wouldn't be an easy problem if it was expected that you should do it that way
def divisorGame(self, n: int) -> bool:
    return n % 2 == 0