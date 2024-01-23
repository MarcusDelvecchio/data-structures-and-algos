def minimum_coins(m, coins):
    memo = {}
    memo[0] = 0
    for i in range(1, m + 1): 
        for coin in coins:
            subproblem = i - coin
            if subproblem < 0:
                continue
                
            memo[i] = min_ignore_none(memo.get(i), memo.get(subproblem) + 1)
return memo[m]