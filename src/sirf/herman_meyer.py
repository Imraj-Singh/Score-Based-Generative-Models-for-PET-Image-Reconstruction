import math

def herman_meyer_order(n):
    # Assuming that the subsets are in geometrical order
    n_variable = n
    i = 2
    factors = []
    while i * i <= n_variable:
        if n_variable % i:
            i += 1
        else:
            n_variable //= i
            factors.append(i)
    if n_variable > 1:
        factors.append(n_variable)
    n_factors = len(factors)
    order =  [0 for _ in range(n)]
    value = 0
    for factor_n in range(n_factors):
        n_rep_value = 0
        if factor_n == 0:
            n_change_value = 1
        else:
            n_change_value = math.prod(factors[:factor_n])
        for element in range(n):
            mapping = value
            n_rep_value += 1
            if n_rep_value >= n_change_value:
                value = value + 1
                n_rep_value = 0
            if value == factors[factor_n]:
                value = 0
            order[element] = order[element] + math.prod(factors[factor_n+1:]) * mapping
    return order
