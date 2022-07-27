import torch

# random values from uniform distribution [0,1]
t_unif = torch.rand(3)
print("uniform tensor: ", t_unif)

# random values from standard normal distribution mu=0, std=1
t_st_norm = torch.randn(3)
print("standard normal tensor: ", t_st_norm)

# random integer values from uniform distribution between low and high
t_int = torch.randint(low=0, high=10, size=(3,))
print("int tensor: ", t_int)

# random permutation of integers from 0 to n-1
t_perm = torch.randperm(3)
print("permutation tensor: ", t_perm)

# random binary numbers 0 or 1 from Bernoulli distribution
t_bern = torch.bernoulli(torch.tensor([0.99, 0.01, 0.5]))
print("bernoulli tensor: ", t_bern)

# random numbers from multinomial distribution
t_mult = torch.multinomial(torch.tensor([0.99, 0.99, 0.99]), 2)
print("multinomial tensor: ", t_mult)

t_like = torch.randint_like(t_int, 10)
print(t_like)