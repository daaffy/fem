from aggs import *
import numpy as np

# # ----- example: using an agg object
# agg_1 = agg() # initialise

# x = 1
# agg_1.update(x)
# print(agg_1.curr) # notice that agg_1 has updated

# x = 2
# agg_1.update(x)

# agg_1.finalise()
# print(agg_1.statistics) # notice e.g., the average has been correctly calculated: (1 + 2)*0.5 = 1.5



# # ----- test: make sure agg statistics are calculated correctly
# n = 1000 # sample size
# random_data = np.random.rand(n,1)

# agg_1 = agg()
# for x in random_data:
#     agg_1.update(x)
# agg_1.finalise()

# # the statistics below should be equal within some small error
# print([[agg_1.statistics[0], np.average(random_data)],
#         [agg_1.statistics[1], np.var(random_data)]])


# # ----- test: aggs class
# aggs_inst = aggs(2)

# n = 10000
# for i in range(n):
#     new_data = (np.random.rand(2,1),2*np.random.rand(2,1))
#     aggs_inst.update(new_data)

# aggs_inst.finalise()
# agg_1 = aggs_inst[0]
# agg_2 = aggs_inst[1]
# print(agg_1.statistics)
# print(agg_2.statistics)
# print(agg_1)

# ----- test: agg combine
agg_1 = agg()
agg_2 = agg()

n = 5000
dat1 = np.random.rand(n,1)
dat2 = np.random.rand(n,1)

for i in range(n):
    agg_1.update(dat1[i])
    agg_2.update(dat2[i])

# agg_1.finalise()
# agg_2.finalise()

agg_1.combine(agg_2)
agg_1.finalise()

print(np.mean([dat1, dat2]))
print(agg_1.statistics[0])

print(np.var([dat1, dat2]))
print(agg_1.statistics[1])
