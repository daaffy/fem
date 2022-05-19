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


# ----- test: aggs class
aggs_inst = aggs(2)

n = 10000
for i in range(n):
    new_data = (np.random.rand(2,1),2*np.random.rand(2,1))
    aggs_inst.update(new_data)

aggs_inst.finalise()
agg_1 = aggs_inst[0]
agg_2 = aggs_inst[1]
print(agg_1.statistics)
print(agg_2.statistics)
