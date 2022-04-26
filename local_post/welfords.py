# WELFORDS ONLINE ALGORITHM FOR CALCULATING VARIANCE
# adapted from wiki page

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def _update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def _finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)


def update_agg(agg_list,sol_list):
    for i in range(len(agg_list)):
        agg_list[i] = _update(agg_list[i],sol_list[i])
    return agg_list

def finalize(agg_list):
    for i in range(len(agg_list)):
        agg_list[i] = _finalize(agg_list[i])
    return agg_list