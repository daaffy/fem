import random
import numpy as np
import matplotlib.pyplot as plt

def plot(eval_points,U):
    # ---- PLOTTING
    N = 5000
    idx=random.sample(range(np.size(eval_points,1)),N) # random sample of evaluation points; take it easy on the graphics

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(eval_points[0,idx],eval_points[1,idx],U[idx],marker=".")
    plt.show()