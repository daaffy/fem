# from multiprocessing import Pool

# def f(x):
#     return x*x

# if __name__ == '__main__':
#     with Pool(5) as p:
#         print(p.map(f, [1, 2, 3]))

# ---------------------------------------

import multiprocessing
import time
 
def task():
    print('Sleeping...')
    time.sleep(3)
    print('Finished sleeping')
 
if __name__ == "__main__":
    start_time = time.perf_counter()
    processes = []
    
    # create multiple processes then start them
    for i in range(10):
        p = multiprocessing.Process(target=task)
        p.start()
        processes.append(p)

    # join all processes
    for p in processes:
        p.join()

    finish_time = time.perf_counter()
 
    print(f"Program finished in {finish_time-start_time} seconds")
