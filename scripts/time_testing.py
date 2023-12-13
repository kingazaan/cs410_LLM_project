import time

starttime = time.time()

for i in range(10000):
    print(i*2)

endtime = time.time()

total_time = endtime - starttime

print('time taken:')
print(total_time)
