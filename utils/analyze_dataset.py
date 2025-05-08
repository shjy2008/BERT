
rate_to_count = {}

max_length = 0

# path = './data/sst-test.txt'
path = './data_ext/sst-train-ext-combine.txt'

with open(path) as f:
    while True:
        line = f.readline()
        if line:
            rate = int(line[0])
    
            if rate in rate_to_count:
                rate_to_count[rate] += 1
            else:
                rate_to_count[rate] = 1
            
            if len(line) > max_length:
                max_length = len(line)
        else:
            break

print(rate_to_count)
total = sum(rate_to_count.values())
print([rate_to_count[rate]/total for rate in rate_to_count])
print("max_length:", max_length)