
rate_to_count = {}

with open('./data/sst-train.txt') as f:
    while True:
        line = f.readline()
        if line:
            rate = int(line[0])
    
            if rate in rate_to_count:
                rate_to_count[rate] += 1
            else:
                rate_to_count[rate] = 1
        else:
            break

print(rate_to_count)
total = sum(rate_to_count.values())
print([rate_to_count[rate]/total for rate in rate_to_count])