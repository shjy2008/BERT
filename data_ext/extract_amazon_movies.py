import json

read_file = './data_ext/Movies_and_TV_5.json' # Source: https://amazon-reviews-2023.github.io/ Movies_and_TV
write_file = './data_ext/sst-train-ext1.txt'

max_extract_data = 10000
num_labels = 5
each_label_data_count = max_extract_data / num_labels

rate_to_count = {}

def is_done():
    for i in range(num_labels):
        if rate_to_count.get(i, 0) < each_label_data_count:
            return False
    return True

with open(read_file, 'r') as f_dataset:
    with open(write_file, 'w') as f_new:
        while True:
            line = f_dataset.readline()
            d = json.loads(line)
            rate = int(d['overall']) - 1
            text = d.get('reviewText')
            if not text:
                continue
            
            if len(text) > 300: # Because sst-test.txt all < 300 (max 263)
                continue

            # Ensure each label the same number of samples
            if rate_to_count.get(rate, 0) >= each_label_data_count:
                continue

            text = text.replace('\n', ' ')
            new_line = f'{rate} ||| {text}\n'
            f_new.write(new_line)

            if rate in rate_to_count:
                rate_to_count[rate] += 1
            else:
                rate_to_count[rate] = 1
            
            if is_done():
                break
            
            

print(rate_to_count)
total = sum(rate_to_count.values())
print([rate_to_count[rate]/total for rate in rate_to_count])


