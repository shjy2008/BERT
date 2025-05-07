import csv

read_file = './data_ext/rotten_tomatoes_critic_reviews.csv' # Source: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
write_file = './data_ext/sst-train-ext3.txt'

max_extract_data = 45000
num_labels = 5
each_label_data_count = max_extract_data / num_labels

rate_to_count = {}

def is_done():
    for i in range(num_labels):
        if rate_to_count.get(i, 0) < each_label_data_count:
            return False
    return True

count = 0
with open(read_file, 'r') as f_dataset:
    with open(write_file, 'w') as f_new:
        reader = csv.DictReader(f_dataset)
        for row in reader:
            rate_str = row['review_score']
            text = row['review_content']

            if not text:
                continue

            if '/' not in rate_str:
                continue

            score, max_score = rate_str.split('/')
            try:
                score = float(score)
            except (ValueError, TypeError):
                continue
            try:
                max_score = float(max_score)
            except (ValueError, TypeError):
                continue

            if score > max_score:
                continue

            rate = int(round(score / max_score * 4))
            
            # Ensure each label the same number of samples
            if rate_to_count.get(rate, 0) >= each_label_data_count:
                continue

            text = text.replace('\n', ' ')
            f_new.write(f'{rate} ||| {text}\n')

            if rate in rate_to_count:
                rate_to_count[rate] += 1
            else:
                rate_to_count[rate] = 1
        
            if is_done():
                break
            
            

print(rate_to_count)
total = sum(rate_to_count.values())
print([rate_to_count[rate]/total for rate in rate_to_count])
