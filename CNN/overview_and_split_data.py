import sys
import random
import pandas as pd



def shuffle_split(train_size, test_size, filename, output1, output2, output3, header_needed, preprocess):
    data = open(filename, "r", encoding='latin-1')
    train1 = open(output1, "w", encoding='latin-1')
    test1 = open(output2, "w", encoding='latin-1')
    valid1 = open(output3, "w", encoding='latin-1')
    if header_needed:
        if preprocess:
            first_line_unused = data.readline().replace('<br /><br />', '')
        read_data = data.readlines()
        random.shuffle(read_data)
        length_train = (len(read_data) * train_size) / 100
        length_test = (len(read_data) * test_size) / 100
        b = []
        c = []
        d = []
    else:
        read_data = data.readlines()
        random.shuffle(read_data)
        length_train = (len(read_data) * train_size) / 100
        length_test = (len(read_data) * test_size) / 100
        b = []
        c = []
        d = []
    if preprocess:
        for x, y in enumerate(read_data):
            if x < length_train:
                b.append(y.replace('<br /><br />', ''))
            elif x < length_test:
                c.append(y.replace('<br /><br />', ''))
            else:
                d.append(y.replace('<br /><br />', ''))
    else:
        for x, y in enumerate(read_data):
            if x < length_train:
                b.append(y)
            elif x < length_test:
                c.append(y)
            else:
                d.append(y)

    train1.writelines(b)
    test1.writelines(c)
    valid1.writelines(d)

    train1.close()
    test1.close()
    valid1.close()
    data.close()


def classify_binary(filename, output1, output2, header_needed, preprocess, threshold=50001):
    data = open(filename, "r", encoding='latin-1')
    pos = open(output1, "w", encoding='latin-1')
    neg = open(output2, "w", encoding='latin-1')
    if header_needed:
        if preprocess:
            first_line = data.readline().replace('<br /><br />', '')
        read_data = data.readlines()
        print(len(read_data))
        pos_arr = []
        neg_arr = []
    else:
        read_data = data.readlines()
        pos_arr = []
        neg_arr = []
    count = 0
    if preprocess:
        for line in read_data:
            count += 1
            if count >= threshold:
                break
            cur_line = line.strip().rsplit(',', 1)
            if cur_line[1] == 'positive':
                pos_arr.append(cur_line[0].replace('<br /><br />', ''))
            elif cur_line[1] == 'negative':
                neg_arr.append(cur_line[0].replace('<br /><br />', ''))
    else:
        for line in read_data:
            count += 1
            if count >= threshold:
                break
            cur_line = line.strip().rsplit(',', 1)
            if cur_line[1] == 'positive':
                pos_arr.append(cur_line[0])
            elif cur_line[1] == 'negative':
                neg_arr.append(cur_line[0])

    print(len(pos_arr), len(neg_arr))
    pos.writelines([l + '\n' for l in pos_arr])
    neg.writelines([l + '\n' for l in neg_arr])

    pos.close()
    neg.close()
    data.close()

def classify_ternary(filename, output1, output2, output3, header_needed, preprocess, threshold=50001):
    data = open(filename, "r", encoding='latin-1')
    pos = open(output1, "w", encoding='latin-1')
    neu = open(output2, "w", encoding='latin-1')
    neg = open(output3, "w", encoding='latin-1')
    if header_needed:
        if preprocess:
            first_line = data.readline().replace('<br /><br />', '')
        read_data = data.readlines()
        print(len(read_data))
        pos_arr = []
        neu_arr = []
        neg_arr = []
    else:
        read_data = data.readlines()
        pos_arr = []
        neu_arr = []
        neg_arr = []
    count = 0
    if preprocess:
        for line in read_data:
            count += 1
            if count >= threshold:
                break
            cur_line = line.strip().split(',', 1)
            if cur_line[0] == 'positive':
                pos_arr.append(cur_line[1].replace('<br /><br />', ''))
            elif cur_line[0] == 'neutral':
                neu_arr.append(cur_line[1].replace('<br /><br />', ''))
            elif cur_line[0] == 'negative':
                neg_arr.append(cur_line[1].replace('<br /><br />', ''))
    else:
        for line in read_data:
            count += 1
            if count >= threshold:
                break
            cur_line = line.strip().split(',', 1)
            if cur_line[0] == 'positive':
                pos_arr.append(cur_line[1])
            if cur_line[0] == 'neutral':
                neu_arr.append(cur_line[1])
            elif cur_line[0] == 'negative':
                neg_arr.append(cur_line[1])

    print(len(pos_arr), len(neu_arr), len(neg_arr))
    pos.writelines([l + '\n' for l in pos_arr])
    neu.writelines([l + '\n' for l in neu_arr])
    neg.writelines([l + '\n' for l in neg_arr])

    pos.close()
    neg.close()
    data.close()


if __name__ == '__main__':
    data = pd.read_csv('IMDB Dataset.csv')
    data['length'] = data['review'].apply(len)
    data['words'] = [len(x.split(" ")) for x in data['review']]

    print("------------IMDB DATASET-------------\n")
    print('Average Number of Characters:\n', data['length'].mean())
    print('Average Number of Words:\n', data['words'].mean())
    print('Size:\n', data['sentiment'].count())
    print('Positive Count\n', data[data['sentiment'] == 'positive']['sentiment'].count())
    print('Negative Count\n', data[data['sentiment'] == 'negative']['sentiment'].count())

    print("------------Financial Sentiment DATASET-------------\n")
    data = pd.read_csv('financial-sentiment.csv', encoding='latin-1', names = ['sentiment', 'review'])
    data['length'] = data['review'].apply(len)
    data['words'] = [len(x.split(" ")) for x in data['review']]
    print('Average Number of Characters:\n', data['length'].mean())
    print('Average Number of Words:\n', data['words'].mean())
    print('Size:\n', data['sentiment'].count())
    print('Positive Count\n', data[data['sentiment'] == 'positive']['sentiment'].count())
    print('Negative Count\n', data[data['sentiment'] == 'negative']['sentiment'].count())
    print('Neutral Count\n', data[data['sentiment'] == 'neutral']['sentiment'].count())
    
    '''
    shuffle_split(70, 90, "IMDB Dataset.csv", "IMDB - P - Train.csv",
                  "IMDB - P - Test.csv", "IMDB - P - Validate.csv", True, True)
    '''
    if len(sys.argv) == 2:
        print("Trying to split original dataset")
        if sys.argv[1] == 'IMDB Dataset.csv':
            shuffle_split(70, 90, "IMDB Dataset.csv", "IMDB - P - Train.csv",
                  "IMDB - P - Test.csv", "IMDB - P - Validate.csv", True, True)
            classify_binary("IMDB Dataset.csv", "IMDB.pos",
                    "IMDB.neg", True, True)
        elif sys.argv[1] == 'financial-sentiment.csv':
            shuffle_split(70, 90, "financial-sentiment.csv", "financial-sentiment - Train.csv",
                  "financial-sentiment - Validate.csv", "financial-sentiment - Test.csv", True, True)
            classify_ternary("financial-sentiment.csv", "financial.pos", "financial.neu",
                    "financial.neg", True, True)