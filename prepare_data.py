import os
import pandas as pd

def load_imdb_data(data_dir):
    data = {'review': [], 'sentiment': []}
    for sentiment in ['pos', 'neg']:
        sentiment_dir = os.path.join(data_dir, sentiment)
        for fname in os.listdir(sentiment_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(sentiment_dir, fname), 'r', encoding='utf-8') as f:
                    data['review'].append(f.read())
                    data['sentiment'].append(1 if sentiment == 'pos' else 0)
    return pd.DataFrame(data)

# Load training data
train_data = load_imdb_data("C:\\Users\\sabya\\Desktop\\project\\Sentiment Analysis of Movie Reviews\\aclImdb\\train")
train_data.to_csv('imdb_train.csv', index=False)

# Load test data
test_data = load_imdb_data("C:\\Users\\sabya\\Desktop\\project\\Sentiment Analysis of Movie Reviews\\aclImdb\\test")
test_data.to_csv('imdb_test.csv', index=False)

print("Data preparation is complete.")
