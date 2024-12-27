# Step 1: Import necessary libraries
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the cleaned and processed data files
posts_df = pd.read_csv('/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_posts2.csv')
comments_df = pd.read_csv('/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_comments2.csv')

# Step 3: Combine tokens from posts and comments into a single list
posts_tokens_list = posts_df['title_tokens'].dropna().tolist() + posts_df['selftext_tokens'].dropna().tolist()
comments_tokens_list = comments_df['body_tokens'].dropna().tolist()
combined_tokens_list = posts_tokens_list + comments_tokens_list
all_tokens = [word for tokens in combined_tokens_list for word in eval(tokens)]

# Step 4: Generate N-grams (Bigrams, Trigrams, Fourgrams)
def get_ngrams(tokens, n=2):
    n_grams = ngrams(tokens, n)
    return Counter(n_grams)

bigrams = get_ngrams(all_tokens, 2).most_common(100)
trigrams = get_ngrams(all_tokens, 3).most_common(100)
fourgrams = get_ngrams(all_tokens, 4).most_common(100)

# Step 5: Prepare Sentiment Analysis on Posts Data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Flatten n-grams to a list of common phrases
common_ngrams = [' '.join(ngram) for ngram, _ in (bigrams + trigrams + fourgrams)]

# Step 6: Initialize sentiment counters for each n-gram
ngram_sentiment_counts = {ngram: {'positive': 0, 'neutral': 0, 'negative': 0} for ngram in common_ngrams}

# Step 7: Analyze sentiment for each post and count sentiment per n-gram
for _, row in posts_df.iterrows():
    text = row['selftext']
    if isinstance(text, str):
        text_lower = text.lower()
        sentiment_score = sia.polarity_scores(text)['compound']  # Analyze the sentiment of the full text
        for ngram in common_ngrams:
            if ngram in text_lower:
                # Count the sentiment based on the full text sentiment score
                if sentiment_score > 0.05:
                    ngram_sentiment_counts[ngram]['positive'] += 1
                elif sentiment_score < -0.05:
                    ngram_sentiment_counts[ngram]['negative'] += 1
                else:
                    ngram_sentiment_counts[ngram]['neutral'] += 1

# Step 8: Convert sentiment counts to DataFrame and save to CSV
ngram_sentiment_df = pd.DataFrame.from_dict(ngram_sentiment_counts, orient='index').reset_index()
ngram_sentiment_df.columns = ['ngram', 'positive_count', 'neutral_count', 'negative_count']

output_csv_path = '/Users/granthuston/Desktop/Reddit JSON/ngram_sentiment_analysis_counts.csv'
ngram_sentiment_df.to_csv(output_csv_path, index=False)

print(f"Sentiment analysis results saved to {output_csv_path}")

# Step 9: Generate Heatmap for N-grams with at least 100 total mentions
def generate_ngram_heatmap(df, top_n=40):
    # Filter n-grams with at least 100 total combined counts
    df['total_count'] = df['positive_count'] + df['neutral_count'] + df['negative_count']
    filtered_df = df[df['total_count'] >= 100]

    # Sort and limit to top N n-grams based on total count
    filtered_df = filtered_df.sort_values(by='total_count', ascending=False).head(top_n)

    if filtered_df.empty:
        print("No n-grams found with at least 100 total mentions.")
        return

    # Calculate negative ratio for sorting purposes in visualization
    filtered_df['negative_ratio'] = filtered_df['negative_count'] / filtered_df['total_count']
    filtered_df = filtered_df.sort_values(by='negative_ratio', ascending=False)

    # Prepare data for the heatmap
    filtered_df = filtered_df[['ngram', 'negative_count', 'neutral_count', 'positive_count']].set_index('ngram')
    filtered_df_ratio = filtered_df.div(filtered_df.sum(axis=1), axis=0)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(filtered_df_ratio, annot=False, cmap='viridis', fmt='.2f', cbar=True)
    plt.title('Sentiment Distribution Ratios for Top 40 N-grams with Highest Counts')
    plt.xlabel('Sentiment Type')
    plt.ylabel('N-gram')
    plt.xticks(rotation=45)

    # Add counts and ratios as annotations with adaptive text color
    for i in range(filtered_df_ratio.shape[0]):
        for j in range(filtered_df_ratio.shape[1]):
            ngram = filtered_df_ratio.index[i]
            sentiment = filtered_df_ratio.columns[j].replace('_count', '')
            ratio = filtered_df_ratio.iloc[i, j]
            count = filtered_df.iloc[i, j]
            text = f'{ratio:.2f} ({count})'
            color = 'white' if ratio < 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    plt.show()

# Step 10: Generate Heatmap for Top N N-grams
generate_ngram_heatmap(ngram_sentiment_df, top_n=40)


def generate_negative_ngram_heatmap(df, top_n=40):
    # Filter n-grams with at least 100 total combined counts
    df['total_count'] = df['positive_count'] + df['neutral_count'] + df['negative_count']
    filtered_df = df[df['total_count'] >= 100]

    # Calculate negative ratio and sort by highest negative ratio
    filtered_df['negative_ratio'] = filtered_df['negative_count'] / filtered_df['total_count']
    filtered_df = filtered_df.sort_values(by='negative_ratio', ascending=False).head(top_n)

    if filtered_df.empty:
        print("No n-grams found with at least 100 total mentions.")
        return

    # Prepare data for the heatmap
    filtered_df = filtered_df[['ngram', 'negative_count', 'neutral_count', 'positive_count']].set_index('ngram')
    filtered_df_ratio = filtered_df.div(filtered_df.sum(axis=1), axis=0)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(filtered_df_ratio, annot=False, cmap='viridis', fmt='.2f', cbar=True)
    plt.title('Sentiment Distribution Ratios for Top N N-grams with Highest Negative Ratio and at least 100 Total Mentions')
    plt.xlabel('Sentiment Type')
    plt.ylabel('N-gram')
    plt.xticks(rotation=45)

    # Add counts and ratios as annotations with adaptive text color
    for i in range(filtered_df_ratio.shape[0]):
        for j in range(filtered_df_ratio.shape[1]):
            ngram = filtered_df_ratio.index[i]
            sentiment = filtered_df_ratio.columns[j].replace('_count', '')
            ratio = filtered_df_ratio.iloc[i, j]
            count = filtered_df.iloc[i, j]
            text = f'{ratio:.2f} ({count})'
            color = 'white' if ratio < 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    plt.show()

# Step 10: Generate Heatmap for Top N N-grams based on highest negative ratio
generate_negative_ngram_heatmap(ngram_sentiment_df, top_n=40)
