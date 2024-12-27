# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.util import ngrams

# Ensure NLTK downloads are available
nltk.download('stopwords')

# Step 2: Load the cleaned and processed data files
posts_df = pd.read_csv('/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_posts2.csv')
comments_df = pd.read_csv('/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_comments2.csv')

# Step 3: Combine tokens from posts and comments into a single list
# Extract tokens from posts
posts_tokens_list = posts_df['title_tokens'].dropna().tolist() + posts_df['selftext_tokens'].dropna().tolist()
# Extract tokens from comments
comments_tokens_list = comments_df['body_tokens'].dropna().tolist()
# Combine tokens lists
combined_tokens_list = posts_tokens_list + comments_tokens_list
# Flatten tokens and convert string representations to lists
all_tokens = [word for tokens in combined_tokens_list for word in eval(tokens)]

# Step 4: Word Frequency Analysis
# Count word frequencies in the combined tokens
word_freq = Counter(all_tokens)
# Get the top 200 most common words
most_common_words = word_freq.most_common(200)

# Print the most common words
print("Most common words in the combined dataset:")
print(most_common_words)

# Step 5: Visualization - Word Cloud
# Generate word cloud for combined tokens
wordcloud = WordCloud(width=800, height=400, max_words=500, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('r/veteransbenefits posts and comments')
plt.show()

# Step 6: N-Gram Analysis
def get_ngrams(tokens, n=2):
    n_grams = ngrams(tokens, n)
    return Counter(n_grams)

# Get Bigrams
bigrams = get_ngrams(all_tokens, 2)
most_common_bigrams = bigrams.most_common(100)

# Get Trigrams
trigrams = get_ngrams(all_tokens, 3)
most_common_trigrams = trigrams.most_common(100)

# Get Four-Grams
fourgrams = get_ngrams(all_tokens, 4)
most_common_fourgrams = fourgrams.most_common(100)

# Step 7: Save Results to CSV
# Convert N-Gram Data to DataFrame for Saving
def save_ngrams_to_csv(most_common_ngrams, filename):
    ngram_list = [{'ngram': ' '.join(ngram), 'frequency': count} for ngram, count in most_common_ngrams]
    ngram_df = pd.DataFrame(ngram_list)
    ngram_df.to_csv(filename, index=False)

# Save N-Grams
save_ngrams_to_csv(most_common_bigrams, '/Users/granthuston/Desktop/Reddit JSON/most_common_bigrams.csv')
save_ngrams_to_csv(most_common_trigrams, '/Users/granthuston/Desktop/Reddit JSON/most_common_trigrams.csv')
save_ngrams_to_csv(most_common_fourgrams, '/Users/granthuston/Desktop/Reddit JSON/most_common_fourgrams.csv')

# Save most common words to CSV
pd.DataFrame(most_common_words, columns=['Word', 'Frequency']).to_csv('/Users/granthuston/Desktop/Reddit JSON/most_common_words.csv', index=False)

# Combine CSV paths for Excel output
csv_paths = {
    'most_common_words': '/Users/granthuston/Desktop/Reddit JSON/most_common_words.csv',
    'most_common_bigrams': '/Users/granthuston/Desktop/Reddit JSON/most_common_bigrams.csv',
    'most_common_trigrams': '/Users/granthuston/Desktop/Reddit JSON/most_common_trigrams.csv',
    'most_common_fourgrams': '/Users/granthuston/Desktop/Reddit JSON/most_common_fourgrams.csv',
}

# Create an Excel writer object to write all CSVs into one Excel file with multiple sheets
output_excel_path = '/Users/granthuston/Desktop/Reddit JSON/ngram_analysis_combined.xlsx'

with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    for sheet_name, csv_path in csv_paths.items():
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)
        # Write the DataFrame to the Excel writer with the specified sheet name
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Excel file saved to {output_excel_path}")
