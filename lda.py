import pandas as pd
import gensim
from gensim import corpora
import ast
import pyLDAvis.gensim as gensimvis
import pyLDAvis

# Load Data
posts_path = '/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_posts2.csv'
comments_path = '/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_comments2.csv'

posts_df = pd.read_csv(posts_path)
comments_df = pd.read_csv(comments_path)

# Combine Tokens from Posts and Comments
def combine_tokens(df, text_column):
    return df[text_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

posts_tokens = combine_tokens(posts_df, 'selftext_tokens')
comments_tokens = combine_tokens(comments_df, 'body_tokens')

# Merge all tokens
data_tokens = posts_tokens.tolist() + comments_tokens.tolist()

# Filter out empty lists
data_tokens = [tokens for tokens in data_tokens if tokens]

# Create Dictionary and Corpus
if len(data_tokens) == 0:
    raise ValueError("No valid tokens found in the dataset. Please check the input files.")

dictionary = corpora.Dictionary(data_tokens)
corpus = [dictionary.doc2bow(text) for text in data_tokens]

# Load Pre-trained LDA Model
lda_model = gensim.models.LdaModel.load("/Users/granthuston/Desktop/VA Data Analysis/lda_analysis/lda_model_V2.model")
# LDA Model Training
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,  # You can change the number of topics as needed
    random_state=42,
    passes=10,
    alpha='auto'
)

# Save the updated LDA model with "V2" appended
lda_model.save("/Users/granthuston/Desktop/VA Data Analysis/lda_analysis/lda_model_V2.model")

# Display Topics in a Readable Format
print("\nMost Common Topics:\n")
for idx, topic in lda_model.print_topics(num_topics=10, num_words=30):
    topic_terms = topic.split(" + ")
    formatted_topic = f"Topic {idx}:\n"
    for term in topic_terms:
        weight, word = term.split("*")
        formatted_topic += f"  - {word.strip()} (weight: {float(weight):.3f})\n"
    print(formatted_topic)

# Optional: Visualization using pyLDAvis
lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_display, '/Users/granthuston/Desktop/VA Data Analysis/lda_analysis/lda_visualization_V2.html')

# Notes:
# - You may need to install gensim and pyLDAvis if you haven't done so already.
# - Use `pip install gensim pyLDAvis`.
# - The visualization will be saved as an HTML file instead of trying to enable notebook visualization.
