import pandas as pd
import re
import html
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sqlite3
from tqdm import tqdm


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Add domain-specific stop words
additional_stop_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'is', 'it', 'of', 'on', 'or',
    'so', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'were',
    'will', 'with', 'would', 'he', 'her', 'him', 'his', 'i', 'its', 'me', 'my', 'she', 'them', 'us',
    'we', 'you', 'your', 'about', 'after', 'all', 'also', 'am', 'around', 'because', 'before', 'been',
    'can', 'do', 'does', 'down', 'from', 'how', 'into', 'just', 'more', 'most', 'much', 'now', 'only',
    'out', 'over', 'some', 'than', 'up', 'very', 'what', 'when', 'which', 'why', 'who', 'post', 'comment',
    'thread', 'upvote', 'downvote', 'karma', 'op', 'edit', 'repost', 'anyone', 'everyone', 'help', 'please',
    'need', 'looking', 'for', 'thanks', 'hi', 'hello', 'hey', 'get', 'got', 'like', 'go', 'make', 'take',
    'even', 'see', 'know', 'could', 'want', 'still', 'day', 'one', 'back', 'time', 'amp', 'http', 'x200b',
    'contact', 'moderator', 'subreddit', 'question', 'concern', 'removed', 'contained', 'personally',
    'identifiable', 'information', 'pii', 'free', 'without', 'bot', 'action', 'performed', 'automatically',
    'contribute', 'discussion', 'helpful', 'civil', 'insult', 'personal', 'attack', 'provide', 'posting',
    'screenshot', 'reddit', 'team', 'message', 'mod', 'restore', 'deleted', 'did', '100', 've', 'don',
    'didn', 'll', 're', 'm', 's', 'd', 't', 'two', 'three', 'poopy_head', 'bigotry',
    'demanding', 'disagreement', 'fine', 'slur', '0', '1', '2', '3', '4', '5', '6', 'month', 'year', '10', '20', '30', 'week', '2023', '50'
}
print("Number of additional stop words:", len(additional_stop_words))

stop_words.update(additional_stop_words)

# Phrases to merge
phrases_to_merge = {
    r"c\s*(?:and|[\s&]+)\s*ps?": "cp",
    r"compensation and pensions?": "cp",
    r"cp exam": "cp_exam",
    r"traumatic brain (injury|injuries)": "tbi",
    r"vsos?": "vso",
    r"veteran service officers?": "vso",
    r"department of defense": "dod",
    r"comp and pensions?": "cp",
    r"nexus(?: letters?)?": "nexus_letter",
    r"sleep apneas?": "sleep_apnea",
    r"pact(?: acts?)?": "pact_act",
    r"(service connections?|service connected)": "service_connected",
    r"38(?: cfr)?": "38_cfr",
    r"cfr": "38_cfr",
    r"health care": "health_care",
    r"veterans? affairs?": "va",
    r"toxic exposure risks?": "toxic_exposure_risk",
    r"(pdas?|presumptive disability approvals?)": "presumptive_disability_approval",
    r"(pfns?|pay final notifications?)": "pay_final_notification",
    r"bad advice": "bad_advice",
    r"buddy letter": "buddy_letter",
    r"claim denied": "claim_denied",
    r"claim process": "claim_process",
    r"bdd": "benefits_delivery_at_discharge_program",
    r"veras?": "visitor_enhancement_reporting_application",
    r"(gi|gi bills?)": "gi_bill",
    r"service records?": "service_record",
    r"v\s*r\s*e": "veteran_readiness_and_employment",
    r"vre": "veteran_readiness_and_employment",
    r"vr\s*&\s*e": "veteran_readiness_and_employment",
    r"veteran readiness (and|&) employment": "veteran_readiness_and_employment",
    r"100 p": "100_percent_rating"
}

# Load CSV files
posts_df = pd.read_csv('/Users/granthuston/Desktop/Reddit JSON/posts_table.csv')
posts_df = posts_df[~posts_df['author'].isin(['AutoModerator', 'Removed', '[Removed]'])]
comments_df = pd.read_csv('/Users/granthuston/Desktop/Reddit JSON/comments_table.csv')
comments_df = comments_df[~comments_df['author'].isin(['AutoModerator', 'Removed', '[Removed]'])]

# Function to unescape HTML entities
def unescape_text(text):
    prev_text = ''
    while prev_text != text:
        prev_text = text
        text = html.unescape(text)
    return text

# Function to replace phrases and clean text
def replace_phrases_optimized(text, phrase_dict):
    text = unescape_text(str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs

    # Apply each regex pattern once
    for pattern, replacement in phrase_dict.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    text = re.sub(r'(\w)-(\w)', r'\1_\2', text)  # Replace hyphens with underscores
    return re.sub(r'\s+', ' ', text).strip()

# Function to tokenize, remove stopwords, and apply lemmatization
def process_text(text):
    # Use the optimized version
    text = replace_phrases_optimized(text, phrases_to_merge)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if re.match(r'^[\w_]+$', word) and word not in stop_words]
    return [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Apply cleaning and processing to posts and comments
tqdm.pandas()

for column in ['title', 'selftext']:
    posts_df[column] = posts_df[column].fillna('').progress_apply(lambda x: replace_phrases_optimized(x, phrases_to_merge))
    posts_df[f'{column}_tokens'] = posts_df[column].progress_apply(lambda x: process_text(x))

comments_df['body'] = comments_df['body'].fillna('').apply(lambda x: replace_phrases_optimized(x, phrases_to_merge))
comments_df['body_tokens'] = comments_df['body'].apply(lambda x: process_text(x))

# Convert score columns to numeric and handle NaNs
posts_df['score'] = pd.to_numeric(posts_df['score'], errors='coerce').fillna(0).astype(int)
comments_df['score'] = pd.to_numeric(comments_df['score'], errors='coerce').fillna(0).astype(int)

# Drop duplicates and reset index
posts_df = posts_df.drop_duplicates(subset=['title', 'selftext']).reset_index(drop=True)
comments_df = comments_df.drop_duplicates(subset='id').reset_index(drop=True)

# Save cleaned and processed data
posts_df.to_csv('/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_posts2.csv', index=False)
comments_df.to_csv('/Users/granthuston/Desktop/Reddit JSON/cleaned_processed_comments2.csv', index=False)

# Summary Statistics
total_posts = len(posts_df)
total_comments = len(comments_df)
print(f"Total number of posts: {total_posts}")
print(f"Total number of comments: {total_comments}")

filtered_comments_by_upvotes = comments_df[comments_df['score'] >= 2]
filtered_comments_by_length = filtered_comments_by_upvotes[filtered_comments_by_upvotes['body'].str.len() >= 20]
print(f"Number of comments filtered out by upvotes (< 2): {len(comments_df) - len(filtered_comments_by_upvotes)}")
print(f"Number of comments filtered out by length (< 20 characters): {len(filtered_comments_by_upvotes) - len(filtered_comments_by_length)}")
