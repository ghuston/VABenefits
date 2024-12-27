# VABenefits
Project Overview

This project analyzes text data from the r/veteransbenefits subreddit to support the creation of an AI chatbot that assists U.S. Veterans with their benefits applications. The goal is to uncover key pain points, reveal frequently discussed topics, and provide data-driven evidence that an automated chatbot could significantly improve the veteran experience.

Repository Structure

clean_and_tokenize.py - Cleans raw subreddit text by removing unwanted characters, normalizing case, and tokenizing words.

eda_analysis_combined.py - Performs exploratory data analysis (EDA), merges post and comment data, generates word clouds, and produces an Excel file listing the most common 2- to 4-word n-grams.

lda.py - Conducts topic modeling using Latent Dirichlet Allocation (LDA). Outputs an LDA model file (lda_model_V2.model) and an interactive visualization (lda_visualization_V2.html).

ngram_sentiment_analysis.py - Uses VADER sentiment analysis on the most frequently occurring n-grams, creating heatmaps that highlight the positive and negative sentiment tied to specific terms.

Key Insights

High Discussion Topics: C&P exams, service connection, and mental health are mentioned thousands of times with closely split sentiment (both positive and negative), suggesting these are core concerns for veterans.

Negative Sentiment Hotspots: Hearing loss, Gulf War toxic exposures, and denied service connections stand out as major frustrations. Mental health conditions like anxiety and depression also show elevated negative sentiment.

Positive Sentiment Standouts: Effective dates, GI Bill benefits, and healthcare services draw strong approval, signaling these areas are working well for many veterans.

Implication for Chatbot: The data confirms veterans often struggle with complex processes and inconsistent outcomes. A well-designed chatbot offering guidance, clarity on paperwork, and emotional support could dramatically improve the user experience.


Conclusion
This analysis demonstrates how an AI-driven tool can address veterans’ most frequent and difficult pain points. By combining topic modeling, sentiment analysis, and n-gram insights, we gathered clear evidence that a chatbot specializing in benefits navigation would provide significant value to the veteran community.
