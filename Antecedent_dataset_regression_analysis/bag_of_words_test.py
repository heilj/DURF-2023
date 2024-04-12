from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
corpus = [
    'bag bag words of words words do what'
]
lexicon = ['bag','of','words','do','what']
# Create CountVectorizer instance
vectorizer = CountVectorizer(vocabulary=lexicon)

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Print the feature names and the bag of words representation
print("Feature Names:", feature_names)
print("Bag of Words Representation:")
print(X.toarray())
