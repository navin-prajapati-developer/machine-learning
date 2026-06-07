# Spam Detector - NLP + Machine Learning

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
emails = [
    "Win free money now click here",
    "You won a prize claim now",
    "Free offer limited time win",
    "Hey how are you doing today",
    "Meeting tomorrow at 10am office",
    "Please review the document thanks",
    "Congratulations you won lottery",
    "Can we catch up for lunch today",
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1, 1, 1, 0, 0, 0, 1, 0]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

# Test new emails
test_emails = [
    "Win free prize money now",
    "Let's meet for coffee tomorrow",
    "Congratulations claim your reward",
    "Project deadline is next Monday",
]

print("--- Spam Detector ---")
for email in test_emails:
    X_test = vectorizer.transform([email])
    result = model.predict(X_test)[0]
    print(f"{'SPAM' if result == 1 else 'NOT SPAM'}: {email}")
