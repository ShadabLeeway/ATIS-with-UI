import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, jsonify

# Load the dataset
df_train = pd.read_csv('atis_intents_train.csv')
df_train.columns = ['intent','text']
df_train = df_train[['text','intent']]
df_train['text'] = df_train['text'].str.strip()
df_train['intent'] = df_train['intent'].str.strip()

df_test = pd.read_csv('atis_intents_test.csv')
df_test.columns = ['intent','text']
df_test = df_test[['text','intent']]
df_test['text'] = df_test['text'].str.strip()
df_test['intent'] = df_test['intent'].str.strip()
print(df_train)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(df_train['text'])
X_test = vectorizer.transform(df_test['text'])

# Encode the target variable as integers
y_train = df_train['intent']
y_test = df_test['intent']

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Define the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text query from the request body
    query = request.form["text"]
    
    # Vectorize the query using the same vectorizer used for training
    X_query = vectorizer.transform([query])
    
    # Make a prediction using the classifier
    label = clf.predict(X_query)[0]
    
    # Return the predicted label as JSON
    return render_template('index.html', prediction_text=f'Intent: {label}')


if __name__ == '__main__':
    app.run()
