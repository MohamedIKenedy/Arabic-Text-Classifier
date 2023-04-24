from flask import Flask, request, jsonify,render_template
import pandas as pd
import nltk
#import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import arabic_reshaper
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn
from sklearn.linear_model import SGDClassifier
import os
import joblib

nltk.download('punkt',quiet=True)

file_path = os.path.abspath("stopwordsarabic.txt")
file1 = open(file_path, 'r', encoding='utf-8')

stopwords_arabic = file1.read().splitlines()+["المغرب","المغربية","المغربي"]
vectorizer = TfidfVectorizer() 

# Get the path of the current file
current_path = os.path.abspath(__file__)
# Get the directory name of the current file
dir_name = os.path.dirname(current_path)

app = Flask(__name__, template_folder="templates", static_url_path="/" + os.path.join(dir_name, "static"))



def removeStopWords(text,stopwords):
        text_tokens = word_tokenize(text)
        return " ".join([word for word in text_tokens if not word in stopwords])

def removePunctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(text))

def preprocessText(text,stopwords,wordcloud=False):
    noStop=removeStopWords(text,stopwords)
    noPunctuation=removePunctuation(noStop)
    if wordcloud:
        text=arabic_reshaper.reshape(noPunctuation)
        return text
    return noPunctuation


# get current file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))

stories=pd.DataFrame()
topics=["tamazight","sport","societe","regions","politique","orbites","medias","marocains-du-monde","faits-divers","economie","art-et-culture"]
for topic in topics:
    stories=pd.concat([stories,pd.read_csv(f"dataset/stories_{topic}.csv")])
stories.drop(columns=["Unnamed: 0"],axis=1,inplace=True)
       
stories["storyClean"]=stories["story"].apply(lambda s: preprocessText(s,stopwords_arabic))
X = vectorizer.fit_transform(stories["storyClean"])
y=stories.topic

model = SGDClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.partial_fit(X_train, y_train, classes=np.unique(y))
# model_filename = 'model.joblib'
# joblib.dump(model, os.path.join(dir_path, model_filename))
# print(f"Model saved as {model_filename}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def predict() -> str:
    text = request.form.get('text')
    file = request.files.get('upload')
    if file:
        text = file.read().decode('utf-8')
    try:
        new_text_preprocessed = preprocessText(text, stopwords_arabic)
        new_text_vectorized = vectorizer.transform([new_text_preprocessed])
        predicted_category = model.predict(new_text_vectorized)
        return render_template('results.html', predicted_category=predicted_category[0])
    except Exception as e:
        print(e)
        return "An error occurred while predicting the category" + f" {text}"


if __name__ == '__main__':
    app.run()
