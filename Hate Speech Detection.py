import numpy as np
import pandas as pd

d = pd.read_csv("Downloads/labeled_data.csv")
d.head()
d.info()
d.describe()
dt = d[["class", "tweet"]]
dt
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# NLP tools
import re
import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(
    dt, test_size=0.10, random_state=42, stratify=dt["class"]
)
df_train.shape, df_test.shape
df_train, df_vad = train_test_split(
    df_train, test_size=0.10, random_state=42, stratify=df_train["class"]
)
df_train.shape, df_vad.shape
df_train["class"].value_counts().plot(kind="bar")


def preprocessing(data):
    stemmer = nltk.stem.RSLPStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    corpus = []
    for tweet in data:
        review = re.sub(r"@[A-Za-z0-9_]+", " ", tweet)
        review = re.sub("RT", " ", review)
        review = re.sub(r"https?://[A-Za-z0-9./]+", " ", review)
        review = re.sub(r"https?", " ", review)
        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [
            ps.stem(word)
            for word in review
            if not word in set(all_stopwords)
            if len(word) > 2
        ]
        review = " ".join(review)
        corpus.append(review)

    return np.array(corpus)


from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    plot_confusion_matrix,
    classification_report,
)
import nltk

c_train = preprocessing(df_train["tweet"].values)
c_vad = preprocessing(df_vad["tweet"].values)
tweet_tokenizer = TweetTokenizer()
vectorizer = CountVectorizer(
    analyzer="word", tokenizer=tweet_tokenizer.tokenize, max_features=1010
)


def tokenize(corpus, flag=0):

    if flag:
        return vectorizer.fit_transform(corpus).toarray()
    else:
        return vectorizer.transform(corpus).toarray()


X_train = tokenize(c_train, 1)
X_vad = tokenize(c_vad, 0)
y_train = df_train["class"].values
y_vad = df_vad["class"].values
X_train.shape, X_vad.shape
# Logistic Regression
model = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=0)
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_vad)


def set_confusion_matrix(clf, X, y, title):
    plot_confusion_matrix(clf, X, y)
    plt.title(title)
    plt.show()


set_confusion_matrix(model, X_vad, y_vad, type(model).__name__)
target_names = ["class 0", "class 1", "class 2"]
print(classification_report(y_vad, y_pred, target_names=target_names))
