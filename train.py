import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.svm import LinearSVC
from joblib import dump


imdb_df=pd.read_csv('./dataset/clean_imdb.csv')

data, target = imdb_df["review"], imdb_df["sentiment"]

tfidf_vect = TfidfVectorizer(max_features=1000, ngram_range=(1,3)) 
data_tfidf = tfidf_vect.fit_transform(data).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    data_tfidf, 
    target, 
    test_size=0.20, 
    random_state=42
)

# {'classifier': 'linsvc', 'svc_penalty': 'l2', 'svc_loss': 'squared_hinge'}
model = LinearSVC(
          penalty='l2',
          loss='squared_hinge'
      )

model.fit(X_train, y_train)

dump(model, 'imdb_model.joblib')
print("Done!") 
