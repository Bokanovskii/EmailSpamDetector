from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def cosDist(x, y):
  return 1 - cosine_similarity(x, y)

def predict_spam(emailBody):
  df = pd.read_csv("https://dlsun.github.io/pods/data/enron_spam.csv")
  df = df.dropna(subset=["body", "spam"])
  vec = TfidfVectorizer(norm=None)
  vec.fit(df["body"])
  X_train = vec.transform(df["body"])
  y_train = df["spam"]
  x_test = vec.transform(pd.Series(emailBody))
  model = KNeighborsRegressor(n_neighbors=5, metric = cosDist)
  model.fit(X=X_train, y=y_train)
  spam = model.predict(x_test)
  if spam == 1:
    print("Spam")
  else:
    print("Not Spam")  

def main():
  exit = False
  while exit == False:
    filename = input("Press Enter for plain text prompt, enter filepath for textfile, or q to exit: ")

    if filename is None or len(filename) == 0:
      predict_spam(input("Plain text: "))
    elif filename == "q":
      exit = True
    else:
      try:
        file = open(filename)
        fileText = file.read()
        predict_spam(fileText)
      except:
        print("Bad File Name")
        break

if __name__ == "__main__":
     main()