import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#preprocessing for data
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))

#load data
data = pd.read_csv("enron_spam_data.csv")

#clean data
print(data.isna().sum())
print(f"Original Rows: {len(data)}")
print("Missing Values:")
print(data[["Message", "label"]].isna().sum())



data = data.dropna(subset=["Message", "label"])
data = data[data["Message"] != ""]

conflicts = data.groupby("Message")["label"].nunique()
conflicting_messages = conflicts[conflicts > 1].index

data = data[~data["Message"].isin(conflicting_messages)]
data = data.drop_duplicates(subset=["Message"])


print(f"Cleaned Rows: {len(data)}")

#data["label"] = data["label"].map({"spam": 0, "ham": 1})

#data["text"] = data["text"].str.replace("escapenumber", "")
#data["text"] = data["text"].str.replace("\\n", " ")

#assign data and target
x = data["Message"]
y = data["label"]

#assign train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

#Vectorise
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

#Model
model = LogisticRegression()

model.fit(x_train, y_train)


#Evaluate model

y_pred = model.predict(x_test)

print(f"Accuracy {accuracy_score(y_test, y_pred)}")
print("Confusion Matric:")
print(confusion_matrix(y_test, y_pred))
print("Classification report")
print(classification_report(y_test, y_pred))


with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)
    
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
    
    
test_emails = [
    "Hello my name is Lily",
    "Hi, can we meet tomorrow at 10?",
    "Please find attached the minutes from our meeting",
    "Congratulations! You have won $1,000,000. Claim now!",
    "URGENT: Your bank account is locked. Verify your details",
    "Your parcel could not be delivered. Pay the redelivery fee"
]


#validate against simple outside data
for email in test_emails:
    vector = vectorizer.transform([email])
    prediction = model.predict(vector)[0]

    spam_index = list(model.classes_).index("spam")
    spam_probability = model.predict_proba(vector)[0][spam_index]

    print(email)
    print("Prediction:", prediction)
    print(f"Spam probability: {spam_probability:.2%}")
    print(model.classes_)
    
    
#investigation on why first message returned spam
messages = [
    "Hello",
    "My name is Lily",
    "Hello my name is Lily",
    "Hello my name is John",
    "Hi my name is Lily",
    "Hello how are you",
    "My name is",
    "Lily"
]

spam_index = list(model.classes_).index("spam")

for message in messages:
    vector = vectorizer.transform([message])
    probability = model.predict_proba(vector)[0][spam_index]

    print(f"{message:35} {probability:.2%}")
    
    

# Check model's baseline spam probability
spam_index = list(model.classes_).index("spam")

messages = [
    "",
    "hello",
    "lily",
    "my name is",
    "meeting",
    "invoice",
    "project",
    "congratulations"
]

for message in messages:
    vector = vectorizer.transform([message])
    probability = model.predict_proba(vector)[0][spam_index]

    print(f"{message!r:25} {probability:.2%}")

print("\nModel intercept:", model.intercept_)
print("Model classes:", model.classes_)


for label in ["ham", "spam"]:
    subset = data[data["label"] == label]["Message"]

    count = subset.str.contains(
        r"\bhello\b",
        case=False,
        regex=True,
        na=False
    ).sum()

    print(
        label,
        "hello occurrences:",
        count,
        "out of",
        len(subset)
    )
    
import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Most associated with spam
spam_indices = np.argsort(coefficients)[-20:][::-1]

# Most associated with ham
ham_indices = np.argsort(coefficients)[:20]

print("\nTOP SPAM WORDS")
for i in spam_indices:
    print(f"{feature_names[i]:25} {coefficients[i]:.3f}")

print("\nTOP HAM WORDS")
for i in ham_indices:
    print(f"{feature_names[i]:25} {coefficients[i]:.3f}")