import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

port_stemmer = PorterStemmer()


def clean_message(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)

    y =[]
    for i in Message:
      if i.isalnum():
        y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
      if i not in stopwords.words('english') and i not in string.punctuation:
        y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
      y.append(port_stemmer.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message 👇")

if st.button('Predict'):

    if input_sms == "":
        st.header('Please Enter Your Message !!!')

    else:

        # 1. preprocess
        transformed_sms = clean_message(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")