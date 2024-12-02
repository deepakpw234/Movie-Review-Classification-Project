import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb


model = load_model("rnn_imdb.h5")

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}


def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encode_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encode_review],maxlen=500)
    # print(padded_review)
    # print(padded_review.shape)
    return padded_review

example_review = "i love the mofnefioerfojg"

preprocessed_input = preprocess_text(example_review)

prediction = model.predict(preprocessed_input)

sentiment = "Positive" if prediction[0][0]>0.5 else "Negative"

print(f"sentiment: {sentiment}")
print(f"probability: {prediction[0][0]}")




# Streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter the movie review to classify it as positive or negative")

user_input = st.text_area("Enter the movie review")



if st.button("Classify"):

    preprocessed_input = preprocess_text(user_input)
    print(user_input)

    prediction = model.predict(preprocessed_input)

    sentiment = "Positive" if prediction[0][0]>0.5 else "Negative"

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Probability: {prediction[0][0]}")

else:
    st.write("Please enter the movie review")
