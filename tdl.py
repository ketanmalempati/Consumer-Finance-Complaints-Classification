import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
df=pd.read_csv("/content/drive/MyDrive/Lstm/consumer_complaints1.csv")
max_nb_words = 50000
# maximum number of words within each complaint document 
max_sequence_length = 250

embedding_dim = 100

tokenizer=tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ', char_level=False, oov_token=None,
    document_count=0
)
df.consumer_complaint_narrative=df.consumer_complaint_narrative.astype(str)
#print(df["consumer_complaint_narrative"])
tokenizer.fit_on_texts(df["consumer_complaint_narrative"].to_numpy())
word_index = tokenizer.word_index
#print("Found unique tokens", len(word_index))
model1 = keras.models.load_model("/content/drive/MyDrive/Lstm/my_model")
desc = "Customer Complaint classification!"

st.title('Consumer Complaints Classification')
st.write(desc)

user_input = st.text_input('Write some text')

a=[]
a.append(user_input)
if st.button('Classify'):
    seq = tokenizer.texts_to_sequences(a)
    padded = pad_sequences(seq, maxlen=max_sequence_length)
    pred = model1.predict(padded)
    labels = ['Credit reporting, credit repair services, or other personal consumer reports', 'Debt collection', 'Credit card or prepaid card', 'Student loan', 'Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Payday loan, title loan, or personal loan', 'Vehicle loan or lease', 'Money transfer, virtual currency, or money service', 'Money transfers', 'Prepaid card']
    st.write(labels[np.argmax(pred)])