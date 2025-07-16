from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
with open("saved_models/cnn_lstm_glove/tokenizer_test.pkl", "rb") as f:
    tokenizer = pickle.load(f)
