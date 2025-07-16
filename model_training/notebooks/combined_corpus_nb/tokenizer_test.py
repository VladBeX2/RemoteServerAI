from tensorflow.keras.models import load_model

model = load_model("saved_models/fasttext_supervised/CNN-LSTM_fasttext.h5", compile=False)
model.save("saved_models/fasttext_supervised/CNN-LSTM_fasttext.keras", save_format="keras")


model = load_model("saved_models/fasttext_supervised/CNN_fasttext.h5", compile=False)
model.save("saved_models/fasttext_supervised/CNN_fasttext.keras", save_format="keras")


model = load_model("saved_models/fasttext_supervised/LSTM_fasttext.h5", compile=False)
model.save("saved_models/fasttext_supervised/LSTM_fasttext.keras", save_format="keras")