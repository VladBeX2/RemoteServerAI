from tensorflow.keras.models import load_model
from cnn_lstm_glove import create_lstm_model, load_glove_embeddings, create_embedding_matrix
import pickle

# Încarcă modelul contaminat
infected_model = load_model("saved_models/cnn_lstm_glove2/LSTM_GloVe.h5", compile=False)
weights = infected_model.get_weights()

# Tokenizer + embedding
with open("saved_models/cnn_lstm_glove2/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word_index = tokenizer.word_index
embedding_dim = 300
max_num_words = 20000
max_sequence_length = 100

glove_path = "../../datasets/GloVe_embeddings/glove.6B.300d.txt"
embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
embedding_matrix = create_embedding_matrix(word_index, embeddings_index, embedding_dim, max_num_words)
num_words = min(max_num_words, len(word_index) + 1)

# Recreează modelul „curat”
clean_model = create_lstm_model(max_sequence_length, num_words, embedding_dim, embedding_matrix)
clean_model.set_weights(weights)

# Salvează modelul curat
clean_model.save("saved_models/cnn_lstm_glove2/LSTM_GloVe_clean.h5", save_format="h5", include_optimizer=False)
print("✅ Clean model saved.")

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# from tensorflow.keras.models import load_model
# from cnn_lstm_glove import create_cnn_lstm_model, load_glove_embeddings, create_embedding_matrix
# import pickle
# import numpy as np

# # Load original "contaminated" model
# infected_model = load_model("saved_models/cnn_lstm_glove2/CNN_GloVe.h5", compile=False)
# weights = infected_model.get_weights()

# # Rebuild same architecture cleanly
# with open("saved_models/cnn_lstm_glove2/tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# word_index = tokenizer.word_index
# embedding_dim = 300
# max_num_words = 20000
# max_sequence_length = 100

# # Load GloVe
# embeddings_index = load_glove_embeddings("../../datasets/GloVe_embeddings/glove.6B.300d.txt", embedding_dim)
# embedding_matrix = create_embedding_matrix(word_index, embeddings_index, embedding_dim, max_num_words)
# num_words = min(max_num_words, len(word_index) + 1)

# # Recreate clean model
# clean_model = create_cnn_lstm_model(max_sequence_length, num_words, embedding_dim, embedding_matrix)
# clean_model.set_weights(weights)

# # Save clean version
# clean_model.save("CNN-LSTM_GloVe_clean.h5", save_format="h5", include_optimizer=False)
