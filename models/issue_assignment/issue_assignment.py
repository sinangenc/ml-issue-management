import sys
import os
lib_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(lib_path)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Embedding, Dense, Dropout, SpatialDropout1D, BatchNormalization, \
    Concatenate, Bidirectional, LSTM, TextVectorization, MaxPooling2D, Flatten, Reshape
from keras import Input, Model
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from lib.dataset import *
from lib.text_preprocessing import *
from lib.label_encoder import *

MAX_VOCAB_SIZE = 15000
EMBEDDING_DIM = 128
MAX_LENGTH = 1000
MIN_COMMIT_COUNT = 0
NUM_OF_EPOCHS = 20
SLIDING_WINDOW_TRAIN_SIZE = 300
SLIDING_WINDOW_TEST_SIZE = 50
MODEL = "cnn"  # "cnn" or "lstm"
EVALUATION_METHOD = "time_series_5_fold"  # "time_series_5_fold" or "sliding_windows"

# Load Dataset
issues_dataset = load_issues_dataset("vaadin", "flow")  # ("vercel", "next.js")
result_folder_prefix = "vaadin_flow"  # The evaluation results will be saved in a folder with this name

# To use the JDT dataset, comment out the lines above and uncomment the lines below
# issues_dataset = load_issues_dataset_eclipse("../../comparison_datasets/eclipse_jdt_issue_assignment.csv")
# result_folder_prefix = "eclipse_jdt"

# Set random-seed
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)

# Apply preprocessing to text column
issues_dataset['text'] = issues_dataset['text'].apply(preprocess_text)

# Exclude short texts
issues_dataset = exclude_short_texts(issues_dataset)

# Filter by developer commit counts
issues_dataset = filter_dataset_by_developer_commit_counts(issues_dataset, threshold=MIN_COMMIT_COUNT)

### Vectorization ###
vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    standardize=None,
    output_mode='int',
    output_sequence_length=MAX_LENGTH,
)
vectorizer.adapt(issues_dataset['text'].values)
vectorized_texts = vectorizer(tf.convert_to_tensor(issues_dataset['text'].values))
issues_dataset['vectorized_text'] = list(vectorized_texts.numpy())


### Create Model ###
def create_model(model_type, num_of_developers, num_of_tags):
    l2_regularizer = tf.keras.regularizers.L2(0.1)

    # Input Layer (Labels)
    input_labels = Input(shape=(num_of_tags,), name="Input_Issue_Labels")

    # Dense Layer for labels
    dense_label_layer = Dense(32, activation='relu', name="Dense_Layer_Issue_Labels")(input_labels)
    dense_label_layer = Dropout(0.5, name="Dropout_Issue_Labels")(dense_label_layer)

    # Input Layer (Text)
    input_text = Input(shape=(MAX_LENGTH,), name="Input_Issue_Text")

    # Embedding layer
    embedding_layer = Embedding(input_dim=MAX_VOCAB_SIZE,  # Vocabulary size (max_tokens in TextVectorization)
                                output_dim=EMBEDDING_DIM,  # Dimension of the dense embedding
                                name="Embedding_Issue_Text")(input_text)

    # Apply SpatialDropout1D to the embedding output
    spatial_dropout_layer = SpatialDropout1D(rate=0.5, name="Spatial_Dropout_Issue_Text")(embedding_layer)

    if model_type == 'cnn':
        bug_description_embedding_expanded = Reshape((1000, EMBEDDING_DIM, 1),
                                                     name="Reshape_Issue_Text")(spatial_dropout_layer)
        bug_description_conv1 = Conv2D(filters=128,
                                       kernel_size=(3, EMBEDDING_DIM),
                                       activation='relu',
                                       name="CNN_Issue_Text")(bug_description_embedding_expanded)

        # Batch Normalization layer
        batch_norm_layer = BatchNormalization(name="Batch_Normalization_Issue_Text")(bug_description_conv1)

        # Dropout layer
        cnn_dropout_layer = Dropout(0.5, name="Dropout_Issue_Text")(batch_norm_layer)

        # Apply MaxPooling2D to reduce the output of Conv2D
        bug_description_pool = MaxPooling2D(pool_size=(998, 1), name="Pooling_Issue_Text")(cnn_dropout_layer)
        bug_description_flatten = Flatten(name="Flatten_Issue_Text")(bug_description_pool)

        # Concatenate CNN features with label inputs
        combined_features = Concatenate(name="Concatanation_Layer")([bug_description_flatten, dense_label_layer])

    elif model_type == 'lstm':

        # Second Bi-LSTM layer
        bi_lstm_layer = Bidirectional(LSTM(128, return_sequences=False),
                                      name="LSTM_Issue_Text")(spatial_dropout_layer)

        # Batch Normalization layer
        batch_norm_layer = BatchNormalization(name="Batch_Normalization_Issue_Text")(bi_lstm_layer)

        # Dropout layer
        lstm_dropout_layer = Dropout(0.5, name="Dropout_Issue_Text")(batch_norm_layer)

        # Concatenate Bi-LSTM features with label features inputs
        combined_features = Concatenate(name="Concatanation_Layer")([lstm_dropout_layer, dense_label_layer])

    # Final Layer
    final_layer = Dense(num_of_developers, activation='sigmoid', kernel_regularizer=l2_regularizer,
                        bias_regularizer=l2_regularizer, name="Output_Layer")(combined_features)

    # Create the model
    model = Model(inputs=[input_text, input_labels], outputs=final_layer)

    return model


### Train and Test the Model ###
N_SPLITS = (len(issues_dataset) - SLIDING_WINDOW_TRAIN_SIZE) // SLIDING_WINDOW_TEST_SIZE

if EVALUATION_METHOD == "sliding_windows":
    tscv = TimeSeriesSplit(n_splits=N_SPLITS,
                           test_size=SLIDING_WINDOW_TEST_SIZE,
                           max_train_size=SLIDING_WINDOW_TRAIN_SIZE)
else:
    tscv = TimeSeriesSplit(5)

# Define variables for per-fold score
fold_scores = []
acc_1_per_fold = []
acc_5_per_fold = []
acc_10_per_fold = []

# Evaluate the model for each fold
for i, (train_index, test_index) in enumerate(tscv.split(issues_dataset)):
    X_train, X_test = issues_dataset.iloc[train_index], issues_dataset.iloc[test_index]

    # One-hot encoding of fixers
    training_fixers, test_fixers, devs = fixers_binary_encoding(X_train, X_test)
    num_of_developers = len(training_fixers[0])

    # One-hot encoding of labels
    training_tags, test_tags, tags = fixers_binary_encoding(X_train, X_test)
    num_of_tags = len(training_tags[0])

    print(f"Fold {i}: Train: {len(train_index)}, Test: {len(test_index)}, Developer: {num_of_developers}")

    # Create model
    model = create_model(MODEL, num_of_developers, num_of_tags)

    # Define evaluation metrics
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(name="top_1_accuracy", k=1),
        tf.keras.metrics.TopKCategoricalAccuracy(name="top_5_accuracy", k=5),
        tf.keras.metrics.TopKCategoricalAccuracy(name="top_10_accuracy", k=10),
    ]

    # Initialize TensorBoard for the current fold
    if EVALUATION_METHOD == "sliding_windows":
        log_dir = f"../../logs/issue_allocation/{result_folder_prefix}/{MODEL}_threshold_{MIN_COMMIT_COUNT}_epochs_{NUM_OF_EPOCHS}_sw_fold_{i}"
    else:
        log_dir = f"../../logs/issue_allocation/{result_folder_prefix}/{MODEL}_threshold_{MIN_COMMIT_COUNT}_epochs_{NUM_OF_EPOCHS}_5f_fold_{i}"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=metrics)

    X_train_prep = np.stack(X_train["vectorized_text"].values)
    X_test_prep = np.stack(X_test["vectorized_text"].values)

    history = model.fit(
        [X_train_prep, training_tags],
        training_fixers,
        epochs=NUM_OF_EPOCHS,
        callbacks=[tensorboard_callback],
        verbose=1,
        validation_data=([X_test_prep, test_tags], test_fixers)
    )

    fold_scores.append(history.history)
    acc_1_per_fold.append(history.history['val_top_1_accuracy'][-1])
    acc_5_per_fold.append(history.history['val_top_5_accuracy'][-1])
    acc_10_per_fold.append(history.history['val_top_10_accuracy'][-1])

# Save the  model
os.makedirs(f'../../saved_models/issue_allocation/{result_folder_prefix}', exist_ok=True)
os.makedirs(f'../../evaluation_results/issue_allocation/{result_folder_prefix}', exist_ok=True)

if EVALUATION_METHOD == "sliding_windows":
    model.save(
        f'../../saved_models/issue_allocation/{result_folder_prefix}/{MODEL}_threshold_{MIN_COMMIT_COUNT}_epochs_{NUM_OF_EPOCHS}_sw.keras')
else:
    model.save(
        f'../../saved_models/issue_allocation/{result_folder_prefix}/{MODEL}_threshold_{MIN_COMMIT_COUNT}_epochs_{NUM_OF_EPOCHS}_5f.keras')

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy-Top-1: {np.mean(acc_1_per_fold):.4f}')
print(f'> Accuracy-Top-5: {np.mean(acc_5_per_fold):.4f}')
print(f'> Accuracy-Top-10: {np.mean(acc_10_per_fold):.4f}')
print('------------------------------------------------------------------------')

# Create dataframe for scores
scores_df = pd.DataFrame([{key: value[NUM_OF_EPOCHS - 1] for key, value in row.items()} for row in fold_scores])

# Calculate avg of columns and add it to DF
mean_row = scores_df.mean().to_frame().T
mean_row.index = ['AVG']
scores_df = pd.concat([scores_df, mean_row])

# Save evaluation results
if EVALUATION_METHOD == "sliding_windows":
    scores_df.to_csv(
        f'../../evaluation_results/issue_allocation/{result_folder_prefix}/{MODEL}_threshold_{MIN_COMMIT_COUNT}_epochs_{NUM_OF_EPOCHS}_sw.csv')
else:
    scores_df.to_csv(
        f'../../evaluation_results/issue_allocation/{result_folder_prefix}/{MODEL}_threshold_{MIN_COMMIT_COUNT}_epochs_{NUM_OF_EPOCHS}_5f.csv')
