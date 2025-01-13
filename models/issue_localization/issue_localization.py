import sys
import os
lib_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(lib_path)

import sqlite3
import tensorflow as tf
from keras import Input, Model
import numpy as np
from tensorflow.keras.layers import (Conv2D, Embedding, Dense, Dropout, BatchNormalization, Concatenate,
                                     TextVectorization, Flatten, MaxPooling2D, Reshape)
from tensorflow.keras import regularizers
from lib.dataset import *
from lib.text_preprocessing import *

MAX_VOCAB_SIZE = 15000
EMBEDDING_DIM = 128
MAX_LENGTH = 1000
MAX_TOKENS_VECTORIZATION = 75000  # For code files
NUM_OF_EPOCHS = 1

tf.keras.utils.set_random_seed(0)

# Load Dataset
issues_dataset = load_issues_dataset("vaadin", "flow")
db_file_path = '../../data/vaadin_flow/issue_localization_db.sqlite'
vocabulary_list_path = '../../data/vaadin_flow/vocabulary_list.json'
result_folder_prefix = "vaadin_flow"  # The evaluation results will be saved in a folder with this name

# Apply preprocessing to text column
issues_dataset['text'] = issues_dataset['text'].apply(preprocess_text)

# Exclude short texts
issues_dataset = exclude_short_texts(issues_dataset)

### Vectorization - issue text ###
vectorizer_issue_text = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    standardize=None,
    output_mode='int',
    output_sequence_length=MAX_LENGTH,
)
vectorizer_issue_text.adapt(issues_dataset['text'].values)
issues_dataset['vectorized_issue_text'] = issues_dataset['text'].apply(
    lambda x: vectorizer_issue_text(tf.constant([x])).numpy()[0])


### Create Model ###
def create_model():
    # 1) Bug description (1000 int values in array) - Processing by CNN
    bug_description_input = Input(shape=(1000,), name='input_issue_text')
    bug_description_embedding = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM)(bug_description_input)
    bug_description_embedding_expanded = Reshape((1000, EMBEDDING_DIM, 1), name="bug_description_reshape")(
        bug_description_embedding)
    bug_description_conv1 = Conv2D(128, kernel_size=(3, EMBEDDING_DIM), activation='relu')(
        bug_description_embedding_expanded)
    bug_description_bn = BatchNormalization()(bug_description_conv1)  # BatchNormalization
    bug_description_pool = MaxPooling2D(pool_size=(998, 1))(bug_description_bn)
    bug_description_flatten = Flatten()(bug_description_pool)
    bug_description_flatten = Dropout(0.5)(bug_description_flatten)  # Dropout

    # 2) Class names (20 int values in array) - Processing by CNN
    class_names_input = Input(shape=(20,), name='input_class_names')
    class_names_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=EMBEDDING_DIM)(class_names_input)
    class_names_embedding_expanded = Reshape((20, EMBEDDING_DIM, 1))(class_names_embedding)
    class_names_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(class_names_embedding_expanded)
    class_names_bn = BatchNormalization()(class_names_conv1)  # BatchNormalization
    class_names_pool = MaxPooling2D(pool_size=(18, 1))(class_names_bn)
    class_names_flatten = Flatten()(class_names_pool)
    class_names_flatten = Dropout(0.5)(class_names_flatten)  # Dropout

    # 3) Method names (120 int values in array) - Processing by CNN
    method_names_input = Input(shape=(120,), name='input_method_names')
    method_names_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=EMBEDDING_DIM)(method_names_input)
    method_names_embedding_expanded = Reshape((120, EMBEDDING_DIM, 1))(method_names_embedding)
    method_names_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(method_names_embedding_expanded)
    method_names_bn = BatchNormalization()(method_names_conv1)  # BatchNormalization
    method_names_pool = MaxPooling2D(pool_size=(118, 1))(method_names_bn)
    method_names_flatten = Flatten()(method_names_pool)
    method_names_flatten = Dropout(0.5)(method_names_flatten)  # Dropout

    # 4) Parameter names (120 int values in array) - Processing by CNN
    parameter_names_input = Input(shape=(120,), name='input_parameter_names')
    parameter_names_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=EMBEDDING_DIM)(
        parameter_names_input)
    parameter_names_embedding_expanded = Reshape((120, EMBEDDING_DIM, 1))(parameter_names_embedding)
    parameter_names_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(
        parameter_names_embedding_expanded)
    parameter_names_bn = BatchNormalization()(parameter_names_conv1)  # BatchNormalization
    parameter_names_pool = MaxPooling2D(pool_size=(118, 1))(parameter_names_bn)
    parameter_names_flatten = Flatten()(parameter_names_pool)
    parameter_names_flatten = Dropout(0.5)(parameter_names_flatten)  # Dropout

    # 5) Variable names (200 int values in array) - Processing by CNN
    variable_names_input = Input(shape=(200,), name='input_variable_names')
    variable_names_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=EMBEDDING_DIM)(
        variable_names_input)
    variable_names_embedding_expanded = Reshape((200, EMBEDDING_DIM, 1))(variable_names_embedding)
    variable_names_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(
        variable_names_embedding_expanded)
    variable_names_bn = BatchNormalization()(variable_names_conv1)  # BatchNormalization
    variable_names_pool = MaxPooling2D(pool_size=(198, 1))(variable_names_bn)
    variable_names_flatten = Flatten()(variable_names_pool)
    variable_names_flatten = Dropout(0.5)(variable_names_flatten)  # Dropout

    # 6) API calls (300 int values in array) - Processing by CNN
    api_calls_input = Input(shape=(300,), name='input_api_calls')
    api_calls_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=EMBEDDING_DIM)(api_calls_input)
    api_calls_embedding_expanded = Reshape((300, EMBEDDING_DIM, 1))(api_calls_embedding)
    api_calls_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(api_calls_embedding_expanded)
    api_calls_bn = BatchNormalization()(api_calls_conv1)  # BatchNormalization
    api_calls_pool = MaxPooling2D(pool_size=(298, 1))(api_calls_bn)
    api_calls_flatten = Flatten()(api_calls_pool)
    api_calls_flatten = Dropout(0.5)(api_calls_flatten)  # Dropout

    # 7) Literals (500 int values in array) - Processing by CNN
    literals_input = Input(shape=(500,), name='input_literals')
    literals_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=128)(literals_input)
    literals_embedding_expanded = Reshape((500, 128, 1), name="literals_reshape")(literals_embedding)
    literals_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(literals_embedding_expanded)
    literals_bn = BatchNormalization()(literals_conv1)  # BatchNormalization
    literals_pool = MaxPooling2D(pool_size=(498, 1))(literals_bn)
    literals_flatten = Flatten()(literals_pool)
    literals_flatten = Dropout(0.5)(literals_flatten)  # Dropout

    # 8) Comments (750 int values in array) - Processing by CNN
    comments_input = Input(shape=(750,), name='input_comments')
    comments_embedding = Embedding(input_dim=MAX_TOKENS_VECTORIZATION, output_dim=128)(comments_input)
    comments_embedding_expanded = Reshape((750, 128, 1), name="comments_reshape")(comments_embedding)
    comments_conv1 = Conv2D(64, kernel_size=(3, EMBEDDING_DIM), activation='relu')(comments_embedding_expanded)
    comments_bn = BatchNormalization()(comments_conv1)  # BatchNormalization
    comments_pool = MaxPooling2D(pool_size=(748, 1))(comments_bn)
    comments_flatten = Flatten()(comments_pool)
    comments_flatten = Dropout(0.5)(comments_flatten)  # Dropout

    # 9) File modification frequency (1 int value) - directly connect to dense layer
    file_modification_freq_input = Input(shape=(1,), name='input_file_modification')

    # 10) Last modification value (1 double value) - directly connect to dense layer
    last_modification_input = Input(shape=(1,), name='input_last_modification')

    # Concatenate all layers
    concatenated = Concatenate()([
        bug_description_flatten,
        class_names_flatten,
        method_names_flatten,
        parameter_names_flatten,
        variable_names_flatten,
        api_calls_flatten,
        literals_flatten,
        comments_flatten,
    ])

    dense_1 = Dense(256, activation='relu')(concatenated)
    dense_1 = Dropout(0.5)(dense_1)  # Dropout

    final_concatenated = Concatenate()([dense_1, file_modification_freq_input, last_modification_input])

    # Final Layer
    output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1))(final_concatenated)

    # Create the model
    model = Model(inputs=[
        bug_description_input,
        class_names_input,
        method_names_input,
        parameter_names_input,
        variable_names_input,
        api_calls_input,
        literals_input,
        comments_input,
        file_modification_freq_input,
        last_modification_input
    ], outputs=output)

    # Print the model summary
    # model.summary()

    return model


model = create_model()


### Prepare dataset for training ###
def get_issue_text_from_dataframe(df, issue_number):
    result = df[df['number'] == int(issue_number)]['vectorized_issue_text']
    if not result.empty:
        return result.values[0]
    else:
        return None


def data_generator():
    # Connect to db
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Get train data
    cursor.execute(
        "SELECT issue_number, file_path, is_modified, last_modification, modification_freq, vectorized_class_names, vectorized_method_names, vectorized_variable_names, vectorized_parameter_names, vectorized_api_calls, vectorized_literals, vectorized_comments  FROM train_data")
    rows = cursor.fetchall()

    # Iterate all rows
    for row in rows:
        vectorized_issue_desc = get_issue_text_from_dataframe(issues_dataset, row[0])
        if vectorized_issue_desc is None:
            continue

        vectorized_class_names = json.loads(row[5]) if row[5] != '[]' else [0] * 20
        vectorized_method_names = json.loads(row[6]) if row[6] != '[]' else [0] * 120
        vectorized_variable_names = json.loads(row[7]) if row[7] != '[]' else [0] * 200
        vectorized_parameter_names = json.loads(row[8]) if row[8] != '[]' else [0] * 120
        vectorized_api_calls = json.loads(row[9]) if row[9] != '[]' else [0] * 300
        vectorized_literals = json.loads(row[10]) if row[10] != '[]' else [0] * 500
        vectorized_comments = json.loads(row[11]) if row[11] != '[]' else [0] * 750

        # order --> desc, class, method, param, var,api,literals,comments, modfreq,lastmod
        features = (
            vectorized_issue_desc,
            vectorized_class_names,
            vectorized_method_names,
            vectorized_parameter_names,
            vectorized_variable_names,
            vectorized_api_calls,
            vectorized_literals,
            vectorized_comments,
            row[4],
            row[3]
        )
        label = row[2]  # is_modified

        # Return features tuple and corresponding label
        yield features, label

    # Close the connection
    # conn.close()


# Build dataset
combined_dataset = tf.data.Dataset.from_generator(
    data_generator,
    # args=[json_files_train],
    output_signature=(
        (tf.TensorSpec(shape=(1000,), dtype=tf.int64),
         tf.TensorSpec(shape=(20,), dtype=tf.int64),
         tf.TensorSpec(shape=(120,), dtype=tf.int64),
         tf.TensorSpec(shape=(120,), dtype=tf.int64),
         tf.TensorSpec(shape=(200,), dtype=tf.int64),
         tf.TensorSpec(shape=(300,), dtype=tf.int64),
         tf.TensorSpec(shape=(500,), dtype=tf.int64),
         tf.TensorSpec(shape=(750,), dtype=tf.int64),
         tf.TensorSpec(shape=(), dtype=tf.int64),
         tf.TensorSpec(shape=(), dtype=tf.float32)
         ),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    )
)


### Train the model ###
combined_dataset_batched = combined_dataset.batch(32)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              )

history = model.fit(combined_dataset_batched, epochs=NUM_OF_EPOCHS)

# Save the  model
os.makedirs(f'../../saved_models/issue_localization/{result_folder_prefix}', exist_ok=True)
os.makedirs(f'../../evaluation_results/issue_localization/{result_folder_prefix}', exist_ok=True)
model.save(
    f'../../saved_models/issue_localization/{result_folder_prefix}/issue_localization_epochs_{NUM_OF_EPOCHS}.keras')


### Test the model ###

# JSON dosyasını okuma
with open(vocabulary_list_path, 'r', encoding='utf-8') as f:
    word_freq_dict = json.load(f)

vocab = list(word_freq_dict.keys())[:(MAX_TOKENS_VECTORIZATION - 2)]

### Vectorization - code file
vectorizer_class_names = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=20,
    vocabulary=vocab
)

vectorizer_method_names = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=120,
    vocabulary=vocab
)

vectorizer_parameter_names = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=120,
    vocabulary=vocab
)

vectorizer_variable_names = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=200,
    vocabulary=vocab
)

vectorizer_api_calls = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=300,
    vocabulary=vocab
)

vectorizer_literals = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=500,
    vocabulary=vocab
)

vectorizer_comments = TextVectorization(
    max_tokens=MAX_TOKENS_VECTORIZATION,
    standardize=None,
    output_mode='int',
    output_sequence_length=750,
    vocabulary=vocab
)

model = tf.keras.models.load_model(
    f'../../saved_models/issue_localization/{result_folder_prefix}/issue_localization_epochs_{NUM_OF_EPOCHS}.keras')

# Create DB Connection
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Counters for statistics
top_1_correct_predictions = 0
top_5_correct_predictions = 0
top_10_correct_predictions = 0
test_dataset_length = 0

# Get issue numbers
cursor.execute("SELECT DISTINCT issue_number FROM test_data ORDER BY issue_number")
unique_issues = [row[0] for row in cursor.fetchall()]
# print("unique_issues", len(unique_issues))

cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_issue_number ON test_data(issue_number)
""")

# Get corresponding entries for each issue number
for issue_number in unique_issues:
    # Skip if bug report does not exist
    vectorized_issue_desc = get_issue_text_from_dataframe(issues_dataset, int(issue_number))
    if vectorized_issue_desc is None:
        #print(issue_number, vectorized_issue_desc)
        continue

    # Get all records for given issue number
    cursor.execute("""
        SELECT * 
        FROM test_data
        WHERE issue_number = ?
    """, (issue_number,))
    rows = cursor.fetchall()

    # print(f"Issue: {issue_number}  ({test_dataset_length}/{top_1_correct_predictions}/{top_5_correct_predictions}/{top_10_correct_predictions}/)")

    # Modified list of an issue
    modified_files_list = [row[1] for row in rows if row[2] == 1]

    if len(modified_files_list) == 0:
        # print("No modified")
        continue

    # Preparing for batch processing
    class_names_batch = []
    method_names_batch = []
    variable_names_batch = []
    parameter_names_batch = []
    api_calls_batch = []
    literals_batch = []
    comments_batch = []
    file_paths = []

    for row in rows:
        file_paths.append(row[1])  # Store the file path

        class_names = row[6].split(',') if row[6] else []
        method_names = row[7].split(',') if row[7] else []
        variable_names = row[8].split(',') if row[8] else []
        parameter_names = row[9].split(',') if row[9] else []
        api_calls = row[10].split(',') if row[10] else []
        literals = json.loads(row[11]) if row[11] else []
        comments = json.loads(row[12]) if row[12] else []

        # preprocessing
        class_names = [preprocess_text(str(s)) for s in class_names]
        method_names = [preprocess_text(str(s)) for s in method_names]
        variable_names = [preprocess_text(str(s)) for s in variable_names]
        parameter_names = [preprocess_text(str(s)) for s in parameter_names]
        api_calls = [preprocess_text(str(s)) for s in api_calls]
        literals = [preprocess_text(str(s)) for s in literals]
        comments = [preprocess_text(str(s)) for s in comments]

        class_names = check_string(" ".join(class_names))
        method_names = check_string(" ".join(method_names))
        parameter_names = check_string(" ".join(parameter_names))
        variable_names = check_string(" ".join(variable_names))
        api_calls = check_string(" ".join(api_calls))
        literals = check_string(" ".join(literals))
        comments = check_string(" ".join(comments))

        # Add corresponding data to each batch list
        class_names_batch.append(class_names)
        method_names_batch.append(method_names)
        variable_names_batch.append(variable_names)
        parameter_names_batch.append(parameter_names)
        api_calls_batch.append(api_calls)
        literals_batch.append(literals)
        comments_batch.append(comments)

    # Vectorization of batch-arrays
    vectorized_class_names = vectorizer_class_names(tf.constant(class_names_batch))
    vectorized_method_names = vectorizer_method_names(tf.constant(method_names_batch))
    vectorized_variable_names = vectorizer_variable_names(tf.constant(variable_names_batch))
    vectorized_parameter_names = vectorizer_parameter_names(tf.constant(parameter_names_batch))
    vectorized_api_calls = vectorizer_api_calls(tf.constant(api_calls_batch))
    vectorized_literals = vectorizer_literals(tf.constant(literals_batch))
    vectorized_comments = vectorizer_comments(tf.constant(comments_batch))

    # order --> desc, class, method, param, var, api, literals, comments, modfreq, lastmod
    feature_list = []
    for i, row in enumerate(rows):
        features = [
            vectorized_issue_desc,
            vectorized_class_names[i],
            vectorized_method_names[i],
            vectorized_parameter_names[i],
            vectorized_variable_names[i],
            vectorized_api_calls[i],
            vectorized_literals[i],
            vectorized_comments[i],
            [row[4]],  # modification_freq
            [row[3]]  # last_modification
        ]
        feature_list.append(features)

    # Convert to tensor
    feature_array = [tf.convert_to_tensor(np.array(f), dtype=tf.float32) for f in zip(*feature_list)]

    # Predict - batch
    prediction_scores = model.predict(feature_array)

    # Compare predicted files with real modified files
    prediction_scores_dict = {file_path: score[0] for file_path, score in zip(file_paths, prediction_scores)}

    # Sort the prediction scores
    sorted_prediction_scores = sorted(prediction_scores_dict.items(), key=lambda item: item[1], reverse=True)

    # Get top-n records
    top_1_result = sorted_prediction_scores[:1]
    top_5_results = sorted_prediction_scores[:5]
    top_10_results = sorted_prediction_scores[:10]

    # File names of Top-n records
    predicted_files_top_1 = [key for key, value in top_1_result]
    predicted_files_top_5 = [key for key, value in top_5_results]
    predicted_files_top_10 = [key for key, value in top_10_results]

    # Calculate accuracy
    test_dataset_length += 1
    if set(modified_files_list) & set(predicted_files_top_1):  # & --> kesişim kontrolü yapar
        top_1_correct_predictions += 1

    if set(modified_files_list) & set(predicted_files_top_5):  # & --> kesişim kontrolü yapar
        top_5_correct_predictions += 1

    if set(modified_files_list) & set(predicted_files_top_10):  # & --> kesişim kontrolü yapar
        top_10_correct_predictions += 1

    # First 10 file names
    # for file_name, score in top_10_results:
    #    print(f"{file_name}: {score}")
    # print("Modified list:", modified_files_list)

# Close connection
conn.close()

# Print results
top_1_percent = top_1_correct_predictions / test_dataset_length
top_5_percent = top_5_correct_predictions / test_dataset_length
top_10_percent = top_10_correct_predictions / test_dataset_length
print('------------------------------------------------------------------------')
print('Scores:')
print(f'> Accuracy-Top-1: {np.mean(top_1_percent):.4f}')
print(f'> Accuracy-Top-5: {np.mean(top_5_percent):.4f}')
print(f'> Accuracy-Top-10: {np.mean(top_10_percent):.4f}')
print('------------------------------------------------------------------------')

# Save evaluation results
scores_df = pd.DataFrame(data={
    "Top_1": [top_1_percent],
    "Top_5": [top_5_percent],
    "Top_10": [top_10_percent],
})

# Export als CSV
scores_df.to_csv(
    f'../../evaluation_results/issue_localization/{result_folder_prefix}/issue_localization_epochs_{NUM_OF_EPOCHS}.csv',
    index=False)
