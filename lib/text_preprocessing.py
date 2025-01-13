import re
import string
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


def sanitize_input(input_str):
    sanitized_str = re.sub(r'[^a-zA-Z0-9_-]', '_', input_str)
    return sanitized_str


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def unique_items_in_order(original_list):
    unique_list = []
    for item in original_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def split_camel_case(s):
    # Split string at position where a lowercase followed by a uppercase
    split_str = re.findall('.+?(?:(?<=[a-z])(?=[A-Z])|$)', s)
    # Append original string to the list
    split_str.append(s)
    return unique_items_in_order(split_str)


def flatten(array2d):
    return [x for array1d in array2d for x in array1d]


def flatten_and_check_length(array2d):
    words = [x for array1d in array2d for x in array1d]
    words = [w for w in words if len(w) > 1]
    return words


def preprocess_text(text):
    # print(text)
    # Remove punctuation
    text = remove_punctuation(text)

    # Tokenization - Split words
    words = text.split()

    # Split camel-case words
    words = flatten([split_camel_case(words) for words in words])

    # Convert all letters to lowercase
    words = [word.lower() for word in words]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]

    # Remove elements that consist only of numbers
    filtered_list = [element for element in lemmatized_text if not element.isdigit()]

    return " ".join(filtered_list)
