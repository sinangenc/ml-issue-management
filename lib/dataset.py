import json
import os
import re
from collections import Counter

import pandas as pd


def load_json_data(json_file_path):
    if not os.path.exists(json_file_path):
        raise Exception(f"JSON File does not exist")

    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
            return json_data
    except json.JSONDecodeError:
        print(f"{json_file_path} can not be decoded.")


def create_repo_path(owner, repository):
    # Remove invalid characters
    def sanitize_input(input_str):
        sanitized_str = re.sub(r'[^a-zA-Z0-9_-]', '_', input_str)
        return sanitized_str

    # Sanitize owner and repo name
    clean_owner = sanitize_input(owner)
    clean_repository = sanitize_input(repository)

    # Build path
    repo_path = os.path.join('..', '..', 'data', f'{clean_owner}_{clean_repository}')

    return repo_path


def get_contributors_for_given_issue(issue_number, owner, repository):
    repo_path = create_repo_path(owner, repository)
    issues_related_prs_json_file_path = os.path.join(repo_path, 'issue_related_pull_requests.json')
    pr_developers_json_file_path = os.path.join(repo_path, 'pull_request_developers.json')
    # pr_developers_json_file_path = os.path.join(repo_path, 'pull_request_developers_combined.json')

    issues_related_prs_json_data = load_json_data(issues_related_prs_json_file_path)
    pr_developers_json_data = load_json_data(pr_developers_json_file_path)

    pr_list = issues_related_prs_json_data.get(str(issue_number))
    if pr_list is not None:
        dev_list = set()
        for pr in pr_list:
            pr_dev_list = pr_developers_json_data.get(str(pr))
            dev_list.update(pr_dev_list)

        return list(dev_list)

    return None


def load_issues_dataset_from_json(owner, repository):
    repo_path = create_repo_path(owner, repository)
    issues_json_file_path = os.path.join(repo_path, 'issues.json')
    issues_related_prs_json_file_path = os.path.join(repo_path, 'issue_related_pull_requests.json')

    issues_json_data = load_json_data(issues_json_file_path)
    issues_related_prs_json_data = load_json_data(issues_related_prs_json_file_path)

    data = {'number': [issue["number"] for issue in issues_json_data],
            'type': ["pull_request" if "pull_request" in issue else "issue" for issue in issues_json_data],
            'title': [issue["title"] for issue in issues_json_data],
            'body': [issue["body"] for issue in issues_json_data],
            'state': [issue["state"] for issue in issues_json_data],
            'state_reason': [issue["state_reason"] for issue in issues_json_data],
            # 'comments': [issue["comments"] for issue in json_data],
            'labels': [[] if len(issue["labels"]) == 0 else [label["name"] for label in issue["labels"]] for issue in
                       issues_json_data],
            'assignee': [None if issue["assignee"] is None else issue["assignee"]["login"] for issue in
                         issues_json_data],
            'assignees': [
                None if len(issue["assignees"]) == 0 else [assignee["login"] for assignee in issue["assignees"]] for
                issue in issues_json_data],
            'closing_prs': [issues_related_prs_json_data.get(str(issue["number"])) for issue in issues_json_data],
            'fixers': [get_contributors_for_given_issue(issue["number"], owner, repository)
                                for issue in issues_json_data],
            'created_at': [issue["created_at"] for issue in issues_json_data],
            'closed_at': [issue["closed_at"] for issue in issues_json_data]}

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create new column from concatenation of title and body
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    # Sort by closing date
    df.sort_values(by=['closed_at'], ignore_index=True, inplace=True)

    return df


def filter_dataset(df):
    # Return only issues which are closed(as completed) and have at least 1 contributor
    df = df[df["type"] == "issue"]
    df = df[df["state"] == "closed"]
    df = df[df["state_reason"] == "completed"]  # exclude 'not planned'
    df = df[df['closing_prs'].notna()]

    return df


def exclude_short_texts(df, min_word_count=10):
    return df[df['text'].apply(lambda x: len(x.split()) >= min_word_count)]


def filter_dataset_by_developer_commit_counts(df, threshold=10):
    # Get all devs
    all_devs = [fixer for sublist in df['fixers'] for fixer in sublist]

    # Calculate frequency of each dev
    dev_commit_counts = Counter(all_devs)

    # Define filter function to exclude developers who committed <= 30
    def filter_devs(devs):
        return [dev for dev in devs if dev_commit_counts[dev] >= threshold]

    # Apply filter to dataset
    df['fixers'] = df['fixers'].apply(filter_devs)

    # Remove rows which have no developer in fixers column after filtering
    df = df[df['fixers'].map(len) > 0]

    return df


def load_issues_dataset(owner, repository):
    dataset = load_issues_dataset_from_json(owner, repository)

    dataset = filter_dataset(dataset)

    return dataset


def load_files_dataset(owner, repository):
    repo_base_path = create_repo_path(owner, repository)
    json_files_path = os.path.join(repo_base_path, 'processed')


    files_dataset = []

    # traverse all files in the directory
    for file_name in os.listdir(json_files_path):
        # Filter .json files
        if file_name.endswith('.json'):
            json_file_path = os.path.join(json_files_path, file_name)
            # Read file
            data = load_json_data(json_file_path)

            filtered_data = [{"issue_number": record["issue_number"],
                              "file_name": record["file_name"],
                              "vectorized_file_content": record["vectorized_file_content"],
                              "is_positive_sample": record["is_positive_sample"],
                              } for record in data]

            files_dataset.extend(filtered_data)

    return files_dataset


# Eclipse dataset specific functions
#####################################
def load_issues_dataset_eclipse(file_path):
    df = pd.read_csv(file_path)

    # Drop rows where 'bug_id' is null
    df = df[df['bug_id'].notnull()]

    # Rename columns
    df.rename(columns={'bug_id': 'number', 'Assignee': 'fixers', 'Description': 'text', 'creation_ts': 'created_at'}, inplace=True)

    # Convert 'number' column to integer
    df['number'] = df['number'].astype(int)

    # Wrap 'fixers' column values in a list
    df['fixers'] = df['fixers'].apply(lambda x: [x])

    # Apply the function to create 'labels' and clean 'text'
    df['labels'], df['text'] = zip(*df['text'].apply(extract_labels_from_text_column_in_jdt_dataset))

    # Sort by  date
    df.sort_values(by=['created_at'], ascending=True, ignore_index=True, inplace=True)

    return df


def extract_labels_from_text_column_in_jdt_dataset(text):
    labels = []
    new_label = re.findall(r'^\[([^\]]+)\]', text)  # Remove leading [...content...] s
    while new_label:
        labels.append(new_label[0])
        text = re.sub(r'^\[([^\]]+)\]', '', text)
        new_label = re.findall(r'^\[([^\]]+)\]', text)

    #labels = []
    return labels, text

def check_string(input_string):
    return input_string if input_string else ">>"