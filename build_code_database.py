import csv
import json
import os
import random
import sqlite3
import git
from datetime import datetime
from collections import Counter
from lib.java_parser import *
from lib.text_preprocessing import preprocess_text, flatten_and_check_length
from tensorflow.keras.layers import TextVectorization


def create_repo_path(owner, repository):
    # Remove invalid characters
    def sanitize_input(input_str):
        sanitized_str = re.sub(r'[^a-zA-Z0-9_-]', '_', input_str)
        return sanitized_str

    # Sanitize owner and repo name
    clean_owner = sanitize_input(owner)
    clean_repository = sanitize_input(repository)

    # Build path
    repo_path = os.path.join('.', 'data', f'{clean_owner}_{clean_repository}')

    return repo_path


def load_json_data(json_file_path):
    if not os.path.exists(json_file_path):
        raise Exception(f"JSON File doesn't exist!")

    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
            return json_data
    except json.JSONDecodeError:
        print(f"{json_file_path} can't be decoded.")


def clone_repository(owner, repository):
    repo_url = f"https://github.com/{owner}/{repository}.git"
    local_path = os.path.join(create_repo_path(owner, repository), 'repo')
    try:
        print(f"Cloning: {owner}/{repository}")
        repo = git.Repo.clone_from(repo_url, local_path)
        print(f"{owner}/{repository} cloned to: {local_path}")
        return repo
    except Exception as e:
        print(f"An error occured: {e}")
        return None


def create_db_issue_localization(owner, repository):
    repo_base_path = create_repo_path(owner, repository)
    db_path = os.path.join(repo_base_path, 'issue_localization_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS train_data (
            issue_number TEXT,
            file_path TEXT,
            is_modified INTEGER,
            last_modification REAL,
            modification_freq INTEGER,
            similarity REAL,
            class_names TEXT,
            method_names TEXT,
            variable_names TEXT,
            parameter_names TEXT,
            api_calls TEXT,
            literals TEXT,
            comments TEXT,
            vectorized_class_names TEXT,
            vectorized_method_names TEXT,
            vectorized_variable_names TEXT,
            vectorized_parameter_names TEXT,
            vectorized_api_calls TEXT,
            vectorized_literals TEXT,
            vectorized_comments TEXT
        )
        """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS train_check 
        ON train_data(issue_number, file_path)
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_data (
            issue_number TEXT,
            file_path TEXT,
            is_modified INTEGER,
            last_modification REAL,
            modification_freq INTEGER,
            class_names TEXT,
            method_names TEXT,
            variable_names TEXT,
            parameter_names TEXT,
            api_calls TEXT,
            literals TEXT,
            comments TEXT,
            vectorized_class_names TEXT,
            vectorized_method_names TEXT,
            vectorized_variable_names TEXT,
            vectorized_parameter_names TEXT,
            vectorized_api_calls TEXT,
            vectorized_literals TEXT,
            vectorized_comments TEXT
        )
        """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS test_check 
        ON test_data(issue_number, file_path)
        """)

    # Create a table to store each file's modification history
    cursor.execute('''CREATE TABLE IF NOT EXISTS file_change_history (
                        file_path TEXT,
                        modified_at TEXT)''')

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS history_query
        ON file_change_history (file_path, modified_at);
        """)

    conn.commit()
    conn.close()

    print("sqlite DB created successfully")


def populate_file_modification_table_for_issue_localization(owner, repository):
    repo_base_path = create_repo_path(owner, repository)
    db_path = os.path.join(repo_base_path, 'issue_localization_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    repository_path = os.path.join(repo_base_path, 'repo')

    # Open the Git repository
    repo = git.Repo(repository_path)

    all_commits = list(repo.iter_commits())  # Convert commits to a list
    print(f"Populating file modification table. Total number of commits: {len(all_commits)}")

    # Iterate through all commits
    i = 1
    for commit in all_commits:
        commit_time = datetime.fromtimestamp(commit.committed_date).strftime('%Y-%m-%d %H:%M:%S')

        # Get the list of files changed in the commit
        for file_path in commit.stats.files:
            cursor.execute("INSERT INTO file_change_history (file_path, modified_at) VALUES (?, ?)",
                           (file_path, commit_time))
        #print(i)
        i += 1

    # Save changes to the database and close the connection
    conn.commit()
    conn.close()

    print("File modification table populated successfully")


def checkout_commit(repo_path, commit_sha):
    # Open repo
    repo = git.Repo(repo_path)

    # Reset any changes in the working directory
    repo.git.reset('--hard')

    # Checkout the given commit SHA
    repo.git.checkout(commit_sha)


def read_code_files(repo_path, file_list):
    files = []
    contents = []
    result = dict()
    for file_name in file_list:
        try:  # To prevent errors for unreadable files like .jar, .png
            with open(os.path.join(repo_path, file_name), 'r') as file:
                # contents.append(file.read())
                # files.append(file_name)
                result[file_name] = file.read()
        except Exception as e:
            #print(f"Error reading {file_name}")
            pass

    # return files, contents
    return result


def list_all_files_filtered(repo_path, commit_sha):
    # Open repo
    repo = git.Repo(repo_path)

    # Get the given commit
    commit = repo.commit(commit_sha)

    # Get all files in the repository at that commit
    files = [item.path for item in commit.tree.traverse() if item.type == 'blob']

    # List of file extensions to filter
    filter_extensions = ['java']
    # ['java', 'html', 'js', 'xml', 'yaml', 'properties', 'ts', 'txt',
    # 'tpl', 'json', 'gradle','css', 'scss', 'sh', 'scala', 'kt', 'yml']

    # Filter files by their extensions
    filtered_paths = [path for path in files if path.split('.')[-1] in filter_extensions]

    return filtered_paths


def list_changed_files_filtered(repo_path, commit_sha):
    # Open repo
    repo = git.Repo(repo_path)

    # Get the given commit
    commit = repo.commit(commit_sha)

    # Get the previous commit (if not the first commit)
    parents = commit.parents
    if len(parents) == 0:
        print(f"Commit {commit_sha} is the inital commit, no prev. commit to compare")
        return []

    parent_commit = parents[0]

    # Get the changed files
    diffs = commit.diff(parent_commit)
    changed_files = [diff.a_path for diff in diffs]

    # Filter files by their extensions
    filter_extensions = ['java']
    filtered_paths = [path for path in changed_files if path.split('.')[-1] in filter_extensions]

    return filtered_paths


def split_issue_localization_dataset(input_dict, split_ratio=0.8):
    keys = list(input_dict.keys())
    random.shuffle(keys)
    split_index = int(len(keys) * split_ratio)

    # %80
    train_dict = {key: input_dict[key] for key in keys[:split_index]}

    # %20
    test_dict = {key: input_dict[key] for key in keys[split_index:]}

    return train_dict, test_dict


def get_file_change_stats_from_db(cursor, file_path, iso_date):
    # Convert the date to '%Y-%m-%d %H:%M:%S' format
    issue_date_obj = datetime.strptime(iso_date, '%Y-%m-%dT%H:%M:%SZ')
    formatted_date = issue_date_obj.strftime('%Y-%m-%d %H:%M:%S')

    # 1. Find how many times the file was modified before the given date
    cursor.execute("SELECT COUNT(*) FROM file_change_history WHERE file_path=? AND modified_at < ?",
                   (file_path, formatted_date))
    change_count = cursor.fetchone()[0]

    # 2. Find the last modification date of the file before the given date
    cursor.execute("SELECT MAX(modified_at) FROM file_change_history WHERE file_path=? AND modified_at < ?",
                   (file_path, formatted_date))
    last_modified_str = cursor.fetchone()[0]

    if last_modified_str:
        # Convert to datetime object
        last_modified_obj = datetime.strptime(last_modified_str, '%Y-%m-%d %H:%M:%S')

        # Calculate the difference between months
        issue_month_total = issue_date_obj.year * 12 + issue_date_obj.month
        last_commit_month_total = last_modified_obj.year * 12 + last_modified_obj.month

        month_diff = issue_month_total - last_commit_month_total

        # Calculate score
        score = 1 / (month_diff + 1)
    else:
        # When the file has not yet been modified
        score = 1

    return score, change_count


def populate_db_issue_localization(owner, repository):
    repo_base_path = create_repo_path(owner, repository)
    db_path = os.path.join(repo_base_path, 'issue_localization_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Split dataset
    repo_base_path = create_repo_path(owner, repository)
    repository_path = os.path.join(repo_base_path, 'repo')

    pull_request_shas_json_file_path = os.path.join(repo_base_path, 'pull_request_shas.json')
    issues_related_prs_json_file_path = os.path.join(repo_base_path, 'issue_related_pull_requests.json')
    issues_json_file_path = os.path.join(repo_base_path, 'issues.json')

    pull_request_shas_json_data = load_json_data(pull_request_shas_json_file_path)
    issues_related_prs_json_data = load_json_data(issues_related_prs_json_file_path)
    issues_json_data = load_json_data(issues_json_file_path)

    issues_train, issues_test = split_issue_localization_dataset(issues_related_prs_json_data)

    # Populate train_data table
    i=0
    for issue_number, pr_list in issues_train.items():
        #i += 1
        #if i % 50 == 0:
        #    break

        issue_text = ""
        all_files = list()
        modified_files_with_content = dict()

        for issue in issues_json_data:
            if issue["number"] == int(issue_number):
                issue_text = issue["title"] + " " + (issue["body"] if issue["body"] is not None else "")
                issue_closed_at = issue["closed_at"] if issue["closed_at"] is not None else issue["updated_at"]

        for pr_number in pr_list:
            try:
                merge_commit_sha = pull_request_shas_json_data[pr_number].get("merge_commit_sha")

                # Roll the repo back to merge-commit
                checkout_commit(repository_path, merge_commit_sha)

                if len(all_files) == 0:
                    # Get the list of all files
                    all_files = list_all_files_filtered(repository_path, merge_commit_sha)

                # Get the list of modified files
                modified_files_via_pr = list_changed_files_filtered(repository_path, merge_commit_sha)

                # Rollback to pre-commit version
                checkout_commit(repository_path, merge_commit_sha + "~1")

                modified_files_by_issue_with_content = read_code_files(repository_path, modified_files_via_pr)
                modified_files_with_content.update(modified_files_by_issue_with_content)

                # print(issue_number, pr_number)
            except:
                print("Cannot rollback to sha: " + str(issue_number))
                pass

        # Skip issue if no java file was changed
        if len(modified_files_with_content) == 0:
            continue

        # Get the set of unmodified files
        unmodified_files = list(set(all_files) - set(modified_files_with_content.keys()))

        # Find the unmodified and  least-similar files

        # Select 250 random files
        random_selected_unmodified_files = random.sample(unmodified_files, 250)
        unmodified_files_with_content = read_code_files(repository_path, random_selected_unmodified_files)

        # Positive Samples
        for file_name, file_content in modified_files_with_content.items():
            # print(file_name, last_modification_list.get(file_name))

            try:
                last_modification, modification_freq = get_file_change_stats_from_db(cursor, file_name, issue_closed_at)
                # Parse file
                parsed_data = parse_java_file(file_content)

                class_names = ",".join(parsed_data.get("class_or_interface_names"))
                method_names = ",".join(parsed_data.get("method_names"))
                parameter_names = ",".join(parsed_data.get("parameter_names"))
                variable_names = ",".join(parsed_data.get("variable_names"))
                api_calls = ",".join(parsed_data.get("api_calls"))
                literals = json.dumps(parsed_data.get("literals"))
                comments = json.dumps(parsed_data.get("comments"))

                # Insert to train_data table
                cursor.execute("""
            INSERT INTO train_data (issue_number, file_path, is_modified, last_modification, modification_freq, similarity, class_names, method_names, variable_names, parameter_names, api_calls, literals, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (issue_number, file_name, 1, last_modification, modification_freq, 1, class_names,
              method_names, variable_names, parameter_names, api_calls, literals, comments))
            except:
                pass

        # Negative Samples
        for file_name, file_content in unmodified_files_with_content.items():
            # print(curr[1], last_modification_list.get(curr[1]), modification_freq_list.get(curr[1]))

            try:
                last_modification, modification_freq = get_file_change_stats_from_db(cursor, file_name, issue_closed_at)
                # Parse file
                parsed_data = parse_java_file(file_content)

                class_names = ",".join(parsed_data.get("class_or_interface_names"))
                method_names = ",".join(parsed_data.get("method_names"))
                parameter_names = ",".join(parsed_data.get("parameter_names"))
                variable_names = ",".join(parsed_data.get("variable_names"))
                api_calls = ",".join(parsed_data.get("api_calls"))
                literals = json.dumps(parsed_data.get("literals"))
                comments = json.dumps(parsed_data.get("comments"))

                # Insert to train_data table
                cursor.execute("""
            INSERT INTO train_data (issue_number, file_path, is_modified, last_modification, modification_freq, similarity, class_names, method_names, variable_names, parameter_names, api_calls, literals, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (issue_number, file_name, 0, last_modification, modification_freq,
              0, class_names, method_names, variable_names,
              parameter_names, api_calls, literals, comments))

            except:
                pass

        conn.commit()
        #print("Finished train data: " + str(issue_number))

    # Populate test_data table
    i = 0
    for issue_number, pr_list in issues_test.items():
        #i += 1
        #if i % 50 == 0:
        #    break

        for issue in issues_json_data:
            if issue["number"] == int(issue_number):
                issue_date = issue["created_at"]

        all_files_with_content = dict()
        modified_files = list()
        for pr_number in pr_list:
            try:
                merge_commit_sha = pull_request_shas_json_data[pr_number].get("merge_commit_sha")

                # Roll the repo back to  merge-commit
                checkout_commit(repository_path, merge_commit_sha)

                if len(all_files_with_content) == 0:
                    # Get the list of all files
                    all_files = list_all_files_filtered(repository_path, merge_commit_sha)
                    all_files_with_content = read_code_files(repository_path, all_files)

                # Get the list of modified files
                modified_files_via_pr = list_changed_files_filtered(repository_path, merge_commit_sha)
                modified_files.extend(modified_files_via_pr)

                # Rollback to pre-commit version
                # checkout_commit(repository_path, merge_commit_sha+"~1") # kaldirilabilir?

                modified_files_by_issue_with_content = read_code_files(repository_path, modified_files_via_pr)
                all_files_with_content.update(modified_files_by_issue_with_content)
            except:
                pass

        i = 0
        for file_name, file_content in all_files_with_content.items():
            i += 1
            # print(i, len(all_files_with_content))
            try:
                last_modification, modification_freq = get_file_change_stats_from_db(cursor, file_name, issue_date)

                is_modified = 1 if file_name in modified_files else 0
                # Parse file
                parsed_data = parse_java_file(file_content)

                class_names = ",".join(parsed_data.get("class_or_interface_names"))
                method_names = ",".join(parsed_data.get("method_names"))
                parameter_names = ",".join(parsed_data.get("parameter_names"))
                variable_names = ",".join(parsed_data.get("variable_names"))
                api_calls = ",".join(parsed_data.get("api_calls"))
                literals = json.dumps(parsed_data.get("literals"))
                comments = json.dumps(parsed_data.get("comments"))

                # Insert to test_data table
                cursor.execute("""
            INSERT INTO test_data (issue_number, file_path, is_modified, last_modification, modification_freq, class_names, method_names, variable_names, parameter_names, api_calls, literals, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (issue_number, file_name, is_modified, last_modification, modification_freq, class_names,
              method_names, variable_names, parameter_names, api_calls, literals, comments))
            except:
                pass

        conn.commit()
        #print("Finished test data: " + str(issue_number))

    conn.close()
    print("Issue localization DB populated successfully")


def populate_db_issue_localization_from_txt(owner, repository):
    repo_base_path = create_repo_path(owner, repository)
    db_path = os.path.join(repo_base_path, 'issue_localization_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Split dataset
    repo_base_path = create_repo_path(owner, repository)
    repository_path = os.path.join(repo_base_path, 'repo')

    txt_data_path = os.path.join(repo_base_path, 'raw_data.txt')

    # Open and read the TSV file
    with open(txt_data_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        next(reader)  # Skip first line
        rows = list(reader)

        # Dataset format : ['id', 'bug_id', 'summary', 'description', 'report_time',
        # 'report_timestamp', 'status', 'commit', 'commit_timestamp', 'files', '']

        # Listeyi karıştır
        random.shuffle(rows)

        # Split the data
        split_index = int(len(rows) * 0.8)
        issues_train = rows[:split_index]
        issues_test = rows[split_index:]

        # Populate train_data table
        for row in issues_train:
            issue_number = row[1]
            issue_text = row[2] + " " + row[3]
            issue_closed_at = datetime.utcfromtimestamp(int(row[8])).strftime('%Y-%m-%dT%H:%M:%SZ')

            try:
                merge_commit_sha = row[7]

                # Rollback to pre-commit version
                checkout_commit(repository_path, merge_commit_sha + "~1")

                # Get the list of all files
                all_files_list = list_all_files_filtered(repository_path, merge_commit_sha + "~1")

                # Get the list of modified files
                modified_files_list = list_changed_files_filtered(repository_path, merge_commit_sha)

            except:
                print("Cannot rollback to sha: " + issue_number)
                pass

            # Skip issue if no java file was changed
            if len(modified_files_list) == 0:
                continue

            # Get the set of unmodified files
            unmodified_files = list(set(all_files_list) - set(modified_files_list))

            # Find the unmodified and  least-similar files
            # Select 250 random files
            random_selected_unmodified_files = random.sample(unmodified_files, 250)

            # Read all unmodified files
            train_data_files_with_content = read_code_files(repository_path,
                                                            random_selected_unmodified_files + modified_files_list)

            # Positive + Negative Samples
            for file_name, file_content in train_data_files_with_content.items():
                try:
                    is_modified = 1 if file_name in modified_files_list else 0

                    last_modification, modification_freq = get_file_change_stats_from_db(cursor, file_name,
                                                                                         issue_closed_at)
                    # Parse file
                    parsed_data = parse_java_file(file_content)

                    class_names = ",".join(parsed_data.get("class_or_interface_names"))
                    method_names = ",".join(parsed_data.get("method_names"))
                    parameter_names = ",".join(parsed_data.get("parameter_names"))
                    variable_names = ",".join(parsed_data.get("variable_names"))
                    api_calls = ",".join(parsed_data.get("api_calls"))
                    literals = json.dumps(parsed_data.get("literals"))
                    comments = json.dumps(parsed_data.get("comments"))

                    # Insert to train_data table
                    cursor.execute("""
                INSERT INTO train_data (issue_number, file_path, is_modified, last_modification, modification_freq, similarity, class_names, method_names, variable_names, parameter_names, api_calls, literals, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (issue_number, file_name, is_modified, last_modification, modification_freq, 0, class_names,
                  method_names, variable_names, parameter_names, api_calls, literals, comments))
                except:
                    pass

            conn.commit()
            #print("Finished train data: " + str(issue_number))

        # Populate test_data table
        for row in issues_test:
            issue_number = row[1]
            issue_closed_at = datetime.utcfromtimestamp(int(row[8])).strftime('%Y-%m-%dT%H:%M:%SZ')

            try:
                merge_commit_sha = row[7]

                # Roll the repo back to pre merge-commit
                checkout_commit(repository_path, merge_commit_sha + "~1")

                # Get the list of all files
                all_files = list_all_files_filtered(repository_path, merge_commit_sha + "~1")
                all_files_with_content = read_code_files(repository_path, all_files)

                # Get the list of modified files
                modified_files_list = list_changed_files_filtered(repository_path, merge_commit_sha)
            except:
                pass

            # Skip issue if no java file was changed
            if len(modified_files_list) == 0:
                continue

            for file_name, file_content in all_files_with_content.items():
                try:
                    is_modified = 1 if file_name in modified_files_list else 0
                    last_modification, modification_freq = get_file_change_stats_from_db(cursor, file_name,
                                                                                         issue_closed_at)
                    # Parse file
                    parsed_data = parse_java_file(file_content)

                    class_names = ",".join(parsed_data.get("class_or_interface_names"))
                    method_names = ",".join(parsed_data.get("method_names"))
                    parameter_names = ",".join(parsed_data.get("parameter_names"))
                    variable_names = ",".join(parsed_data.get("variable_names"))
                    api_calls = ",".join(parsed_data.get("api_calls"))
                    literals = json.dumps(parsed_data.get("literals"))
                    comments = json.dumps(parsed_data.get("comments"))

                    # Insert to test_data table
                    cursor.execute("""
                INSERT INTO test_data (issue_number, file_path, is_modified, last_modification, modification_freq, class_names, method_names, variable_names, parameter_names, api_calls, literals, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (issue_number, file_name, is_modified, last_modification, modification_freq, class_names,
                  method_names, variable_names, parameter_names, api_calls, literals, comments))
                except:
                    pass

            conn.commit()
            #print("Finished test data: " + str(issue_number))

    conn.close()
    print("Issue localization DB populated successfully")
    return ""


def create_vocabulary_list_from_db(owner, repository):
    print("Creating vocabulary list...")
    code_tokenizer_frequency = Counter()

    repo_base_path = create_repo_path(owner, repository)
    db_path = os.path.join(repo_base_path, 'issue_localization_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all record from train table
    cursor.execute("SELECT * FROM train_data")
    rows = cursor.fetchall()
    for row in rows:
        # If the column not empty separate by ','
        class_names = row[6].split(',') if row[6] else []
        method_names = row[7].split(',') if row[7] else []
        variable_names = row[8].split(',') if row[8] else []
        parameter_names = row[9].split(',') if row[9] else []
        api_calls = row[10].split(',') if row[10] else []

        literals = json.loads(row[11]) if row[11] else []
        comments = json.loads(row[12]) if row[12] else []

        # preprocessing
        class_names = [preprocess_text(s) for s in class_names]
        method_names = [preprocess_text(s) for s in method_names]
        variable_names = [preprocess_text(s) for s in variable_names]
        parameter_names = [preprocess_text(s) for s in parameter_names]
        api_calls = [preprocess_text(s) for s in api_calls]
        literals = [preprocess_text(s) for s in literals]
        comments = [preprocess_text(s) for s in comments]

        #
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in class_names]))
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in method_names]))
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in variable_names]))
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in parameter_names]))
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in api_calls]))
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in literals]))
        code_tokenizer_frequency.update(flatten_and_check_length([s.split(" ") for s in comments]))

    conn.close()

    # Order by frequency
    sorted_frequency = dict(code_tokenizer_frequency.most_common())

    with open(os.path.join(repo_base_path, 'vocabulary_list.json'), 'w', encoding='utf-8') as f:
        json.dump(sorted_frequency, f, ensure_ascii=False, indent=4)

    print("Vocabulary list created.")
    return sorted_frequency


def vectorize_code_files(owner, repository):
    repo_base_path = create_repo_path(owner, repository)
    # Read json file
    with open(os.path.join(repo_base_path, 'vocabulary_list.json'), 'r', encoding='utf-8') as f:
        word_freq_dict = json.load(f)

    # MAX_TOKENS_VECTORIZATION = len(word_freq_dict) + 2
    MAX_TOKENS_VECTORIZATION = 75000  # For jdt: 180000
    vocab = list(word_freq_dict.keys())[:(MAX_TOKENS_VECTORIZATION - 2)]

    ### Vectorization
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

    repo_base_path = create_repo_path(owner, repository)
    db_path = os.path.join(repo_base_path, 'issue_localization_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()


    # Get all data from train_data table
    cursor.execute("SELECT * FROM train_data WHERE vectorized_variable_names is null")
    rows = cursor.fetchall()
    i = 0
    for row in rows:
        #print("Train Table: " + str(i))
        i += 1
        # Split by ',', when the column is not empty
        class_names = row[6].split(',') if row[6] else []
        method_names = row[7].split(',') if row[7] else []
        variable_names = row[8].split(',') if row[8] else []
        parameter_names = row[9].split(',') if row[9] else []
        api_calls = row[10].split(',') if row[10] else []

        literals = json.loads(row[11]) if row[11] else []
        comments = json.loads(row[12]) if row[12] else []

        # preprocessing
        class_names = [preprocess_text(s) for s in class_names]
        method_names = [preprocess_text(s) for s in method_names]
        variable_names = [preprocess_text(s) for s in variable_names]
        parameter_names = [preprocess_text(s) for s in parameter_names]
        api_calls = [preprocess_text(s) for s in api_calls]
        literals = [preprocess_text(s) for s in literals]
        comments = [preprocess_text(s) for s in comments]

        class_names = check_string(" ".join(class_names))
        method_names = check_string(" ".join(method_names))
        parameter_names = check_string(" ".join(parameter_names))
        variable_names = check_string(" ".join(variable_names))
        api_calls = check_string(" ".join(api_calls))
        literals = check_string(" ".join(literals))
        comments = check_string(" ".join(comments))

        vectorized_class_names = json.dumps(vectorizer_class_names(class_names).numpy().tolist())
        vectorized_method_names = json.dumps(vectorizer_method_names(method_names).numpy().tolist())
        vectorized_variable_names = json.dumps(vectorizer_variable_names(variable_names).numpy().tolist())
        vectorized_parameter_names = json.dumps(vectorizer_parameter_names(parameter_names).numpy().tolist())
        vectorized_api_calls = json.dumps(vectorizer_api_calls(api_calls).numpy().tolist())
        vectorized_literals = json.dumps(vectorizer_literals(literals).numpy().tolist())
        vectorized_comments = json.dumps(vectorizer_comments(comments).numpy().tolist())

        # Update the records
        cursor.execute(f"""
            UPDATE train_data
            SET vectorized_class_names = ?, vectorized_method_names = ?, vectorized_variable_names = ?,
             vectorized_parameter_names = ?, vectorized_api_calls = ?, vectorized_literals = ?, vectorized_comments = ?
            WHERE issue_number = ? AND file_path = ?
        """, (
            vectorized_class_names, vectorized_method_names, vectorized_variable_names, vectorized_parameter_names,
            vectorized_api_calls, vectorized_literals, vectorized_comments, row[0], row[1]))

        if i % 1000 == 0:
            # break
            conn.commit()

    # Get all records from test_data
    cursor.execute("SELECT * FROM test_data WHERE vectorized_class_names is null")
    rows = cursor.fetchall()
    i = 0
    for row in rows:
        #print("Test Table: " + str(i))
        i += 1
        # Split by ',', when the column is not empty
        class_names = row[5].split(',') if row[5] else []
        method_names = row[6].split(',') if row[6] else []
        variable_names = row[7].split(',') if row[7] else []
        parameter_names = row[8].split(',') if row[8] else []
        api_calls = row[9].split(',') if row[9] else []

        literals = json.loads(row[10]) if row[10] else []
        comments = json.loads(row[11]) if row[11] else []

        # preprocessing
        class_names = [preprocess_text(s) for s in class_names]
        method_names = [preprocess_text(s) for s in method_names]
        variable_names = [preprocess_text(s) for s in variable_names]
        parameter_names = [preprocess_text(s) for s in parameter_names]
        api_calls = [preprocess_text(s) for s in api_calls]
        literals = [preprocess_text(s) for s in literals]
        comments = [preprocess_text(s) for s in comments]

        class_names = check_string(" ".join(class_names))
        method_names = check_string(" ".join(method_names))
        parameter_names = check_string(" ".join(parameter_names))
        variable_names = check_string(" ".join(variable_names))
        api_calls = check_string(" ".join(api_calls))
        literals = check_string(" ".join(literals))
        comments = check_string(" ".join(comments))

        vectorized_class_names = json.dumps(vectorizer_class_names(class_names).numpy().tolist())
        vectorized_method_names = json.dumps(vectorizer_method_names(method_names).numpy().tolist())
        vectorized_variable_names = json.dumps(vectorizer_variable_names(variable_names).numpy().tolist())
        vectorized_parameter_names = json.dumps(vectorizer_parameter_names(parameter_names).numpy().tolist())
        vectorized_api_calls = json.dumps(vectorizer_api_calls(api_calls).numpy().tolist())
        vectorized_literals = json.dumps(vectorizer_literals(literals).numpy().tolist())
        vectorized_comments = json.dumps(vectorizer_comments(comments).numpy().tolist())

        # Update query
        cursor.execute(f"""
            UPDATE test_data
            SET vectorized_class_names = ?, vectorized_method_names = ?, vectorized_variable_names = ?,
             vectorized_parameter_names = ?, vectorized_api_calls = ?, vectorized_literals = ?, vectorized_comments = ?
            WHERE issue_number = ? AND file_path = ?
        """, (vectorized_class_names, vectorized_method_names, vectorized_variable_names, vectorized_parameter_names,
              vectorized_api_calls, vectorized_literals, vectorized_comments, row[0], row[1]))

        if i % 1000 == 0:
            # break
            conn.commit()

    conn.commit()
    conn.close()

    print("Issue localization DB vectorized successfully")


def check_string(input_string):
    return input_string if input_string else ">>"


if __name__ == "__main__":
    owner = 'vaadin'  # 'vaadin'
    repository = 'flow'  # 'flow'

    # Clone repository
    clone_repository(owner, repository)

    # Create DB for issue localization
    create_db_issue_localization(owner, repository)

    # Populate DB for issue localization
    populate_file_modification_table_for_issue_localization(owner, repository)
    populate_db_issue_localization(owner, repository)

    # Localization Custom Dataset (Only for jdt data)
    # To use the JDT dataset, comment out the line above and uncomment the lines below
    # Additionally, set an appropriate value (e.g. 180000) to MAX_TOKENS_VECTORIZATION

    # populate_db_issue_localization_from_txt(owner, repository)

    # Create vocabulary list
    create_vocabulary_list_from_db(owner, repository)

    # Vectorize code files
    vectorize_code_files(owner, repository)