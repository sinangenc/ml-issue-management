import json
import os
import re
import time

import requests
from dotenv import load_dotenv


def get_api_token():
    # Load .env file
    load_dotenv()

    # Get Github API Token from .env file
    return os.getenv('API_TOKEN')


def load_json_data(json_file_path):
    if not os.path.exists(json_file_path):
        raise Exception(f"JSON File doesn't exist!")

    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
            return json_data
    except json.JSONDecodeError:
        print(f"{json_file_path} can't be decoded.")


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


def download_all_issues(owner, repository, api_token):
    repo_path = create_repo_path(owner, repository)

    # Ensure the directory does not exist
    if os.path.exists(repo_path):
        print(f"ERROR: '{repo_path}' already exists.")
        return
    else:
        os.makedirs(repo_path)
        print(f"Path '{repo_path}' created.")

    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    issue_list = []
    page = 1

    print(f"Downloading: issues...")
    while True:
        params = {
            "page": page,
            "per_page": 100,
            "state": "all",
        }

        response_api = requests.get(f"https://api.github.com/repos/{owner}/{repository}/issues",
                                    headers=headers,
                                    params=params)

        if response_api.status_code == 200:
            issue_list.extend(response_api.json())

            # Check if there is the next page
            if response_api.headers.get("link") and 'rel="next"' in response_api.headers.get("link"):
                # print(f"Data retrieved. {page} Remaining Limit: {response_api.headers.get('X-RateLimit-Remaining')}")
                page += 1
            else:
                break
        elif ((response_api.status_code == 403 or response_api.status_code == 429) and
              response_api.headers.get("X-RateLimit-Remaining") == "0"):
            # Handle rate limit exceeded
            reset_time = int(response_api.headers.get("X-RateLimit-Reset"))
            wait_time = reset_time - time.time() + 60  # Adding 1 min to ensure the reset time has passed
            print(f"Rate limit exceeded, waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            raise Exception(
                f"An error occurred while retrieving data! "
                f"(Status Code: {response_api.status_code}, Response: {response_api.text})")

    # Save Data to JSON file
    filename = os.path.join(repo_path, 'issues.json')

    with open(filename, 'w') as file:
        json.dump(issue_list, file, indent=4)

    print(f"issues downloaded to {filename}")


def fetch_related_issues_for_pr(pull_request_number, owner, repository, token):
    #print("*")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    variables = {
        "owner": owner,
        "name": repository,
        "pullRequestNumber": pull_request_number
    }

    # GraphQL query
    query = """
    query($owner: String!, $name: String!, $pullRequestNumber: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pullRequestNumber) {
          closingIssuesReferences(first: 10) {
            nodes {
              number
              repository {
                name
                owner {
                  login
                }
              }
            }
          }
        }
      }
    }
    """

    # API call
    response = requests.post(
        'https://api.github.com/graphql',
        json={'query': query, 'variables': variables},
        headers=headers
    )

    # Check the response
    if response.status_code == 200:
        # Handle: rate limit exceeded
        if int(response.headers.get("X-RateLimit-Remaining")) < 10:
            wait_time = 60 * 60  # 1 hour
            print(f"Rate limit exceeded, waiting for {wait_time} seconds...")
            time.sleep(wait_time)

        result = response.json()

        # Extract the issues which are in the same repo with PR
        issues = result['data']['repository']['pullRequest']['closingIssuesReferences']['nodes']
        issue_numbers = [
            issue['number']
            for issue in issues
            if (issue['repository']['name'] == repository and issue['repository']['owner']['login'] == owner)
        ]

        return issue_numbers
    else:
        raise Exception(
            f"An error occurred while retrieving data! "
            f"(Status Code: {response.status_code}, Response: {response.text})")


def extract_pr_related_issues(owner, repository, api_token):
    repo_path = create_repo_path(owner, repository)

    issues_json_file_path = os.path.join(repo_path, 'issues.json')

    print(f"Extracting: pull requests - realted issues...")

    # Get all issues + pull requests
    json_data = load_json_data(issues_json_file_path)

    merged_prs = [report["number"] for report in json_data if
                  "pull_request" in report and report["pull_request"]["merged_at"] is not None]

    pr_issues_map = {}

    i = 0
    for pr_number in merged_prs:
        issue_numbers = fetch_related_issues_for_pr(pr_number, owner, repository, api_token)
        pr_issues_map[pr_number] = issue_numbers
        # print(f"{pr_number} -> {issue_numbers}")

    filename = os.path.join(repo_path, 'pull_request_related_issues.json')
    with open(filename, 'w') as file:
        json.dump(pr_issues_map, file, indent=4)

    print(f"pull requests - realted issues extracted to {filename}")


def convert_pr_issues_to_issue_prs(owner, repository):
    repo_path = create_repo_path(owner, repository)

    json_file_path = os.path.join(repo_path, 'pull_request_related_issues.json')

    # Get all issues + pull requests
    json_data = load_json_data(json_file_path)

    reversed_data = {}

    for pr_number, issue_numbers in json_data.items():
        for issue_number in issue_numbers:
            if issue_number not in reversed_data:
                reversed_data[issue_number] = list()

            if pr_number not in reversed_data[issue_number]:
                reversed_data[issue_number].append(pr_number)

    filename = os.path.join(repo_path, 'issue_related_pull_requests.json')
    with open(filename, 'w') as file:
        json.dump(reversed_data, file, indent=4)

    print(f"Extracted: issue - related pull requests ...")


def get_developers_from_pull_request(pull_request_number, owner, repository, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    variables = {
        "owner": owner,
        "name": repository,
        "pullRequestNumber": int(pull_request_number)
    }

    # GraphQL query
    query = """
    query($owner: String!, $name: String!, $pullRequestNumber: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pullRequestNumber) {
          commits(first: 100) {
            nodes {
              commit {
                author {
                  name
                  email
                  user {
                    login
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    # API call
    response = requests.post(
        'https://api.github.com/graphql',
        json={'query': query, 'variables': variables},
        headers=headers
    )

    # Check the response
    if response.status_code == 200:

        # Handle: rate limit exceeded
        if int(response.headers.get("X-RateLimit-Remaining")) < 10:
            wait_time = 60 * 60  # 1 hour
            print(f"Rate limit exceeded, waiting for {wait_time} seconds...")
            time.sleep(wait_time)

        result = response.json()

        developers = set()

        commits = result['data']['repository']['pullRequest']['commits']['nodes']
        for commit in commits:
            developer = commit['commit']['author']['user']
            if developer:
                developers.add(developer['login'])
            else:
                developer = commit['commit']['author']['email']
                if developer:
                    developers.add(developer)

        return list(developers)
    else:
        raise Exception(
            f"An error occurred while retrieving data! "
            f"(Status Code: {response.status_code}, Response: {response.text})")


def download_contributors_of_pull_requests(owner, repository, api_token):
    repo_path = create_repo_path(owner, repository)

    json_file_path = os.path.join(repo_path, 'issue_related_pull_requests.json')

    json_data = load_json_data(json_file_path)

    print(f"Downloading: contributors of pull requests...")

    pr_set = set()
    for issue, pull_requests in json_data.items():
        pr_set.update(pull_requests)

    pr_developers_map = {}

    i = 0
    for pr_number in pr_set:
        developers = get_developers_from_pull_request(pr_number, owner, repository, api_token)
        pr_developers_map[pr_number] = developers
        # print(f"{pr_number} -> {developers}")

    filename = os.path.join(repo_path, 'pull_request_developers.json')
    # filename = os.path.join(repo_path, 'pull_request_developers_combined.json')
    with open(filename, 'w') as file:
        json.dump(pr_developers_map, file, indent=4)

    print(f"contributors of pull requests downloaded to {filename}")


def get_commit_shas(pull_request_number, owner, repository, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    variables = {
        "owner": owner,
        "name": repository,
        "pullRequestNumber": int(pull_request_number)
    }

    # GraphQL query
    query = """
    query($owner: String!, $name: String!, $pullRequestNumber: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pullRequestNumber) {
          mergeCommit {
            oid
          }
          baseRefOid 
        }
      }
    }
    """

    # API call
    response = requests.post(
        'https://api.github.com/graphql',
        json={'query': query, 'variables': variables},
        headers=headers
    )

    # Check the response
    if response.status_code == 200:

        # Handle: rate limit exceeded
        if int(response.headers.get("X-RateLimit-Remaining")) < 10:
            wait_time = 60 * 60  # 1 hour
            print(f"Rate limit exceeded, waiting for {wait_time} seconds...")
            time.sleep(wait_time)

        result = response.json()
        # print(result)
        pr = result['data']['repository']['pullRequest']

        merge_commit_sha = pr['mergeCommit']['oid'] if pr['mergeCommit'] else None
        base_commit_sha = pr['baseRefOid'] if pr['baseRefOid'] else None
        return {
            'merge_commit_sha': merge_commit_sha,
            'base_commit_sha': base_commit_sha
        }
    else:
        raise Exception(
            f"An error occurred while retrieving data! "
            f"(Status Code: {response.status_code}, Response: {response.text})")


def download_commit_shas_of_pull_requests(owner, repository, api_token):
    repo_path = create_repo_path(owner, repository)

    json_file_path = os.path.join(repo_path, 'issue_related_pull_requests.json')

    json_data = load_json_data(json_file_path)

    print(f"Downloading: commit shas of pull requests...")

    pr_shas_map = dict()
    for issue, pull_requests in json_data.items():
        for pr_number in pull_requests:
            pr_shas_map[pr_number] = get_commit_shas(pr_number, owner, repository, api_token)
            # print(f"{pr_number}")

    filename = os.path.join(repo_path, 'pull_request_shas.json')
    with open(filename, 'w') as file:
        json.dump(pr_shas_map, file, indent=4)

    print(f"Shas of pull requests downloaded to {filename}")


if __name__ == "__main__":
    # Set GitHub owner and repository names to download issues
    owner = 'vaadin'
    repository = 'flow'

    # Load API Token from env file
    API_TOKEN = get_api_token()

    # Download all issues
    download_all_issues(owner, repository, API_TOKEN)

    # Extract closed issues list for PRs
    extract_pr_related_issues(owner, repository, API_TOKEN)

    # Convert format from PR->issues to issue->PRs
    convert_pr_issues_to_issue_prs(owner, repository)

    # Download PR contributors
    download_contributors_of_pull_requests(owner, repository, API_TOKEN)

    # Download PR merge commit shas
    download_commit_shas_of_pull_requests(owner, repository, API_TOKEN)
