import re
import javalang


def remove_leading_multiline_comment(code_text):
    pattern = r"^\s*/\*.*?\*/\s*"
    cleaned_content = re.sub(pattern, "", code_text, flags=re.DOTALL)

    return cleaned_content


def parse_java_file(java_code):
    java_code = remove_leading_multiline_comment(java_code)
    java_code = java_code.replace("::", ".")  # to avoid ecxeption

    # Initializing containers for the extracted data
    result = {
        "class_or_interface_names": [],
        "method_names": [],
        "parameter_names": [],
        "variable_names": [],
        "api_calls": [],
        "literals": [],
        "comments": []
    }

    # Tokenize the code using javalang
    tokens = list(javalang.tokenizer.tokenize(java_code))
    tree = javalang.parse.parse(java_code)

    # Extract class/interface names
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        result["class_or_interface_names"].append(node.name)

    for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
        result["class_or_interface_names"].append(node.name)

    # Extract method names and parameters
    for path, node in tree.filter(javalang.tree.MethodDeclaration):
        result["method_names"].append(node.name)
        for param in node.parameters:
            result["parameter_names"].append(param.name)

    # Extract variable names
    for path, node in tree.filter(javalang.tree.VariableDeclarator):
        result["variable_names"].append(node.name)

    # Extract API calls (MethodInvocation nodes)
    for path, node in tree.filter(javalang.tree.MethodInvocation):
        qualifier = f"{node.qualifier}." if node.qualifier else ""
        result["api_calls"].append(f"{qualifier}{node.member}")

    # Extract literals
    for token in tokens:
        if isinstance(token, javalang.tokenizer.String):
            result["literals"].append(token.value)

    # Extract comments using regex (regex handles comments not detected by javalang)
    single_line_pattern = r'//.*?$'
    multi_line_pattern = r'/\*.*?\*/'

    single_line_comments = re.findall(single_line_pattern, java_code, re.MULTILINE)
    multi_line_comments = re.findall(multi_line_pattern, java_code, re.DOTALL)

    # Clean comments
    cleaned_single_line = [comment.lstrip("//").strip() for comment in single_line_comments]
    cleaned_multi_line = [comment.lstrip("/*").rstrip("*/").strip() for comment in multi_line_comments]

    result["comments"].extend(cleaned_single_line + cleaned_multi_line)

    return result
