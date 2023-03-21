import language_tool_python
import os


def is_target_error(rule, target_error_types=None):
    target_error_types = ["GRAMMAR", "PUNCTUATION"] if target_error_types is None else target_error_types
    if "ALL" in target_error_types:
        return True
    else:
        return rule.category in target_error_types


def count_errors(texts, language="de-DE", target_error_types=None, return_errors=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tool = language_tool_python.LanguageTool(language)
    matches = [tool.check(text) for text in texts]
    errors = [
        [rule for rule in match if is_target_error(rule, target_error_types)]
        for match in matches
    ]
    num_errors = sum(map(lambda m: len(m), errors))

    if return_errors:
        return num_errors, errors
    else:
        return num_errors
