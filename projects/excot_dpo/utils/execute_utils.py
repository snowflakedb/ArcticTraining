import re


def _extract_sql(message):
    """
    Given a message, this function extracts the last SQL code by regex matching.
    The code should be wrapped by the MarkDown code block
    """
    pattern = r"```sql(.*?)```$"
    pattern2 = r"sql(.*?)```"

    matches = re.findall(pattern, message, flags=re.DOTALL | re.MULTILINE)
    matches2 = re.findall(pattern2, message, flags=re.DOTALL | re.MULTILINE)
    if matches:
        return matches[-1].strip()
    else:
        if matches2:
            return matches2[-1].strip()
        print("\nno matched code found...\n")
        return None
