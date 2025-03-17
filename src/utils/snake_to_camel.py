from typing import List


def snake_to_camel(string: str) -> str:
    words: List[str] = string.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])
