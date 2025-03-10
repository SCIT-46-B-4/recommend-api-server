import re
from typing import Dict

from fastapi import Request


def camel_to_snake(string: str) -> str:
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', string)
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)

    return snake.lower()

def convert_query_params(request: Request) -> Dict[str, str]:
    params = dict(request.query_params)

    return {camel_to_snake(key): val for key, val in params.items()}