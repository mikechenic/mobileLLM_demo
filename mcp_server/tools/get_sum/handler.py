"""Handler for get-sum tool"""


def handler(a: float, b: float) -> str:
    """Calculate the sum of two numbers"""
    result = a + b
    return f"The sum of {a} and {b} is {result}."
