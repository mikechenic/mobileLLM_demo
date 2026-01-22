"""Handler for get-product tool"""


def handler(a: float, b: float) -> str:
    """Calculate the product of two numbers"""
    result = a * b
    return f"The product of {a} and {b} is {result}."
