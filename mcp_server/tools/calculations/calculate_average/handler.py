"""Handler for calculate-average tool"""


def handler(numbers: list) -> str:
    """Calculate the average of multiple numbers"""
    if not numbers:
        return "No numbers provided"
    
    average = sum(numbers) / len(numbers)
    return f"The average of {numbers} is {average:.2f}."
