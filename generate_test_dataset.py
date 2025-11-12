"""
Generate a test dataset for evaluating arithmetic models.
Creates 100 addition and 100 subtraction questions for each digit length (1-9 digits).
"""

import json
import random
from typing import List, Dict


def generate_test_dataset(
    output_file: str = "test_dataset.json",
    questions_per_digit: int = 100,
    max_digits: int = 9,
    seed: int = 42,
):
    """
    Generate a test dataset with addition and subtraction questions.
    
    Args:
        output_file: Path to save the test dataset
        questions_per_digit: Number of questions per digit length per operation
        max_digits: Maximum number of digits (1 to max_digits)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    data = []
    
    # Generate addition questions
    print(f"Generating addition questions...")
    for n_digits in range(1, max_digits + 1):
        for _ in range(questions_per_digit):
            # Generate two numbers with n_digits
            num1 = random.randint(10**(n_digits-1), 10**n_digits - 1)
            num2 = random.randint(10**(n_digits-1), 10**n_digits - 1)
            
            # Randomly swap order 50% of the time
            if random.random() < 0.5:
                num1, num2 = num2, num1
            
            answer = num1 + num2
            question = f"{num1} + {num2}"
            output = f"{num1} + {num2} = {answer}"
            
            data.append({
                "instruction": f"What is {question}?",
                "input": question,
                "output": output,
                "answer": str(answer),
                "operation": "addition",
                "num_digits": n_digits,
            })
    
    # Generate subtraction questions
    print(f"Generating subtraction questions...")
    for n_digits in range(1, max_digits + 1):
        for _ in range(questions_per_digit):
            # Generate two numbers with n_digits
            num1 = random.randint(10**(n_digits-1), 10**n_digits - 1)
            num2 = random.randint(10**(n_digits-1), 10**n_digits - 1)
            
            # Ensure num1 >= num2 for non-negative results
            if num1 < num2:
                num1, num2 = num2, num1
            
            # Randomly swap order 50% of the time (but keep result non-negative)
            if random.random() < 0.5 and num1 >= num2:
                # Only swap if it won't create negative result
                pass  # Keep num1 >= num2
            
            answer = num1 - num2
            question = f"{num1} - {num2}"
            output = f"{num1} - {num2} = {answer}"
            
            data.append({
                "instruction": f"What is {question}?",
                "input": question,
                "output": output,
                "answer": str(answer),
                "operation": "subtraction",
                "num_digits": n_digits,
            })
    
    # Shuffle the data
    random.shuffle(data)
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Test dataset generated successfully!")
    print(f"{'='*60}")
    print(f"Total questions: {len(data)}")
    print(f"  - Addition: {len([d for d in data if d['operation'] == 'addition'])}")
    print(f"  - Subtraction: {len([d for d in data if d['operation'] == 'subtraction'])}")
    print(f"\nBreakdown by digit length:")
    for n in range(1, max_digits + 1):
        add_count = len([d for d in data if d['operation'] == 'addition' and d['num_digits'] == n])
        sub_count = len([d for d in data if d['operation'] == 'subtraction' and d['num_digits'] == n])
        print(f"  {n}-digit: {add_count} addition, {sub_count} subtraction")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*60}")
    
    return data


if __name__ == "__main__":
    import fire
    fire.Fire(generate_test_dataset)

