from typing import NamedTuple

class Task(NamedTuple):
    category: str
    input_header: str
    examples: str
    task: str
    input_footer: str

class Range(NamedTuple):
    lower: int
    upper: int

class Input(NamedTuple):
    task_indices: Range
    examples_indices: Range
    tokenized: list[int]