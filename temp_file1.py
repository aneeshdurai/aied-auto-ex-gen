```python\ndef cumulative_sum(s):\n    total = 0\n    for value in s:\n        total += value\n        yield total\n```'
```python\n# Test case 1: Basic functionality with a list of positive integers\nassert list(cumulative_sum(iter([1, 2, 3, 4]))) == [1, 3, 6, 10]\n\n# Test case 2: Single element list\nassert list(cumulative_sum(iter([5]))) == [5]\n\n# Test case 3: List with zero\nassert list(cumulative_sum(iter([0, 1, 2, 3]))) == [0, 1, 3, 6]\n\n# Test case 4: List with negative numbers\nassert list(cumulative_sum(iter([-1, -2, -3, -4]))) == [-1, -3, -6, -10]\n\n# Test case 5: List with both positive and negative numbers\nassert list(cumulative_sum(iter([1, -1, 2, -2, 3, -3]))) == [1, 0, 2, 0, 3, 0]\n\n# Test case 6: List with floating point numbers\nassert list(cumulative_sum(iter([0.5, 1.5, 2.5]))) == [0.5, 2.0, 4.5]\n\n# Test case 7: Large numbers\nassert list(cumulative_sum(iter([1000000, 2000000, 3000000]))) == [1000000, 3000000, 6000000]\n\n# Test case 8: Using next() to get the first cumulative sum\nassert next(cumulative_sum(iter([10, 20, 30]))) == 10\n\n# Test case 9: Empty iterator (should not yield anything, but since input is non-empty, this is just for robustness)\ntry:\n    next(cumulative_sum(iter([])))\n    assert False, "Expected StopIteration"\nexcept StopIteration:\n    pass\n\n# Test case 10: Iterator with repeated values\nassert list(cumulative_sum(iter([1, 1, 1, 1, 1]))) == [1, 2, 3, 4, 5]\n```'