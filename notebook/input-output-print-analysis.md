# Input, Output, and Print Analysis

Source file: `notebook/ai-career-recommendation-eda-ml (2).py`

## Summary

| Function | Found | Count |
| --- | --- | ---: |
| `input()` | No | 0 |
| `output()` | No | 0 |
| `print()` | Yes | 9 |

## Findings

### `input()`

No `input()` function calls were found in the file.

### `output()`

No `output()` function calls were found in the file.

### `print()`

The file contains the following `print()` calls:

| Line | Code |
| ---: | --- |
| 42 | `print("Dataset Shape:", data.shape)` |
| 43 | `print("\nDataset Info:")` |
| 114 | `print("Most common skills:\n", skills_count)` |
| 139 | `print("Most common interests:\n", interests_count)` |
| 189 | `print("Shape of text features:", text_features.shape)` |
| 212 | `print("Feature matrix shape:", X.shape)` |
| 213 | `print("Target classes:", target_encoder.classes_)` |
| 222 | `print("Training set shape:", X_train.shape)` |
| 223 | `print("Testing set shape:", X_test.shape)` |

## Notes

The file does not request user input interactively. Its console output is limited to dataset, feature, target, and train/test shape summaries printed during execution.
