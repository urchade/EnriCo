### Annotation for Relation Extraction

#### Format
Each annotation should be a dictionary with:
- **`tokenized_text`**: List of tokens.
- **`spans`**: List of `[start, end]` indices for entities.
- **`relations`**: List of `[start_span_index, end_span_index, relation_type]` entries.

#### Example
```python
{
  'tokenized_text': ['Dr.', 'Jane', 'Smith', 'is', 'a', 'renowned', 'neurosurgeon', 'who', 'works', 'at', 'the', 'Mayo', 'Clinic', 'in', 'Rochester', ',', 'Minnesota', '.', 'She', 'studied', 'at', 'Harvard', 'University', 'and', 'published', 'numerous', 'papers', 'on', 'brain', 'surgery', '.'],
  'spans': [[1, 2], [6, 6], [10, 11], [13, 13], [15, 16], [21, 22], [24, 27], [29, 30]],
  'relations': [
    [0, 1, 'has occupation'],
    [1, 2, 'works at'],
    [2, 3, 'is located in'],
    [3, 4, 'is located in'],
    [0, 5, 'studied at'],
    [0, 6, 'published'],
    [6, 7, 'on topic']
  ]
}
```

#### Steps
1. **Tokenize**: Split text into individual tokens.
2. **Annotate Spans**: Identify entity spans with `[start, end]` indices.
3. **Annotate Relations**: Define relationships with `[start_span_index, end_span_index, relation_type]`.

This ensures a clear and consistent annotation format for relation extraction tasks.