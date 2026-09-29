All Python docstrings in this workspace must follow the Google Python Style Guide layout.

- Use triple double quotes for docstrings.
- Begin with a concise summary sentence.
- Separate the summary from further details with a blank line when needed.
- Use Google-style sections such as `Args:`, `Returns:`, `Yields:`, `Raises:`, and `Attributes:` when applicable.
- Indent section contents consistently by four spaces under the section header.
- Document parameters and attributes as `name (type): description`.
- Document return values as `type: description`; omit `Returns:` when a function returns `None`.
- Keep docstrings accurate, concise, and synchronized with the implementation.

Example:

```python
def load_records(path: str, limit: int | None = None) -> list[dict[str, str]]:
	"""Load records from a file.

	Args:
		path (str): Path to the input file.
		limit (int | None): Maximum number of records to load.

	Returns:
		list[dict[str, str]]: The loaded records.

	Raises:
		FileNotFoundError: If the input file does not exist.
	"""
```
