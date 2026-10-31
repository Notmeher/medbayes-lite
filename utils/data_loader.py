import json

def load_questions_from_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)
            if isinstance(content, dict) and "data" in content:
                return content["data"]
            return content
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
# utils/data_loader.py


def load_jsonl(file_path):
    """Load a JSONL or JSON file and return the list of data items."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)
            if isinstance(content, dict) and "data" in content:
                return content["data"]
            return content
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
