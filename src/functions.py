import unicodedata

def normalize_word(word: str) -> str:
    """
    Normalize Madurese text while preserving special characters
    """
    return unicodedata.normalize('NFC', word).lower()

def get_unicode_code(char: str) -> str | list[str]:
    """
    Get the Unicode code of a character
    """
    # normalize the character form (NFC)
    char = normalize_word(char)

    if len(char) == 1:
        return f"\\u{ord(char):04x}"
    return [f"\\u{ord(c):04x}" for c in char]

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]