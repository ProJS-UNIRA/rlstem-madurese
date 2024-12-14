"""Module for defining rules and recovery modes for stemming."""

from dataclasses import dataclass
from enum import Enum
import re

@dataclass
class Rule:
    """A rule for stemming."""
    name: str
    pattern: str
    replacement: str
    replacements: list[str]

    def __post_init__(self):
        # validate pattern
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {self.pattern}") from exc

    def apply(self, word: str) -> list[str]:
        """Apply the rule to the word."""
        if self.replacements:
            return [re.sub(self.pattern, replacement, word) for replacement in self.replacements]
        if self.replacement:
            return [re.sub(self.pattern, self.replacement, word)]
        return [word] # no change

rules = [
    # REDUPLICATION
    Rule(name="reduplication removal", pattern=r'^.+\-(.+)$', replacement=r'\1', replacements=[]),

    # SUFFIXES
    # -nah, -han suffixes
    Rule(name="remove nah suffix", pattern=r'^(.+)nah$', replacement=r'\1', replacements=[]),
    Rule(name="remove -han suffix", pattern=r'^(.+)han$', replacement=r'\1', replacements=[]),
    
    # Common suffixes group
    Rule(name="remove common suffixes", pattern=r'^(.+)(ya|na|ni|an|ih|eh|en|ah)$',
        replacement=r'\1', replacements=[]),
    
    # Vowel and aghi suffixes
    Rule(name="remove vowel suffix", pattern=r'^(.+)[aei\u00E8]$', replacement=r'\1', replacements=[]),
    Rule(name="remove -aghi suffix", pattern=r'^(.+)aghi$', replacement=r'\1', replacements=[]),
    
    # Special transformation suffixes
    Rule(name="transform gha suffix", pattern=r'^(.+)(gha|gh\u00E2)$', replacement=r'\1k', replacements=[]),
    Rule(name="transform dhan suffix", pattern=r'^(.+)dhan$', replacement=r'\1t', replacements=[]),

    # PREFIXES
    # Simple prefixes
    Rule(name="remove vowel prefix", pattern=r'^([ae\u00E8])(.+)$', replacement=r'\2', replacements=[]),
    Rule(name="remove common prefixes", pattern=r'^(ta|ma|ka|sa|pa|pe)(.+)$',
        replacement=r'\2', replacements=[]),
    Rule(name="remove par prefix", pattern=r'^(par)([^aeuio].+)$',
        replacement=r'\2', replacements=[]),
    
    # Nasal prefixes with transformations
    Rule(name="remove prefix ng", pattern=r'^ng(.+)$', replacement=r'\1', replacements=[]),
    Rule(name="transform ng prefix", pattern=r'^ng([aeio].+)$', replacement="",
        replacements=[r'k\1', r'g\1', r'gh\1']),
    Rule(name="transform m prefix", pattern=r'^m([aeou].+)$', replacement="",
        replacements=[r'b\1', r'p\1', r'bh\1']),
    Rule(name="transform ny prefix", pattern=r'^ny([aeo].+)$', replacement="",
        replacements=[r's\1', r'c\1', r'j\1', r'jh\1']),
    Rule(name="transform n prefix", pattern=r'^n([aeo].+)$', replacement="",
        replacements=[r't\1', r'd\1', r'dh\1']),

    # INFIXES
    Rule(name="remove infixes", pattern=r'^([^aiueo]{1,2})(al|ar|en|in|om|um)(.+)$',
        replacement=r'\1\3', replacements=[]),
]
