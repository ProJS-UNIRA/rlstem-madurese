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
    # reduplication removal, it heavily rely on - character
    # nak-kanak -> kanak
    Rule(name="reduplication removal", pattern=r'^.+\-(.+)$', replacement=r'\1', replacements=[]),

    # remove suffix nah
    # bukunah -> buku, kalambhinah -> kalambhi
    Rule(name="remove nah suffix", pattern=r'^(.+)nah$', replacement=r'\1', replacements=[]),

    # remove suffix -han
    # kawajibhan -> wajib
    Rule(name="remove -han suffix", pattern=r'^(.+)han$', replacement=r'\1', replacements=[]),

    # remove suffix ya,na,ni,an,ih,eh,en,ah
    # soghiya -> soghi, ramana -> rama
    Rule(name="remove suffix ya,na,ni,an,ih,eh,en,ah", pattern=r'^(.+)(ya|na|ni|an|ih|eh|en|ah)$',
        replacement=r'\1', replacements=[]),

    # remove suffix -[aei]
    # kala'a -> kala, sema'e -> sema'
    Rule(name="remove suffix -[aei]", pattern=r'^(.+)[aei]$', replacement=r'\1', replacements=[]),

    # remove suffix -aghi
    # kala'aghi -> kala', sema'aghi -> sema'
    Rule(name="remove suffix -aghi", pattern=r'^(.+)aghi$', replacement=r'\1', replacements=[]),

    # remove prefix [aeè]-
    Rule(name="remove prefix -[aeè]", pattern=r'^([ae\u00E8])(.+)$', replacement=r'\2', replacements=[]),

    # remove prefix ta,ma,ka,sa,pa,pe
    Rule(name="remove prefix ta,ma,ka,sa,pa,pe", pattern=r'^(ta|ma|ka|sa|pa|pe)(.+)$',
        replacement=r'\2', replacements=[]),

    # remove prefix par followed by non-vowel
    Rule(name="remove prefix par followed by non-vowel", pattern=r'^(par)([^aeuio].+)$',
        replacement=r'\2', replacements=[]),

    # remove prefix ng
    Rule(name="remove prefix ng", pattern=r'^ng(.+)$', replacement=r'\1', replacements=[]),

    # modify prefix ng followed by vowel
    Rule(name="modify prefix ng followed by vowel", pattern=r'^ng([aeio].+)$', replacement="",
        replacements=[r'k\1', r'g\1', r'gh\1']),

    # modify prefix m
    Rule(name="modify prefix m", pattern=r'^m([aeou].+)$', replacement="",
        replacements=[r'b\1', r'p\1', r'bh\1']),

    # modify prefix ny
    Rule(name="modify prefix ny", pattern=r'^ny([aeo].+)$', replacement="",
        replacements=[r's\1', r'c\1', r'j\1', r'jh\1']),

    # modify prefix n
    Rule(name="modify prefix n", pattern=r'^n([aeo].+)$', replacement="",
        replacements=[r't\1', r'd\1', r'dh\1']),

    # modify prefix gha
    Rule(name="modify prefix gha", pattern=r'^(.+)(gha)$', replacement=r'\1k',
        replacements=[]),

    # modify prefix dhan
    Rule(name="modify prefix dhan", pattern=r'^(.+)dhan$', replacement=r'\1t',
        replacements=[]),

    # remove infix al,ar,en,in,om,um
    Rule(name="remove infix al,ar,en,in,om,um", pattern=r'^([^aiueo]{1,2})(al|ar|en|in|om|um)(.+)$',
        replacement=r'\1\3', replacements=[]),
]
