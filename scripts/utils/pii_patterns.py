"""
Custom PII patterns for debt collection conversations.
Extends Microsoft Presidio with domain-specific recognizers.
"""

import re
from presidio_analyzer import PatternRecognizer, Pattern


def get_dob_recognizer() -> PatternRecognizer:
    """
    Recognizer for Date of Birth patterns commonly found in
    debt collection calls (verbal confirmation of identity).
    """
    patterns = [
        # Numeric formats: 01/15/1990, 1-15-90, 03.25.1985
        Pattern(
            name="dob_numeric",
            regex=r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
            score=0.6,
        ),
        # Written formats: January 15, 1990 / Jan 15 1990
        Pattern(
            name="dob_written_mdy",
            regex=r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
            score=0.7,
        ),
        # Written formats: 15 January 1990
        Pattern(
            name="dob_written_dmy",
            regex=r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
            score=0.7,
        ),
        # Context-triggered: "date of birth is ...", "dob: ...", "born on ..."
        Pattern(
            name="dob_contextual",
            regex=r"(?:date\s+of\s+birth|d\.?o\.?b\.?|born\s+on)[:\s]+[\w\s,/\-\.]+",
            score=0.85,
        ),
    ]

    return PatternRecognizer(
        supported_entity="DATE_OF_BIRTH",
        patterns=patterns,
        name="DOB Recognizer",
        supported_language="en",
    )


def get_account_number_recognizer() -> PatternRecognizer:
    """
    Recognizer for account/reference numbers in debt collection context.
    """
    patterns = [
        # "account ending in 4532" / "account number 12345678"
        Pattern(
            name="account_context",
            regex=r"(?:account|acct|ref(?:erence)?)\s*(?:number|no\.?|#|ending\s+in)[:\s]*\d{3,12}",
            score=0.85,
        ),
        # Standalone long number sequences (6-12 digits) that could be account numbers
        Pattern(
            name="account_standalone",
            regex=r"\b\d{6,12}\b",
            score=0.3,  # Low confidence — needs context
        ),
    ]

    return PatternRecognizer(
        supported_entity="ACCOUNT_NUMBER",
        patterns=patterns,
        name="Account Number Recognizer",
        supported_language="en",
    )


def get_balance_recognizer() -> PatternRecognizer:
    """
    Recognizer for monetary amounts in debt collection context.
    """
    patterns = [
        # $1,234.56 / $500 / $12,345
        Pattern(
            name="currency_usd",
            regex=r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            score=0.8,
        ),
        # "balance of 1234.56" / "owe 500 dollars" / "amount of $1,234"
        Pattern(
            name="balance_context",
            regex=r"(?:balance|owe|amount|payment|debt)\s+(?:of\s+)?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
            score=0.85,
        ),
    ]

    return PatternRecognizer(
        supported_entity="FINANCIAL_AMOUNT",
        patterns=patterns,
        name="Balance Recognizer",
        supported_language="en",
    )


def get_ssn_recognizer() -> PatternRecognizer:
    """
    Recognizer for Social Security Numbers.
    """
    patterns = [
        # Standard format: 123-45-6789
        Pattern(
            name="ssn_dashes",
            regex=r"\b\d{3}-\d{2}-\d{4}\b",
            score=0.85,
        ),
        # No dashes: 123456789 (with context)
        Pattern(
            name="ssn_context",
            regex=r"(?:social\s+security|ssn|ss\s*#)[:\s]*\d{9}",
            score=0.9,
        ),
        # Last 4 of SSN (common in verification)
        Pattern(
            name="ssn_last4",
            regex=r"(?:last\s+four|last\s+4)[:\s]*\d{4}",
            score=0.7,
        ),
    ]

    return PatternRecognizer(
        supported_entity="US_SSN",
        patterns=patterns,
        name="SSN Recognizer",
        supported_language="en",
    )


def get_all_custom_recognizers() -> list:
    """Return all custom recognizers for the debt collection domain."""
    return [
        get_dob_recognizer(),
        get_account_number_recognizer(),
        get_balance_recognizer(),
        get_ssn_recognizer(),
    ]


# Entity type → placeholder mapping
ENTITY_PLACEHOLDERS = {
    "PERSON": "[PERSON]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "DATE_OF_BIRTH": "[DOB]",
    "US_SSN": "[SSN]",
    "ACCOUNT_NUMBER": "[ACCOUNT_NO]",
    "FINANCIAL_AMOUNT": "[AMOUNT]",
    "LOCATION": "[LOCATION]",
    "CREDIT_CARD": "[CARD_NO]",
    "DATE_TIME": "[DATE]",
    "NRP": "[NRP]",
    "URL": "[URL]",
    "IP_ADDRESS": "[IP]",
}
