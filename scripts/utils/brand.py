"""
RunGopher brand tokens and shared configuration.
All brand values in one place — used by report generator and templates.
"""

BRAND = {
    "colors": {
        "coral": "#ff1c4d",
        "cobalt": "#3259fe",
        "sand": "#f1e8d6",
        "navy": "#1a3a8c",
        "white": "#ffffff",
        "black": "#000000",
        "muted": "#666666",
        "coral_tint": "rgba(255, 28, 77, 0.12)",
        "cobalt_tint": "rgba(50, 89, 254, 0.10)",
        "success": "#10B981",
    },
    "typography": {
        "heading": "'Poppins', Arial, Helvetica, sans-serif",
        "heading_weight": "600",
        "body": "'Recursive', Arial, Helvetica, sans-serif",
        "body_weight": "400",
    },
    "company": "RunGopher",
}

# CSV column mapping — matches RunGopher export format
CSV_COLUMNS = {
    "call_id": "Call ID",
    "assistant": "Assistant",
    "assistant_phone": "Assistant Phone",
    "customer_phone": "Customer Phone",
    "direction": "Direction",
    "user_intent": "User Intent",
    "platform_status": "Platform Status",
    "tool_status": "Tool Status",
    "duration": "Duration (Seconds)",
    "connect_time": "Connect Time",
    "transcript": "Transcript",
}

# PII columns that need redaction (phone numbers in metadata)
PII_METADATA_COLUMNS = ["Assistant Phone", "Customer Phone"]

# Outcome categories for summary classification
OUTCOME_CATEGORIES = [
    "PAYMENT_ARRANGED",
    "CALLBACK_SCHEDULED",
    "REFUSED",
    "HUNG_UP",
    "VOICEMAIL",
    "WRONG_NUMBER",
    "DISPUTE",
    "PARTIAL_PAYMENT",
    "HARDSHIP_CLAIM",
    "COMPLIANCE_ISSUE",
    "NO_ANSWER",
    "TRANSFERRED",
    "OTHER",
]
