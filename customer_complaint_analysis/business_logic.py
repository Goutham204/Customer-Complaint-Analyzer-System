def urgency_score(text, category):
    text = text.lower()

    if "fraud" in text or "charged twice" in text:
        return "High"

    if category == "technical":
        return "Medium"

    return "Low"


def root_cause(text):
    text = text.lower()

    if "app" in text or "login" in text:
        return "Application issue"
    if "refund" in text or "charged" in text:
        return "Billing issue"
    if "delay" in text:
        return "Process delay"

    return "General service issue"
