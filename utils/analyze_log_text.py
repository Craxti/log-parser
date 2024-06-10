def analyze_log_text(log_text, keyword_dict):
    for status, keywords in keyword_dict.items():
        for keyword in keywords:
            if re.search(keyword, log_text, re.IGNORECASE):
                return status
    return "Other"