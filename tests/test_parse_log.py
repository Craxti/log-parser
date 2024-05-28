from utils.parser_log import parse_log

def test_parse_log():
    log_file = "log.txt"
    logs = parse_log(log_file)
    assert len(logs) > 0, "No logs were parsed from the file"

