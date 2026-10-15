from datetime import date

def get_current_date() -> str:
    """
    Returns the current date in YYYY-MM-DD format.
    """
    return date.today().strftime("%Y-%m-%d")
