"""
Configuration settings for the LinkedIn job scraper.
"""

# Default search parameters
DEFAULT_KEYWORD = 'AI Engineer'
DEFAULT_LOCATION = 'Taiwan'
DEFAULT_TIME_FILTER = 'r604800'  # Past week
DEFAULT_MAX_PAGES = 5

# Selenium settings
HEADLESS = True
SCROLL_PAUSE_TIME = 2
PAGE_LOAD_WAIT_TIME = 3
PAGE_TRANSITION_WAIT_TIME = 5

# Output settings
OUTPUT_DIRECTORY = 'output'
INCLUDE_TIMESTAMP = True 