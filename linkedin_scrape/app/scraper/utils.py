from urllib.parse import urlparse
from datetime import datetime

def normalize_url(url):
    """標準化 URL，移除查詢參數以進行去重。"""
    if url:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return None

def generate_timestamp():
    """生成當前時間戳"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_search_params(keyword, location):
    """格式化搜尋參數以用於 URL"""
    # 替換空格為 %20 等 URL 編碼操作
    keyword = keyword.replace(' ', '%20')
    location = location.replace(' ', '%20')
    return keyword, location 