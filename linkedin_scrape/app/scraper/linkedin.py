import time
import re
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import os

from app.scraper.utils import normalize_url, generate_timestamp
from app.database.operations import save_jobs_to_db
from app.config import DEFAULT_KEYWORD, DEFAULT_LOCATION, DEFAULT_TIME_FILTER, DEFAULT_MAX_PAGES

class LinkedInScraper:
    """LinkedIn 職缺爬蟲類別"""
    
    def __init__(self, headless=True):
        """初始化爬蟲設定"""
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # 設定 Chromium 二進制路徑（環境變數）
        chrome_binary = os.environ.get('CHROME_BIN', '/usr/bin/chromium')
        self.chrome_options.binary_location = chrome_binary

    def start_driver(self):
        """啟動 Selenium WebDriver"""
        chromedriver_path = os.environ.get('CHROMEDRIVER_PATH', '/usr/bin/chromedriver')
        if os.path.exists(chromedriver_path):
            service = Service(executable_path=chromedriver_path)
            return webdriver.Chrome(service=service, options=self.chrome_options)
        else:
            return webdriver.Chrome(options=self.chrome_options)
    
    def scrape_jobs(self, keyword=DEFAULT_KEYWORD, location=DEFAULT_LOCATION, 
                   time_filter=DEFAULT_TIME_FILTER, max_pages=DEFAULT_MAX_PAGES, 
                   save_to_db=True):
        """
        爬取 LinkedIn 職缺
        
        Args:
            keyword: 搜尋關鍵字
            location: 地點
            time_filter: 時間過濾器 (例如 r86400 表示 24 小時)
            max_pages: 最大爬取頁數
            save_to_db: 是否儲存到資料庫
            
        Returns:
            DataFrame: 含職缺詳細資訊的 DataFrame
        """
        base_url = f'https://www.linkedin.com/jobs/search?keywords={keyword}&location={location}&f_TPR={time_filter}&position=1'
        
        all_job_data = []
        seen_links = set()
        driver = self.start_driver()

        try:
            for page in range(max_pages):
                url = f"{base_url}&pageNum={page}"
                print(f"爬取第 {page + 1} 頁: {url}")

                driver.get(url)
                time.sleep(3)  # 等待初始載入

                # 捲動載入全部職缺
                self._scroll_page(driver)

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                jobs = soup.find_all('div', class_='job-search-card')
                
                if not jobs:
                    print("此頁未找到職缺。停止爬取。")
                    break

                # 收集職缺連結和發布日期
                job_links, job_datetimes, new_jobs_found = self._collect_job_links(jobs, seen_links)

                if not new_jobs_found:
                    print("此頁無新職缺。停止爬取。")
                    break

                # 爬取職缺詳細資訊
                page_jobs = self._scrape_job_details(driver, job_links, job_datetimes)
                all_job_data.extend(page_jobs)

                time.sleep(5)  # 頁面間延遲

        finally:
            driver.quit()

        # 轉換為 DataFrame
        df = pd.DataFrame(all_job_data)
        
        # 儲存到資料庫 (如果啟用)
        if save_to_db and not df.empty:
            save_jobs_to_db(df)
            print(f"成功儲存 {len(df)} 個職缺到資料庫")
            
        # 可同時儲存為 CSV 作為備份
        timestamp = generate_timestamp()
        csv_filename = f'linkedin_jobs_{timestamp}.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        return df
    
    def _scroll_page(self, driver):
        """捲動頁面以載入所有職缺"""
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def _collect_job_links(self, jobs, seen_links):
        """收集職缺連結和發布日期"""
        job_links = []
        job_datetimes = []
        new_jobs_found = False

        for job in jobs:
            link_tag = job.find('a', class_='base-card__full-link')
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else None
            normalized_link = normalize_url(link)
            
            if normalized_link and normalized_link not in seen_links:
                job_links.append(link)
                seen_links.add(normalized_link)
                new_jobs_found = True
            elif not link:
                print("未找到連結，跳過此職缺。")
                job_links.append(None)
            else:
                print(f"重複職缺已跳過: {normalized_link}")
                continue

            time_tag = job.find('time', class_=re.compile('job-search-card__listdate.*'))
            if time_tag and 'datetime' in time_tag.attrs:
                job_datetimes.append(time_tag['datetime'])
            else:
                time_text = job.find('time')
                job_datetimes.append(time_text.get_text(strip=True) if time_text else "未找到")

        return job_links, job_datetimes, new_jobs_found
    
    def _scrape_job_details(self, driver, job_links, job_datetimes):
        """爬取職缺詳細資訊"""
        job_details = []
        
        for i, link in enumerate(job_links):
            if link is None:
                job_details.append(self._create_empty_job_record(job_datetimes[i]))
                continue
                
            try:
                driver.get(link)
                time.sleep(3)  # 等待職缺頁面載入
                job_soup = BeautifulSoup(driver.page_source, 'html.parser')

                # 提取公司名稱
                company_tag = job_soup.find('a', class_='topcard__org-name-link')
                company_name = company_tag.get_text(strip=True) if company_tag else "未找到"

                # 提取職缺名稱
                title_tag = job_soup.find('h1', class_='topcard__title')
                job_title = title_tag.get_text(strip=True) if title_tag else "未找到"

                # 提取描述
                desc_tag = job_soup.find('div', class_='description__text')
                description = desc_tag.get_text(strip=True) if desc_tag else "未找到"

                # 提取職缺條件
                seniority_level, employment_type, job_function, industries = self._extract_job_criteria(job_soup)

                job_details.append({
                    'Company Name': company_name,
                    'Job Title': job_title,
                    'Description': description,
                    'Link': link,
                    'Posting Date': job_datetimes[i],
                    'Seniority Level': seniority_level,
                    'Employment Type': employment_type,
                    'Job Function': job_function,
                    'Industries': industries,
                    'Scrape Date': datetime.now().isoformat()
                })

                print(f"已爬取: {job_title} at {company_name} (發布: {job_datetimes[i]})")

            except Exception as e:
                print(f"爬取 {link} 時發生錯誤: {str(e)}")
                job_details.append(self._create_error_job_record(link, job_datetimes[i], str(e)))
                
        return job_details
    
    def _extract_job_criteria(self, job_soup):
        """提取職缺條件"""
        seniority_level = employment_type = job_function = industries = "未找到"
        
        criteria_list = job_soup.find_all('li', class_='description__job-criteria-item')
        if criteria_list:
            for item in criteria_list:
                subheader = item.find('h3', class_='description__job-criteria-subheader')
                text = item.find('span', class_='description__job-criteria-text')
                if subheader and text:
                    subheader_text = subheader.get_text(strip=True).lower()
                    value = text.get_text(strip=True)
                    
                    if 'seniority level' in subheader_text:
                        seniority_level = value
                    elif 'employment type' in subheader_text:
                        employment_type = value
                    elif 'job function' in subheader_text:
                        job_function = value
                    elif 'industries' in subheader_text:
                        industries = value
        
        return seniority_level, employment_type, job_function, industries
    
    def _create_empty_job_record(self, datetime_str):
        """建立空的職缺記錄"""
        return {
            'Company Name': '未找到',
            'Job Title': '未找到',
            'Description': '無連結可用',
            'Link': '未找到',
            'Posting Date': datetime_str,
            'Seniority Level': '未找到',
            'Employment Type': '未找到',
            'Job Function': '未找到',
            'Industries': '未找到',
            'Scrape Date': datetime.now().isoformat()
        }
    
    def _create_error_job_record(self, link, datetime_str, error):
        """建立錯誤的職缺記錄"""
        return {
            'Company Name': '錯誤',
            'Job Title': '錯誤',
            'Description': f"錯誤: {error}",
            'Link': link,
            'Posting Date': datetime_str,
            'Seniority Level': '錯誤',
            'Employment Type': '錯誤',
            'Job Function': '錯誤',
            'Industries': '錯誤',
            'Scrape Date': datetime.now().isoformat()
        } 