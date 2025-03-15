"""
LinkedIn job scraper implementation.
"""

import pandas as pd
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

from linkedin_scraper.utils import normalize_url
from linkedin_scraper.config import settings


class LinkedInJobScraper:
    """A class for scraping job listings from LinkedIn."""
    
    def __init__(self, headless=None):
        """
        Initialize the LinkedIn job scraper.
        
        Args:
            headless (bool, optional): Whether to run the browser in headless mode.
                                      Defaults to settings.HEADLESS.
        """
        self.headless = settings.HEADLESS if headless is None else headless
        
    def _setup_driver(self):
        """Set up and return a Selenium WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        return webdriver.Chrome(options=chrome_options)
    
    def _extract_job_links_and_dates(self, soup):
        """
        Extract job links and posting dates from the job search results page.
        
        Args:
            soup (BeautifulSoup): Parsed HTML of the job search results page
            
        Returns:
            tuple: (job_links, job_datetimes, new_jobs_found)
        """
        jobs = soup.find_all('div', class_='job-search-card')
        
        if not jobs:
            print("No jobs found on this page.")
            return [], [], False
            
        job_links = []
        job_datetimes = []
        new_jobs_found = False
        seen_links = getattr(self, 'seen_links', set())
        
        for job in jobs:
            link_tag = job.find('a', class_='base-card__full-link')
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else None
            normalized_link = normalize_url(link)
            
            if normalized_link and normalized_link not in seen_links:
                job_links.append(link)
                seen_links.add(normalized_link)
                new_jobs_found = True
            elif not link:
                print("No link found, skipping this job.")
                job_links.append(None)
            else:
                print(f"Duplicate job skipped: {normalized_link}")
                continue

            time_tag = job.find('time', class_=re.compile('job-search-card__listdate.*'))
            if time_tag and 'datetime' in time_tag.attrs:
                job_datetimes.append(time_tag['datetime'])
            else:
                time_text = job.find('time')
                job_datetimes.append(time_text.get_text(strip=True) if time_text else "Not found")
                
        self.seen_links = seen_links
        return job_links, job_datetimes, new_jobs_found
    
    def _scroll_page(self, driver):
        """
        Scroll down the page to load all job results.
        
        Args:
            driver (webdriver.Chrome): Selenium WebDriver instance
        """
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(settings.SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def _extract_job_details(self, driver, link):
        """
        Extract detailed information from a job posting page.
        
        Args:
            driver (webdriver.Chrome): Selenium WebDriver instance
            link (str): URL of the job posting
            
        Returns:
            dict: Job details
        """
        driver.get(link)
        time.sleep(settings.PAGE_LOAD_WAIT_TIME)
        job_soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract company name
        company_tag = job_soup.find('a', class_='topcard__org-name-link')
        company_name = company_tag.get_text(strip=True) if company_tag else "Not found"

        # Extract job title
        title_tag = job_soup.find('h1', class_='topcard__title')
        job_title = title_tag.get_text(strip=True) if title_tag else "Not found"

        # Extract description
        desc_tag = job_soup.find('div', class_='description__text')
        description = desc_tag.get_text(strip=True) if desc_tag else "Not found"

        # Extract job criteria
        criteria_list = job_soup.find_all('li', class_='description__job-criteria-item')
        seniority_level = employment_type = job_function = industries = "Not found"
        
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

        return {
            'Company Name': company_name,
            'Job Title': job_title,
            'Description': description,
            'Link': link,
            'Seniority Level': seniority_level,
            'Employment Type': employment_type,
            'Job Function': job_function,
            'Industries': industries
        }
    
    def scrape(self, input_keyword=None, location=None, time_filter=None, max_pages=None):
        """
        Scrape LinkedIn jobs with Selenium and handle specific job criteria columns.
        
        Args:
            input_keyword (str, optional): Job search keyword
            location (str, optional): Job location
            time_filter (str, optional): Time filter in seconds (e.g., 'r604800' for 7 days)
            max_pages (int, optional): Maximum pages to scrape
        
        Returns:
            pd.DataFrame: DataFrame with job details
        """
        # Use default values from settings if parameters are not provided
        input_keyword = input_keyword or settings.DEFAULT_KEYWORD
        location = location or settings.DEFAULT_LOCATION
        time_filter = time_filter or settings.DEFAULT_TIME_FILTER
        max_pages = max_pages or settings.DEFAULT_MAX_PAGES
        
        base_url = f'https://www.linkedin.com/jobs/search?keywords={input_keyword}&location={location}&geoId=106907071&distance=0&f_TPR={time_filter}&position=1'
        
        all_job_data = []
        self.seen_links = set()
        driver = self._setup_driver()
        
        try:
            for page in range(max_pages):
                url = f"{base_url}&pageNum={page}"
                print(f"Scraping page {page + 1}: {url}")

                driver.get(url)
                time.sleep(settings.PAGE_LOAD_WAIT_TIME)

                # Scroll to load all jobs
                self._scroll_page(driver)

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                job_links, job_datetimes, new_jobs_found = self._extract_job_links_and_dates(soup)
                
                if not new_jobs_found:
                    print("No new jobs on this page. Stopping.")
                    break

                # Scrape job details
                for i, link in enumerate(job_links):
                    if link is None:
                        all_job_data.append({
                            'Company Name': 'Not found',
                            'Job Title': 'Not found',
                            'Description': 'No link available',
                            'Link': 'Not found',
                            'Posting Date': job_datetimes[i],
                            'Seniority Level': 'Not found',
                            'Employment Type': 'Not found',
                            'Job Function': 'Not found',
                            'Industries': 'Not found'
                        })
                        continue
                    
                    try:
                        job_data = self._extract_job_details(driver, link)
                        job_data['Posting Date'] = job_datetimes[i]
                        all_job_data.append(job_data)

                        print(f"Scraped: {job_data['Job Title']} at {job_data['Company Name']} (Posted: {job_datetimes[i]})")
                        print(f"  Seniority: {job_data['Seniority Level']}, Type: {job_data['Employment Type']}")
                        print(f"  Function: {job_data['Job Function']}, Industries: {job_data['Industries']}")

                    except Exception as e:
                        print(f"Error scraping {link}: {str(e)}")
                        all_job_data.append({
                            'Company Name': 'Error',
                            'Job Title': 'Error',
                            'Description': f"Error: {str(e)}",
                            'Link': link,
                            'Posting Date': job_datetimes[i],
                            'Seniority Level': 'Error',
                            'Employment Type': 'Error',
                            'Job Function': 'Error',
                            'Industries': 'Error'
                        })

                time.sleep(settings.PAGE_TRANSITION_WAIT_TIME)  # Delay between pages

        finally:
            driver.quit()

        return pd.DataFrame(all_job_data)
    
    def save_to_csv(self, df, filename=None):
        """
        Save the scraped data to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame with job details
            filename (str, optional): Filename to save as. If None, uses a timestamp.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{settings.OUTPUT_DIRECTORY}/linkedin_jobs_{timestamp}.csv"
        elif not filename.startswith(f"{settings.OUTPUT_DIRECTORY}/"):
            filename = f"{settings.OUTPUT_DIRECTORY}/{filename}"
            
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Data saved to {filename}")
        return filename 