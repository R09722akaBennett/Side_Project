#!/usr/bin/env python
"""
Main entry point for the LinkedIn job scraper.
"""

import argparse
from datetime import datetime

from linkedin_scraper.scrapers import LinkedInJobScraper
from linkedin_scraper.config import settings


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scrape LinkedIn job listings.')
    
    parser.add_argument(
        '--keyword', '-k',
        type=str,
        default=settings.DEFAULT_KEYWORD,
        help=f'Job search keyword (default: {settings.DEFAULT_KEYWORD})'
    )
    
    parser.add_argument(
        '--location', '-l',
        type=str,
        default=settings.DEFAULT_LOCATION,
        help=f'Job location (default: {settings.DEFAULT_LOCATION})'
    )
    
    parser.add_argument(
        '--time-filter', '-t',
        type=str,
        default=settings.DEFAULT_TIME_FILTER,
        help=f'Time filter (e.g., r86400 for 1 day, r604800 for 7 days) (default: {settings.DEFAULT_TIME_FILTER})'
    )
    
    parser.add_argument(
        '--max-pages', '-p',
        type=int,
        default=settings.DEFAULT_MAX_PAGES,
        help=f'Maximum pages to scrape (default: {settings.DEFAULT_MAX_PAGES})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output filename (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Disable headless mode for browser (show browser UI)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    print("LinkedIn Job Scraper")
    print("===================")
    print(f"Search Keyword: {args.keyword}")
    print(f"Location: {args.location}")
    print(f"Time Filter: {args.time_filter}")
    print(f"Max Pages: {args.max_pages}")
    print("===================")
    
    # Initialize scraper
    scraper = LinkedInJobScraper(headless=not args.no_headless)
    
    try:
        # Scrape jobs
        print("Starting job scraping...")
        start_time = datetime.now()
        
        jobs_df = scraper.scrape(
            input_keyword=args.keyword,
            location=args.location,
            time_filter=args.time_filter,
            max_pages=args.max_pages
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print results
        job_count = len(jobs_df)
        print(f"\nScraped {job_count} jobs in {duration:.2f} seconds")
        
        # Save results
        if job_count > 0:
            output_file = scraper.save_to_csv(jobs_df, args.output)
            print(f"Results saved to: {output_file}")
        else:
            print("No jobs found to save.")
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"\nError during scraping: {str(e)}")
        
    print("\nScraping completed.")


if __name__ == "__main__":
    main() 