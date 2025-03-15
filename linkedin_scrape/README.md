# LinkedIn Job Scraper

A modular Python application for scraping job listings from LinkedIn.

## Features

- Search for jobs by keyword and location
- Filter jobs by time period
- Extract detailed job information including:
  - Company name
  - Job title
  - Job description
  - Seniority level
  - Employment type
  - Job function
  - Industries
  - Posting date

## Directory Structure

```
linkedin_scraper/
├── config/             # Configuration settings
├── scrapers/           # Job scraping modules
├── utils/              # Utility functions
├── __init__.py         # Package initialization
├── main.py             # Main entry point
output/                 # Output directory for CSV files
requirements.txt        # Project dependencies
README.md               # This file
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with:

```python
python linkedin_scraper/main.py
```

Or import and use the scraper in your own code:

```python
from linkedin_scraper.scrapers.linkedin import LinkedInJobScraper

scraper = LinkedInJobScraper()
jobs_df = scraper.scrape(
    input_keyword='AI Engineer',
    location='Taiwan',
    time_filter='r86400',
    max_pages=5
)
jobs_df.to_csv('output/linkedin_jobs.csv', index=False)
```

## Configuration

Edit `linkedin_scraper/config/settings.py` to modify default parameters. 

## Service Initialization

To initialize the scraping service, ensure that your environment variables are set correctly in the `.env` file. This includes setting up your LinkedIn credentials and any other necessary API keys.

## Database Setup

1. Ensure your database is running and accessible.
2. Initialize the database by running the following command:
   ```bash
   python linkedin_scraper/init_db.py
   ```

## Pulling Data from the Database

To retrieve data from the database, you can use the following script:

```python
from linkedin_scraper.database import Database

db = Database()
jobs_data = db.get_all_jobs()
print(jobs_data)
```

This will fetch all job listings stored in the database and print them to the console.

## Docker Deployment

To deploy the application using Docker, use the provided `docker-compose.yml` file. Run the following command to start the services:

```bash
docker-compose up --build
```

Ensure that your Docker environment is set up correctly and that all necessary environment variables are configured in the `.env` file.

## Local Deployment with Scheduled Scraping

You can deploy this application locally and set it to run automatically at 12:00 PM (noon) every day.

### For Linux/Mac Users:

1. Make the startup script executable:
   ```bash
   chmod +x start_local_scheduler.sh
   chmod +x stop_scheduler.sh
   ```

2. Start the scheduler:
   ```bash
   ./start_local_scheduler.sh
   ```

3. To stop the scheduler:
   ```bash
   ./stop_scheduler.sh
   ```

### For Windows Users:

1. Open PowerShell as Administrator.

2. If you haven't already, you may need to set the execution policy to allow running scripts:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Run the startup script:
   ```powershell
   .\start_local_scheduler.ps1
   ```

4. To stop the scheduler:
   ```powershell
   .\stop_scheduler.ps1
   ```

### Checking Logs

The scheduler logs are saved to `scheduler.log`. You can view them by running:
```bash
tail -f scheduler.log  # Linux/Mac
```
Or on Windows:
```powershell
Get-Content scheduler.log -Wait
```

### Managing Search Configurations

Before the scheduler can run effectively, you need to add at least one search configuration:

```bash
python -m app.main add-config --name "Data Science Jobs" --keyword "Data Scientist" --location "Taiwan" --max-pages 5
```

You can list all configurations:
```bash
python -m app.main list-configs
```

And remove configurations you no longer need:
```bash
python -m app.main delete-config --id <config_id>
``` 