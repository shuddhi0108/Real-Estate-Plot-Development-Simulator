# E-Commerce Price Scraper

A Python web scraper that extracts product names, prices, ratings, and availability from an e-commerce website using both **BeautifulSoup** (static pages) and **Selenium** (dynamic/JS-rendered pages).

## Features
- Scrapes product listings across multiple pages
- Handles both static HTML and JavaScript-rendered content
- Exports structured data to CSV
- Polite scraping with request delays
- Follows ethical scraping practices (targets books.toscrape.com — a legal practice site)

## Tech Stack
- Python 3.x
- BeautifulSoup4
- Selenium (Chrome WebDriver)
- Requests
- CSV (stdlib)

## Setup

```bash
pip install -r requirements.txt
```

For Selenium, also install [ChromeDriver](https://chromedriver.chromium.org/) matching your Chrome version.

## Usage

```bash
python scraper.py
```

Output: `products.csv` with columns — `name`, `price`, `rating`, `availability`

## Sample Output

| name | price | rating | availability |
|---|---|---|---|
| A Light in the Attic | £51.77 | 3 | In stock |
| Tipping the Velvet | £53.74 | 1 | In stock |

## Project Structure
```
ecommerce-scraper/
├── scraper.py        # Main scraper (BeautifulSoup + Selenium)
├── requirements.txt  # Dependencies
└── README.md
```
