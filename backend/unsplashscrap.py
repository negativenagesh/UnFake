from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import tempfile
import time
import urllib.parse

# Create a temporary directory for Chrome user data to avoid conflicts
temp_dir = tempfile.mkdtemp()

# Set up Chrome options
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument(f"--user-data-dir={temp_dir}")  # Use a unique temp directory
options.add_argument("--no-sandbox")  # Bypass OS security model
options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
options.add_argument("--disable-gpu")  # Disable GPU
options.add_argument("--remote-debugging-port=9222")  # Avoid port conflicts

# Initialize the Chrome driver
try:
    driver = webdriver.Chrome(options=options)
except Exception as e:
    print(f"Error initializing driver: {e}")
    exit()

# Target URL to scrape
url = "https://unsplash.com/s/photos/india"

# Fetch the page
try:
    driver.get(url)
    # Scroll down to load more images (Unsplash uses lazy loading)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)  # Wait 5 seconds for images to load
except Exception as e:
    print(f"Error fetching page: {e}")
    driver.quit()
    exit()

# Get the rendered HTML
html_content = driver.page_source
driver.quit()

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find all <img> tags with src starting with "https://images.unsplash.com/"
# Filter to exclude profile thumbnails (those with small height or specific classes)
img_tags = soup.find_all('img', src=lambda x: x and x.startswith('https://images.unsplash.com/') and 'photo-' in x)

# Take the first 10 valid image tags
first_10_imgs = img_tags[:10]

# Extract and clean the image URLs to get raw versions
image_urls = []
for img in first_10_imgs:
    src = img['src']
    # Parse the URL and remove query parameters to get the raw image
    parsed_url = urllib.parse.urlparse(src)
    raw_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    image_urls.append(raw_url)

# Print the raw image URLs
for i, url in enumerate(image_urls, 1):
    print(f"Image {i}: {url}")

# Clean up the temporary directory
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)