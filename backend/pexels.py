import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from bs4 import BeautifulSoup
import json
import requests
from webdriver_manager.chrome import ChromeDriverManager

def scrape_pexels_images(search_term, page=1):
    url = f"https://www.pexels.com/search/{search_term}/?page={page}"
    options = Options()
    options.add_argument("--headless")  # Run without UI
    options.add_argument("--no-sandbox")  # Useful for Linux environments
    options.add_argument("--disable-dev-shm-usage")  # Avoid shared memory issues
    options.add_argument("--disable-gpu")  # Disable GPU acceleration
    options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    # Set the correct binary location for your local system
    options.binary_location = "/usr/bin/google-chrome"  # Adjust to "/usr/lib/chromium-browser/chromium" if using Chromium

    # Use webdriver-manager to automatically manage Chromedriver
    service = Service(ChromeDriverManager().install(), log_path='chromedriver.log')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    # Wait for images to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "photo-item__image")))

    # Scroll to load all images
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Try to extract from __NEXT_DATA__
    next_data = soup.find('script', id='__NEXT_DATA__')
    if next_data:
        data = json.loads(next_data.string)
        photos = data['props']['initialReduxState']['search']['photos']['entries']
        images = []
        for photo in photos:
            image = {
                "id": str(photo['id']),
                "display_url": photo['src']['medium'],
                "download_url": photo['src']['original'],
                "alt_text": photo.get('alt', 'Pexels Image'),
                "author_name": photo['photographer']['name'],
                "author_username": photo['photographer']['username']
            }
            images.append(image)
    else:
        # Fallback to HTML parsing
        articles = soup.find_all('article', class_='photo')
        images = []
        for article in articles:
            img = article.find('img')
            if img:
                display_url = img['src']
                if 'srcset' in img:
                    download_url = img['srcset'].split(',')[0].split(' ')[0].split('?')[0]
                else:
                    download_url = display_url
                alt_text = img.get('alt', 'Pexels Image')
                author_link = article.find('a', class_='photo-item__photographer')
                if author_link:
                    author_name = author_link.text.strip()
                    author_username = author_link['href'].split('/')[-1]
                else:
                    author_name = "Unknown"
                    author_username = "unknown"
                photo_link = article.find('a', class_='js-photo-link')
                image_id = photo_link['href'].split('/')[-2] if photo_link else "unknown"
                image = {
                    "id": image_id,
                    "display_url": display_url,
                    "download_url": download_url,
                    "alt_text": alt_text,
                    "author_name": author_name,
                    "author_username": author_username,
                }
                images.append(image)

    driver.quit()
    return images

# Example usage for page 1
search_term = "face"
page = 1
images = scrape_pexels_images(search_term, page)
print(f"Fetched {len(images)} images from page {page}")
for img in images[:5]:
    print(img)

# Download the images
for i, image in enumerate(images):
    try:
        response = requests.get(image['download_url'], stream=True)
        if response.status_code == 200:
            with open(f'image_{i}.jpg', 'wb') as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded image {i} with ID {image['id']}")
        else:
            print(f"Failed to download image {i}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {i}: {e}")