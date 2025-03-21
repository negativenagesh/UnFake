from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os
import tempfile
import time
import urllib.parse

def scrape_unsplash_images(search_term, page=1):
    """
    Scrape images from Unsplash based on search term and page number
    
    Args:
        search_term (str): The search query for Unsplash
        page (int, optional): Page number to scrape. Defaults to 1.
        
    Returns:
        list: List of dictionaries containing image data
    """
    # Create a temporary directory for Chrome user data
    temp_dir = tempfile.mkdtemp()

    # Set up Chrome options
    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"--user-data-dir={temp_dir}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222")

    # Initialize the Chrome driver
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Error initializing driver: {e}")
        return []

    # Format search term for URL
    formatted_search = urllib.parse.quote(search_term.strip())
    
    # Target URL with page parameter if needed
    if page > 1:
        url = f"https://unsplash.com/s/photos/{formatted_search}?page={page}"
    else:
        url = f"https://unsplash.com/s/photos/{formatted_search}"

    # Fetch the page
    try:
        driver.get(url)
        # Scroll down to load more images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for images to load
    except Exception as e:
        print(f"Error fetching page: {e}")
        driver.quit()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return []

    # Get the rendered HTML
    html_content = driver.page_source
    driver.quit()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all relevant image elements
    image_data = []
    figure_elements = soup.find_all('figure')
    
    for figure in figure_elements:
        try:
            img_tag = figure.find('img')
            
            if img_tag and img_tag.get('src') and 'photo-' in img_tag.get('src'):
                image_url = img_tag['src']
                
                # Get high-resolution version by removing query parameters
                parsed_url = urllib.parse.urlparse(image_url)
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                
                # Try to get alt text
                alt_text = img_tag.get('alt', 'Unsplash Image')
                
                # Try to get author information
                parent_element = figure.parent.parent
                author_element = parent_element.find('a', {'rel': 'nofollow'})
                author_name = author_element.text.strip() if author_element else 'Unknown'
                
                # Add image to results
                image_item = {
                    'url': clean_url,
                    'alt_text': alt_text,
                    'author': author_name,
                    'download_url': clean_url
                }
                
                image_data.append(image_item)
                
                # Limit to 10 images
                if len(image_data) >= 10:
                    break
        except Exception as e:
            print(f"Error extracting image data: {e}")
            continue

    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return image_data