from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
import time
import asyncio
from langchain_community.document_loaders import UnstructuredURLLoader
import random

import json
from local_search.utils.config import config
from src.utils.logger import get_custom_logger
from .constants import (
    GET_IMAGE_SEARCH_ICON,
    UPLOAD_IMAGE_FILE,
    SCROLL_PAGE,
    GET_IMAGE_TEXT_LABEL,
    GET_IMAGE_ELEMENTS,
    ERROR_RESPONSE_WEB_SEARCH,
    ERROR_RESPONSE_IMAGE_WEB_SEARCH,
    GET_IMAGE_LOCATOR,
    ESCAPE_KEY
)

logger = get_custom_logger("search_tool")

user_agent = random.choice([
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.94 Chrome/37.0.2062.94 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/8.0.8 Safari/600.8.9",
    "Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36"
])

async def langchain_loader(real_url: str):
    # Fetch page content using LangChain
    loader = UnstructuredURLLoader(urls=[real_url])
    docs = loader.load()
    content = docs[0].page_content if docs else "No content available"
    return content


async def search_bing(query: str, num_results: int = 3) -> dict:


    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,  # Run in visible mode now
            channel="chrome"
        )
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()

        await page.goto(config.BING_SEARCH_URL)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(random.uniform(0.3, 1))

        try:
            await page.click("button:has-text('Accept')", timeout=5000)
            await asyncio.sleep(random.uniform(0.3, 1))

        except:
            pass

        try:

            await page.wait_for_selector("input#sb_form_q, textarea#sb_form_q", timeout=10000)
            await asyncio.sleep(random.uniform(0.3, 1))

        except Exception as e:
            logger.error(f"Error waiting for search box: {e}", exc_info=True)
            await browser.close()
            return ERROR_RESPONSE_WEB_SEARCH

        try:
            search_box_selector = "input[name='q'], textarea[name='q']"
            await page.fill(search_box_selector, query)
            await asyncio.sleep(random.uniform(0.3, 1))
            await page.press(search_box_selector, "Enter")
            await asyncio.sleep(random.uniform(0.3, 1))
            await page.wait_for_selector("#b_results", timeout=10000)
            # await asyncio.sleep(3)
            await asyncio.sleep(random.uniform(0.3, 1))
            await page.mouse.wheel(0, random.randint(100, 400))

            search_results = []
            result_elements = page.locator("#b_results .b_algo")
            count = await result_elements.count()

            for i in range(count):
                try:
                    title_element = result_elements.nth(i).locator("h2 a")
                    await asyncio.sleep(random.uniform(0.3, 1))
                    link = await title_element.get_attribute("href", timeout=5000)
                    if not link:
                        continue
                    try:
                        new_page = await context.new_page()
                        await new_page.goto(link, wait_until="domcontentloaded", timeout=10000)
                        await asyncio.sleep(random.uniform(0.3, 1))
                        # time.sleep(4)
                        # await asyncio.sleep(random.uniform(0.3, 0.6))
                        real_url = new_page.url
                        await new_page.close()
                    except Exception as e:
                        logger.error(f"Error occurred while opening new page: {e}", exc_info=True)
                        continue

                    get_content = await langchain_loader(real_url)

                    result = {
                        "web_link": real_url,
                        "page_content": get_content
                    }

                    if len(search_results) >= num_results:
                        break
                    
                    search_results.append(result)

                    # search_results.append(result)

                    logger.info(f"\nResult {len(search_results)}:")
                    logger.info(f"URL: {real_url}")
                    # logger.info(f"Preview: {get_content[:30]}...\n")

                    
                except Exception as e:
                    logger.error(f"Error processing result {i+1}: {str(e)}")
                    continue

            await browser.close()

            return search_results

        except Exception as e:
            logger.error(f"Error occurred while searching: {e}", exc_info=True)
            return ERROR_RESPONSE_WEB_SEARCH

def image_search(image_path: str, num_results: int = 3):
    logger.info(" [ Image Web Search initiated... ] ")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to Google Images search
        page.goto(config.IMAGE_SEARCH_URL_GOOGLE)

        # Click on the camera icon to upload an image
        page.locator(GET_IMAGE_SEARCH_ICON).click()

        time.sleep(17)

        try:

            # Upload the image
            file_input = page.locator(UPLOAD_IMAGE_FILE)
            file_input.set_input_files(image_path)
            time.sleep(18)

        except Exception as e:
            logger.error(f"Error occurred while uploading image: {e}", exc_info=True)

        # Wait for images to load
        page.wait_for_selector("img")

        time.sleep(7)

        # Extract the first three image URLs
        page.evaluate(SCROLL_PAGE)

        # wait for images to load
        page.wait_for_timeout(2000)

        image_elements = page.locator(GET_IMAGE_ELEMENTS).element_handles()

        try:
            results = []
            for img in image_elements[:num_results]:
                try:
                    src = img.get_attribute("src")
                    text_label = page.evaluate(GET_IMAGE_TEXT_LABEL, img)

                    img.click()
                    page.wait_for_selector("img[src]", state="visible", timeout=50000)  # Wait for the high-res image to appear

                    page.wait_for_timeout(500)

                    # Get high-res image only for this specific clicked image
                    high_res_img = page.locator(GET_IMAGE_LOCATOR).first  # Select first if multiple exist
                    image_url = high_res_img.get_attribute("src") if high_res_img else None # get image url


                    page.mouse.wheel(0, 200) ##scroll
                    page.wait_for_timeout(300)

                    if src:
                        results.append({
                            "text": text_label,
                            "image_source": src,
                            "image_url": image_url
                        })
                    page.keyboard.press(ESCAPE_KEY)
                    time.sleep(20)
                except Exception as e:
                    logger.warning(f"Exception occurred while fetching images, retrying : {e}", exc_info=True)
                    continue
        except Exception as e:
            logger.error(f"Error occurred while fetching image source: {e}", exc_info=True)
            return ERROR_RESPONSE_IMAGE_WEB_SEARCH
        time.sleep(4)
        browser.close()
        return results
