from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

chromeOptions = webdriver.ChromeOptions()
chromeOptions.add_experimental_option(
    "prefs", {"profile.managed_default_content_settings.images": 2}
)
chromeOptions.add_argument("--no-sandbox")
chromeOptions.add_argument("--disable-setuid-sandbox")

chromeOptions.add_argument("--remote-debugging-port=9222")  # this

chromeOptions.add_argument("--disable-dev-shm-using")
chromeOptions.add_argument("--disable-extensions")

# Automatically downloads the correct chromedriver
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()),
    options=chromeOptions,
)
driver.get("https://google.com")
