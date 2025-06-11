# 1. Verify server is reachable
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

# 2. Configure Chrome with essential flags
chrome_options = Options()
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--allow-insecure-localhost")
chrome_options.add_argument("--disable-web-security")
chrome_options.add_argument("start-maximized")
chrome_options.add_argument("disable-infobars")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--host-resolver-rules=MAP * 127.0.0.1")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# 3. Initialize driver with explicit service
service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 4. Navigate using JavaScript to bypass URL handling issues

driver.get("http://localhost:5000")

size = driver.get_window_size()
width = size["width"]
height = size["height"]

actions = ActionChains(driver)

canvas = driver.find_element("tag name", "canvas")

driver.execute_script("""
window.__moveCursorAbsolute = function(x, y) {
    // Ensure coordinates are within viewport
    x = Math.max(0, Math.min(x, window.innerWidth - 1));
    y = Math.max(0, Math.min(y, window.innerHeight - 1));

    // Get element at position
    var element = document.elementFromPoint(x, y);
    if (!element) element = document.body;

    // Create and dispatch event
    var event = new MouseEvent('mousemove', {
        view: window,
        bubbles: true,
        cancelable: true,
        clientX: x,
        clientY: y,
        screenX: x + window.screenX,
        screenY: y + window.screenY
    });
    element.dispatchEvent(event);
};
""")

def move_to_position(x, y):
    px = int(0.5 * (width - 1) * (x + 1))
    py = int(0.5 * (height - 1) * (y + 1))
    driver.execute_script(f"window.__moveCursorAbsolute({px}, {py});")


# Center the cursor.

move_to_position(0, 0)
input()
move_to_position(0.25, 0)
input()
move_to_position(0.5, 0)
input()
move_to_position(0.75, 0)
input()
move_to_position(1, 0)
input()
move_to_position(-0.25, 0)
input()
move_to_position(-0.5, 0)
input()
move_to_position(-0.25, 0)
input()
move_to_position(-1, 0)
input()


driver.quit()

