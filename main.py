from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

geckodriver_path = './geeko/geckodriver.exe'
# Create a Service object pointing to the geckodriver executable
service = Service(geckodriver_path)
options = Options()

# Initialize the WebDriver
driver = webdriver.Firefox(service=service, options=options)
timeout = 10

# Open the website
driver.get("https://www.betpawa.cm/")

# Wait until the login button is clickable and click it
login_button_xpath = "//a[contains(.,'Login')]"
login_button = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, login_button_xpath))
)
login_button.click()

# Wait until the PIN input field is visible
pin_input_xpath = "//input[@name='pin']"  # Update this XPath if needed
pin_input = WebDriverWait(driver, timeout).until(
    EC.visibility_of_element_located((By.XPATH, pin_input_xpath))
)

# Loop until 4 characters are entered
while True:
    pin_value = pin_input.get_attribute('value')
    if len(pin_value) == 4:
        break
    time.sleep(0.5)  # Check every half second

# Wait for an additional 5 seconds after 4 characters are entered
time.sleep(5)

# Find and click the "Log In" button
submit_login_xpath = "//button[contains(.,'Log In')]"  # Update this XPath to the correct one for the "Log In" button
submit_login_button = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, submit_login_xpath))
)
submit_login_button.click()

# Continue with the rest of your code
