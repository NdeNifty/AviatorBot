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
timeout = 8


# Open the website
driver.get("https://www.betpawa.cm/")

# Wait until the login button is clickable and click it
login_button_xpath = "//a[contains(.,'Login')]"
login_button = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, login_button_xpath))
)
login_button.click()

# Wait until the PIN input field is visible
pin_input_xpath = "//input[@name='password']"  # Update this XPath if needed
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
submit_login_xpath = "//input[contains(@value,'Log In')]"  # Update this XPath to the correct one for the "Log In" button
submit_login_button = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, submit_login_xpath))
)
submit_login_button.click()
time.sleep(5)

# Find the Casino Menu and click on it
casino_menu_item_xpath = "//span[@class='menu-text name'][contains(.,'Casino')]"
casino_menu_item = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, casino_menu_item_xpath))
)
#casino_menu_item = driver.find_element(By.XPATH, casino_menu_item_xpath)
casino_menu_item.click()

# Find Aviato game in Casino and click on it
aviator_game_xpath = "//div[@class='card-item-text'][contains(.,'Aviator')]"
aviator_game = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, aviator_game_xpath))
)
aviator_game.click()

balance_amount_xpath = "//span[contains(@class,'amount font-weight-bold')]"
stake_amount_xpath = "(//input[@type='text'])[1]"

#read_amount = driver.find_element(By.XPATH, read_amount_xpath)
read_amount = WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, balance_amount_xpath))
)
amount_value = read_amount.text
print(amount_value)


# Setstake amount

stake_amount = 0.33 * int(amount_value)
stake_input_xpath = "//input[contains(@type,'text')])[1]"
bet_button_xpath = "(//span[contains(.,'Bet 1.00 XAF')])[1]"


# Get the balance amount, ensure it's parsed into a usable format (e.g., removing currency symbols or commas)
balance_amount = WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, balance_amount_xpath))
)
amount_text = balance_amount.text.replace('XAF', '').replace(',', '').strip()  # Adjust this line based on the actual text format
amount_value = float(amount_text) if amount_text else 0

# Calculate the stake amount as 33% of the balance
stake_amount = 0.33 * amount_value

# Find the stake input field
stake_input = WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, stake_input_xpath))
)

# Clear the stake input field
stake_input.clear()

# Input the calculated stake amount
stake_input.send_keys(str(stake_amount))

# Wait for the bet button to be clickable
bet_button = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, bet_button_xpath))
)

# Click on the bet button
bet_button.click()

# ... continue with the rest of your code



# Continue with the rest of your code
