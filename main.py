from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from decimal import Decimal

## Machine Learning Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

geckodriver_path = './geeko/geckodriver.exe'
# Create a Service object pointing to the geckodriver executable
service = Service(geckodriver_path)
options = Options()

# Initialize the WebDriver
driver = webdriver.Firefox(service=service, options=options)
timeout = 18


# Open the website
driver.get("https://www.betpawa.cm/")

# Wait until the login button is clickable and click it
login_button_xpath = "//a[contains(.,'Login')]"
login_button = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, login_button_xpath))
)
login_button.click()

# Wait until the PIN input field is visible and then loop through
pin_input_xpath = "//input[contains(@type,'password')]"  
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

# Find Aviator game in Casino and click on it
aviator_game_xpath = "//div[@class='card-item-text'][contains(.,'Aviator')]"
aviator_game = WebDriverWait(driver, timeout).until(
    EC.element_to_be_clickable((By.XPATH, aviator_game_xpath))
)
aviator_game.click()

balance_amount_xpath = "//span[contains(@class,'amount font-weight-bold')]"
stake_amount_xpath = "(//input[@type='text'])[1]"

#read_amount = driver.find_element(By.XPATH, read_amount_xpath)
read_balance = WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, balance_amount_xpath))
)
balance_value = float(read_balance.text)
print(balance_value)


# Setstake amount

stake_amount = 0.33 * int(balance_value)
stake_input_xpath = "//input[contains(@type,'text')])[1]"
bet_button_xpath = "(//span[contains(.,'Bet  XAF')])[1]"


# Get the balance amount, ensure it's parsed into a usable format (e.g., removing currency symbols or commas)
balance_amount = WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, balance_amount_xpath))
)
amount_text = balance_amount.text.replace('XAF', '').replace(',', '').strip()  # Adjust this line based on the actual text format
amount_value = float(amount_text) if amount_text else 0

# Calculate the stake amount as 33% of the balance
stake_amount = 0.33 * amount_value


################## Start and Stop point ########################

# Read the last result 
last_result_xpath = "(//div[@class='bubble-multiplier font-weight-bold'])[1]"
def get_last_result_text(driver, xpath):
    try:
        element = driver.find_element(By.XPATH, xpath)
        return element.text.strip()
    except Exception as e:
        print(f"Error retrieving text: {e}")
        return None
    
last_known_result = get_last_result_text(driver, last_result_xpath)

while True:
    current_result = get_last_result_text(driver, last_result_xpath)
    
    if current_result != last_known_result:
        print("Change detected! Proceeding with the script...")
        break
    else:
        print("No change detected, checking again...")
        time.sleep(1)  # Wait 1 second before checking again

    last_known_result = current_result





##################                Read past results      ############
# Find History icon and click on it
history_icon_xpath = "//div[@class='history-icon']"
history_icon = WebDriverWait(driver, timeout).until(
    EC.presence_of_element_located((By.XPATH, history_icon_xpath))
)
history_icon.click()

# Find all elements with class 'bubble-multiplier'
past_results = driver.find_elements(By.CLASS_NAME, 'bubble-multiplier')

# Process the elements to extract and clean the text
results = []
for element in past_results:
    result_item = element.text.replace("x", "").strip()
    if result_item:
        results.append(float(result_item))

# Print the results
print(results)


################ Predict the next result ########################
data = results  # Example data, replace with your actual dataset

# Convert the data to a PyTorch tensor
data_tensor = torch.FloatTensor(data)

# Create sequences of 30 numbers
seq_length = 8
sequences = [data_tensor[i:i + seq_length] for i in range(len(data_tensor) - seq_length)]
targets = data_tensor[seq_length:]

# Prepare the dataset and dataloader
# Ensure that sequences and targets have the same size
assert len(sequences) == len(targets), "Size mismatch between sequences and targets"
dataset = TensorDataset(torch.stack(sequences), targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Instantiate the model
model = LSTMModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ...

# Training loop
num_epochs = 30  # Increase the number of epochs
for epoch in range(num_epochs):
    for seqs, targets in loader:
        # Reshape sequences for LSTM input
        seqs = seqs.view(-1, seq_length, 1)

        # Forward pass
        outputs = model(seqs)
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# ...


# Predict the next number in the sequence
# Here, we use the last sequence from the data for prediction
last_seq = data_tensor[-seq_length:].view(1, seq_length, 1)  # Ensure correct dimension for a single sequence
predicted_number = model(last_seq).item()
print(f'Predicted next number: {predicted_number}')

if predicted_number > 1.2:
    print("We might Stake")
else:
    print("No luck, we try again")

# # Find the stake input field
# stake_input = WebDriverWait(driver, timeout).until(
#     EC.presence_of_element_located((By.XPATH, stake_input_xpath))
# )

# # Clear the stake input field
# stake_input.clear()

# # Input the calculated stake amount
# stake_input.send_keys(str(stake_amount))

# # Wait for the bet button to be clickable
# bet_button = WebDriverWait(driver, timeout).until(
#     EC.element_to_be_clickable((By.XPATH, bet_button_xpath))
# )

# # Click on the bet button
# bet_button.click()

