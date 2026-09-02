import gspread
import os

print("Testing Google Sheets API connection...")
try:
    if not os.path.exists('google_creds.json'):
        print("Error: google_creds.json does not exist!")
    else:
        gclient = gspread.service_account(filename='google_creds.json')
        print("Authentication successful! Trying to open 'ChatBot_Logs'...")
        gsheet = gclient.open("ChatBot_Logs").sheet1
        print("Success! Sheet 'ChatBot_Logs' opened.")
except Exception as e:
    print(f"FAILED with error: {e}")
