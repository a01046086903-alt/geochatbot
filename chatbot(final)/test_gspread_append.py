import gspread
import traceback

print("Testing Google Sheets append_row...")
try:
    gclient = gspread.service_account(filename='google_creds.json')
    gsheet = gclient.open("ChatBot_Logs").sheet1
    print("Authentication and open successful. Attempting to append a row...")
    
    # Try to append a test row
    gsheet.append_row(["Test_Time", "10101", "Test Name", "Test Query", "Test Response", "Test Source"])
    print("Append row successful!")
except Exception as e:
    print("FAILED with exception:")
    traceback.print_exc()
