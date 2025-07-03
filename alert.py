# Email sending function
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

def send_email(subject, message, to_email):
    load_dotenv()
    from_email = os.getenv("FROM_EMAIL")
    # Generate it on and named it Mail if you use mail https://myaccount.google.com/apppasswords
    app_password = os.getenv("APP_PASSWORD")

    # Create email
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Attach the message body (plain text or HTML)
    msg.attach(MIMEText(message, "plain"))
    try:
        # Connect and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        print(f"[+] Email sent to {to_email}")

    except Exception as e:
        print(f"[!] Failed to send email: {e}")
        
        
# SMS sending function (using Twilio)
from twilio.rest import Client
def send_sms(message, to_number, account_sid, auth_token, from_number):
    try:
        client = Client(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )
        print(f"[+] SMS sent to {to_number}")
    except Exception as e:
        print(f"[!] Failed to send SMS: {e}")


# Telegram Bot Notification
import requests
def send_telegram(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"  # Optional: allows formatting (e.g., <b>bold</b>)
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("[+] Telegram alert sent successfully.")
        else:
            print(f"[!] Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"[!] Telegram error: {e}")