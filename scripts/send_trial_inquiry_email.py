#!/usr/bin/env python3
import argparse
import json
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

from base.gpt_client import GPTClient

# Add argument parser
parser = argparse.ArgumentParser(
    description="Send a test email with optional debug logging."
)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument(
    "--trial-data", required=True, help="Path to the trial JSON data file"
)
parser.add_argument(
    "--ctd-id", required=True, help="ClinicalTrials.gov ID of the trial"
)
parser.add_argument(
    "--lang",
    choices=["en", "zh"],
    default="en",
    help="Language for the email title (en: English, zh: Chinese)",
)
parser.add_argument(
    "--openai-api-key",
    help="OpenAI API key (overrides OPENAI_API_KEY environment variable)",
)
args = parser.parse_args()

# Configure logging
logging_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_trial_title(trial_data_path, ctd_id):
    """Read the trial data and return the brief title for the given CTD ID."""
    try:
        with open(trial_data_path, "r") as f:
            trials = json.load(f)

        for trial in trials:
            if trial.get("identification", {}).get("nct_id") == ctd_id:
                return trial.get("identification", {}).get(
                    "brief_title", "No title found"
                )

        raise ValueError(f"Trial with CTD ID {ctd_id} not found in the data file")
    except Exception as e:
        logger.error(f"Error reading trial data: {str(e)}")
        raise


def generate_email_title(trial_title, ctd_id, lang, gpt_client):
    """Generate an email title using GPT based on the trial information and language preference."""
    prompt = f"""Generate an email title for a clinical trial information inquiry from the perspective of a cancer patient.

Trial Title: {trial_title}
Trial ID: {ctd_id}
Language: {'Simplified Chinese' if lang == 'zh' else 'English'}

Rules:
- Keep it concise and empathetic
- Include the trial ID
- Include the trial title
- Make it clear it's an information inquiry
- Keep the drug names untranslated

Return ONLY the title text without any additional text or formatting."""

    system_role = "You are a cancer patient seeking information about clinical trials."

    try:
        title, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role=system_role,
            temperature=0.1,
        )
        return title.strip(), cost
    except Exception as e:
        logger.error(f"Failed to generate email title: {str(e)}")
        return f"Clinical Trial Information Inquiry - {ctd_id}", 0.0


def send_test_email():
    # Load environment variables
    load_dotenv()

    # Initialize GPT client with API key from command line or environment
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable"
        )

    gpt_client = GPTClient(
        api_key=openai_api_key,
        cache_size=100000,
        temperature=0.1,
        max_retries=3,
    )

    # Get trial title
    trial_title = get_trial_title(args.trial_data, args.ctd_id)
    logger.info(f"send_test_email: Found trial title: {trial_title}")

    # Generate email title using GPT
    email_title, gpt_cost = generate_email_title(
        trial_title, args.ctd_id, args.lang, gpt_client
    )
    logger.info(f"send_test_email: Generated email title: {email_title}")

    # Email configuration
    sender_email = os.getenv("GMAIL_USER")
    logger.debug(f"send_test_email: Sender email configured: {sender_email}")
    sender_password = os.getenv(
        "GMAIL_APP_PASSWORD"
    )  # Use App Password, not regular password
    logger.debug(f"send_test_email: Sender password configured: {sender_password}")
    receiver_email = os.getenv("GMAIL_USER")  # For testing, sending to same email

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = email_title

    # Email body
    body = f"""
    This is a test email from the ClinTrialFinder application.
    
    Trial: {trial_title}
    CTD ID: {args.ctd_id}
    
    If you're receiving this, the email functionality is working correctly!
    
    Best regards,
    ClinTrialFinder Team
    """

    message.attach(MIMEText(body, "plain"))

    try:
        # Create SMTP session
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # Login
        server.login(sender_email, sender_password)
        logger.info("Successfully logged into SMTP server")

        # Send email
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        logger.info(f"Test email sent successfully to {receiver_email}")

        # Close session
        server.quit()
        logger.debug("SMTP session closed")

    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}", exc_info=True)


if __name__ == "__main__":
    send_test_email()
