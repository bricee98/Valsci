import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from app.config import settings  # Import settings directly instead of using Flask config

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.enable_notifications = getattr(settings, 'ENABLE_EMAIL_NOTIFICATIONS', False)
        self.smtp_server = getattr(settings, 'SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.sender_email = getattr(settings, 'EMAIL_SENDER', '')
        self.app_password = getattr(settings, 'EMAIL_APP_PASSWORD', '')
        self.base_url = getattr(settings, 'BASE_URL', 'http://localhost:5000')

    def send_batch_start_notification(self, recipient_email, batch_id, num_claims, review_type):
        if not self.enable_notifications:
            logger.info("Email notifications are disabled")
            return False

        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = f"Valsci Batch Processing Started - Batch {batch_id}"

            body = f"""
            Your batch processing has started!

            Batch ID: {batch_id}
            Number of Claims: {num_claims}
            Review Type: {review_type}

            You can monitor the progress at:
            {self.base_url}/progress?batch_id={batch_id}

            You will receive another email when processing is complete.

            Thank you for using Valsci!
            """

            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.send_message(message)

            logger.info(f"Successfully sent start notification email to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send start notification email: {str(e)}")
            return False

    def send_batch_completion_notification(self, recipient_email, batch_id, num_claims, review_type):
        if not self.enable_notifications:
            logger.info("Email notifications are disabled")
            return False

        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = f"Valsci Batch Processing Complete - Batch {batch_id}"

            results_url = f"{self.base_url}/batch_results?batch_id={batch_id}"

            body = f"""
            Your batch processing is complete!

            Batch ID: {batch_id}
            Number of Claims: {num_claims}
            Review Type: {review_type}

            You can view your results at:
            {results_url}

            Thank you for using Valsci!
            """

            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.send_message(message)

            logger.info(f"Successfully sent completion notification email to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send completion notification email: {str(e)}")
            return False

    def send_batch_error_notification(self, email: str, batch_id: str, error_message: str):
        if not self.enable_notifications:
            logger.info("Email notifications are disabled")
            return False

        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = email
            message["Subject"] = f"Valsci Batch Processing Error - Batch {batch_id}"

            body = f"""
            An error occurred during your batch processing.

            Batch ID: {batch_id}
            Error Message: {error_message}

            You can check the batch status at:
            {self.base_url}/progress?batch_id={batch_id}

            If this issue persists, please contact support.

            Thank you for using Valsci!
            """

            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.app_password)
                server.send_message(message)

            logger.info(f"Successfully sent error notification email to {email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send error notification email: {str(e)}")
            return False
