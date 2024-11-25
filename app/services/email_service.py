import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app
import logging

logger = logging.getLogger(__name__)

class EmailService:
    @staticmethod
    def send_batch_start_notification(recipient_email, batch_id, num_claims, review_type):
        if not current_app.config['ENABLE_EMAIL_NOTIFICATIONS']:
            logger.info("Email notifications are disabled")
            return False

        try:
            sender_email = current_app.config['EMAIL_SENDER']
            app_password = current_app.config['EMAIL_APP_PASSWORD']
            
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = f"Valsci Batch Processing Started - Batch {batch_id}"

            body = f"""
            Your batch processing has started!

            Batch ID: {batch_id}
            Number of Claims: {num_claims}
            Review Type: {review_type}

            You can monitor the progress at:
            {current_app.config.get('BASE_URL', 'NO URL SET')}/progress?batch_id={batch_id}

            You will receive another email when processing is complete.

            Thank you for using Valsci!
            """

            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(current_app.config['SMTP_SERVER'], current_app.config['SMTP_PORT']) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.send_message(message)

            logger.info(f"Successfully sent start notification email to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send start notification email: {str(e)}")
            return False

    @staticmethod
    def send_batch_completion_notification(recipient_email, batch_id, num_claims, review_type):
        if not current_app.config['ENABLE_EMAIL_NOTIFICATIONS']:
            logger.info("Email notifications are disabled")
            return False

        try:
            sender_email = current_app.config['EMAIL_SENDER']
            app_password = current_app.config['EMAIL_APP_PASSWORD']
            
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = f"Valsci Batch Processing Complete - Batch {batch_id}"

            results_url = f"{current_app.config.get('BASE_URL', 'NO URL SET')}/batch_results?batch_id={batch_id}"

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

            with smtplib.SMTP(current_app.config['SMTP_SERVER'], current_app.config['SMTP_PORT']) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.send_message(message)

            logger.info(f"Successfully sent completion notification email to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send completion notification email: {str(e)}")
            return False

    @staticmethod
    def send_batch_error_notification(email: str, batch_id: str, error_message: str):
        if not current_app.config['ENABLE_EMAIL_NOTIFICATIONS']:
            logger.info("Email notifications are disabled")
            return False

        try:
            sender_email = current_app.config['EMAIL_SENDER']
            app_password = current_app.config['EMAIL_APP_PASSWORD']
            
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = email
            message["Subject"] = f"Valsci Batch Processing Error - Batch {batch_id}"

            body = f"""
            An error occurred during your batch processing.

            Batch ID: {batch_id}
            Error Message: {error_message}

            You can check the batch status at:
            {current_app.config.get('BASE_URL', 'NO URL SET')}/progress?batch_id={batch_id}

            If this issue persists, please contact support.

            Thank you for using Valsci!
            """

            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(current_app.config['SMTP_SERVER'], current_app.config['SMTP_PORT']) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.send_message(message)

            logger.info(f"Successfully sent error notification email to {email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send error notification email: {str(e)}")
            return False
