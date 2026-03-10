from flask import Flask
from app.config.settings import Config  # Updated import path

def create_app(config_class=Config):
    if hasattr(config_class, "validate_config"):
        config_class.validate_config()
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions here

    # Register blueprints here
    from app.api.routes import api
    app.register_blueprint(api)

    return app
