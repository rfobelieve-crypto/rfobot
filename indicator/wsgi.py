"""WSGI entry point for gunicorn."""
from indicator.app import app, start_scheduler

start_scheduler()
