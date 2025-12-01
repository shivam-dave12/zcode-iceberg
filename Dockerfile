# Use specific Python version matching your VSCode
FROM 126919341356.dkr.ecr.ap-south-1.amazonaws.com/aham:1
# Copy all application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the Telegram controller (not main.py directly)
CMD ["python", "-u", "telegram_bot_controller.py"]

