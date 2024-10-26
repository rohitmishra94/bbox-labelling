# import ngrok python sdk
import ngrok
import time
import os
os.environ['NGROK_AUTHTOKEN']='2kf8QbJKaDYzritaQpcqzNt6XcD_76eRHYpyMZPoE67yWRvkA'
# Establish connectivity
listener = ngrok.forward(5000, authtoken_from_env=True)

# Output ngrok url to console
print(f"Ingress established at {listener.url()}")

# Keep the listener alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Closing listener")
