import os
import requests
import dotenv


dotenv.load_dotenv()
webhook_url = os.environ["webhook_url"]
thread_id = "990434052538507294"
webhook_url += f"?thread_id={thread_id}"
content = "test send message to threads"
requests.post(webhook_url, json={"content": content})
