import asyncio
from telegram import Bot
import geocoder
from datetime import datetime

async def send_message(token, chat_id, photo_path):
    now = datetime.now()
    formatted_now = now.strftime(" DATE: %m/%d/%Y, TIME: %I:%M:%S %p")
    g = geocoder.ip('me')
    lat, lng = g.latlng
    bot = Bot(token=token)
    location_url = f"https://www.google.com/maps/place/{lat},{lng}"
    finalmessage = f"VIOLENCE DETECTED !!! \n {location_url} \n {formatted_now}"
    await bot.send_message(chat_id=chat_id, text=location_url)
    await bot.send_message(chat_id=chat_id, text=formatted_now)
    with open(photo_path, 'rb') as photo:
       await bot.send_photo(chat_id=chat_id, photo=photo,caption=finalmessage)

# Use your Bot's API token and chat ID to send a message
asyncio.run(send_message('7178217143:AAHdiwlUSda_M1ZG8VK3NOJhotGHFA2npe4', '-1002029882964', 'E:\saai\\be6a27d4-f0b2-447e-8644-ff62fc8658d7.jpg'))
