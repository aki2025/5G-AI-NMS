from kafka import KafkaProducer
import json
import requests
import logging
from flask import Flask
import tweepy
import osmnx as ox
from cachetools import TTLCache

app = Flask(__name__)
logging.basicConfig(filename='agent_code/agent_log.txt', level=logging.INFO)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda v: json.dumps(v).encode('utf-8'))
cache = TTLCache(maxsize=100, ttl=300)  # Cache external data for 5 minutes

with open('agent_code/config.json', 'r') as f:
    config = json.load(f)

def fetch_external_data(city, lat, lon):
    """Fetch and cache external data."""
    cache_key = f"{city}_{lat}_{lon}"
    if cache_key in cache:
        return cache[cache_key]

    external = {}
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={config['api_keys']['openweathermap']}&units=metric"
        response = requests.get(weather_url, timeout=5)
        weather_data = response.json()
        external["temperature"] = weather_data["main"]["temp"]
        external["precipitation"] = weather_data.get("rain", {}).get("1h", 0)
    except Exception:
        external["temperature"] = 20.0
        external["precipitation"] = 0.0

    try:
        auth = tweepy.OAuthHandler(config['api_keys']['twitter_consumer_key'], config['api_keys']['twitter_consumer_secret'])
        auth.set_access_token(config['api_keys']['twitter_access_token'], config['api_keys']['twitter_access_token_secret'])
        api = tweepy.API(auth)
        query = "5G OR outage OR network near:{city}"
        tweets = api.search_tweets(q=query, count=100)
        external["social_trend_score"] = len(tweets) / 100.0
    except Exception:
        external["social_trend_score"] = 0.0

    try:
        G = ox.graph_from_point((lat, lon), dist=1000, network_type='all')
        external["urban_density"] = len(G.nodes) / 1000.0
    except Exception:
        external["urban_density"] = 0.7

    external["congestion_score"] = 0.5
    external["event_density"] = 0.0  # Placeholder
    cache[cache_key] = external
    return external

@app.route('/ingest', methods=['POST'])
def ingest_log():
    """Fetch, enrich, and stream log to Kafka."""
    url = "https://example.com/realtime_logs.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        log = response.json()
        city, lat, lon = "London", 51.5074, -0.1278  # Example coordinates
        log['external'] = fetch_external_data(city, lat, lon)
        producer.send('network_logs', log)
        producer.flush()
        logging.info(f"Ingested log: {log}")
        return {"status": "success"}, 200
    except Exception as e:
        logging.error(f"Failed to ingest log: {e}")
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)