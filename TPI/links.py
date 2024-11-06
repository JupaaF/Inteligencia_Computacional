import re
import pandas as pd
import numpy as np
# Expresión regular para detectar enlaces
url_pattern = r"(http[s]?://\S+|www\.\S+)"
hashtag_pattern = r"#\w+"
mention_pattern = r"@\w+"
def contains_link_hashtag_mention(text):
    return (bool(re.search(url_pattern, text)),bool(re.search(mention_pattern, text)),bool(re.search(hashtag_pattern, text)))
# Ejemplo de conjunto de datos de tweets

def remove_links(text):
    return re.sub(url_pattern, '', text)

tweets = [
    "Check out the latest news on our website at https://example.com! Mentioning @technews for more info.",
    "Enjoying a sunny day at the beach @beachlover",
    "Here's an interesting article on AI: www.technews.ai @ai_expert",
    "No link here, just sharing some thoughts! @friend1",
    "Breaking news: huge event happening now! @news_channel",
    "Learn more at https://bit.ly/3xyz or contact us. @support",
    "Did you see the new features on the app? @app_updates",
    "Visit us at www.ourwebsite.com for more details.",
    "This is just a normal tweet without any mentions.",
    "Follow the story at https://newsupdate.com/latest-news @storyteller"
]

# Aplicar la función para verificar si cada tweet contiene enlaces
results = [(contains_link_hashtag_mention(tweet),remove_links(tweet)) for tweet in tweets]
print(results)