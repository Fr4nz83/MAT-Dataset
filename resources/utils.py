import re
import random
def extract_first_occurrence(text):
    text = str(text)
    if '"' not in text:
        return text
    else:
        match = re.search(r'"(.*?)"', text)
        return match.group(1) if match else text

def clean_posts(x):
    x = str(x)
    strip_list = ['\n','assistant()','assistant)','assistant-','assistant',"'\'title\'","\n\'text\':",'PostId', 'assistant\n\n', 'Post\n', 'Post \n', 'Post:','Post has been removed. Here is a rewritten version:', "POST"]
    for item in strip_list:
        x = x.strip().replace(item, '')
    x = x.replace('\n', ' ')
    if x.startswith('"'): 
        x = x.replace('"', '')
    if x.startswith("'"):
        x = x[1:]
    if x.endswith("'"):
        x = x[:-1]
    x = x.split('\n')[0]
    if '#' not in x.split('.')[-1]:
        x = "".join(x.split('.')[:-1])
    if x.endswith('#'):
        x = x[:-1]
    return x


def generate_tweet_messages(row, sentiment='positive'):
    # Extract relevant information
    place = row['name'] if row['name'] else 'N/A'
    category = row['category'] or 'N/A'
    arrival = row.get('datetime', 'N/A')
    leaving = row.get('leaving_datetime', 'N/A')

    gender = ['male', 'female', 'other']
    age = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    ethnicity = ['white', 'black', 'hispanic', 'asian', 'other']
    social = ['Twitter', 'Instagram', 'Facebook', 'Tripadvisor']

    metadata = {
        'place': place,
        'category': category,
        'sentiment': sentiment,
        'gender': random.choice(gender),
        'age': random.choice(age),
        'ethnicity': random.choice(ethnicity),
        'social': random.choice(social),
    }

    # System instructions
    system_message = (
        "You are a creative social media post generator. "
        "Your task is to write a short, engaging, and realistic social media post based on a user's visit to a Point of Interest (POI). "
        "Include the most important details: **Location** and **Category**. Reflect the user's **sentiment** in tone and style. "
        "Use hashtags when appropriate. Avoid repeating input literally—be expressive and natural. Do not start with phrases like 'I just visited' or 'I am at', 'Disappointing experience', 'Ugh'."
        "If some information is not available genrate a post without it, but do not mention that the information is missing. "
    )

    # User-provided content
    user_message = (
        f"Here is the information about the visit:\n"
        f"- Location: {metadata['place']}\n"
        f"- Category: {metadata['category']}\n"
        f"- Sentiment: {metadata['sentiment']}\n"
        f"- Gender: {metadata['gender']}\n"
        f"- Age: {metadata['age']}\n"
        f"- Ethnicity: {metadata['ethnicity']}\n"
        f"- Social Media: {metadata['social']}\n\n"
        f"Generate only the post, with no extra commentary.\n"
        f"Post:"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    return messages, metadata