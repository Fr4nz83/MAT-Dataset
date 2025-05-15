#%%
# === Import Libraries ===
import pandas as pd
import os
import re
import random
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#%%
# === Configuration ===

# Set HuggingFace Transformers cache directory
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'

# Specify which GPUs to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,4,5'

#%%
# === Helper Functions ===

def extract_first_occurrence(text):
    """
    Extracts the first quoted string in the given text.
    If no quotes are found, returns the text as is.
    """
    text = str(text)
    match = re.search(r'"(.*?)"', text)
    return match.group(1) if match else text

def clean_line(text, phrase='Assistant:'):
    """
    Cleans the model output by:
    - Removing everything before the given phrase (default: 'Assistant:')
    - Extracting the first quoted string
    - Removing common unwanted tokens
    """
    text = str(text)
    start = text.find(phrase)
    if start != -1:
        text = text[start + len(phrase):].lstrip()
    text = extract_first_occurrence(text)
    text = re.sub(r'assistant\n\n', '', text)
    return text

#%%
def generate_tweet_prompt(row, sentiment='positive', prompt=None):
    """
    Generates a prompt for tweet generation using metadata from a row.
    Randomizes user demographic and social media platform.
    """
    place = row['name'] if row['name'] else None
    category = row['category']
    
    # Simulated user metadata
    metadata = {
        'place': place,
        'category': category,
        'sentiment': sentiment,
        'gender': random.choice(['male', 'female', 'other']),
        'age': random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']),
        'ethnicity': random.choice(['white', 'black', 'hispanic', 'asian', 'other']),
        'social': random.choice(['Twitter', 'Instagram', 'Facebook', 'Tripadvisor'])
    }

    # Prompt template
    prompt = (
        f"You are a creative social media post generator. "
        f"Your task is to write a short, engaging, and realistic social media post based on the user's stop at Point of Interest (POI). "
        f"Include the most important details, especially the **Location** and **Category** of the POI. "
        f"Reflect the user's **sentiment** in the tone and style of the post. "
        f"\nHere is the information about the visit:\n"
        f"- Location: {place or 'N/A'}\n"
        f"- Category: {category or 'N/A'}\n"
        f"- Sentiment: {sentiment}\n"
        f"- Gender: {metadata['gender']}\n"
        f"- Age: {metadata['age']}\n"
        f"- Ethnicity: {metadata['ethnicity']}\n"
        f"- Social Media: {metadata['social']}\n"
        f"\nKeep the post natural, expressive, and in the style of a post. Avoid repeating the input literally—paraphrase and add personality. "
        f"Use hashtags when appropriate, especially for the location and category.\n"
        f'Try not start with "I just visited" or "I am at" or "Just spent".\n'
        f"\nNow, based on the given information, generate a new social media post.\n"
        f"Generate only the post without any additional text. \n\n"
        f"Post: \n"
    )

    return prompt, metadata

#%%
def generate_tweets(model, tokenizer, input_ids, attention_mask, batch_size=4, num_return_sequences=1):
    """
    Generate tweets using the model for each input prompt.
    """
    tweets = []
    model.eval()
    for i in tqdm(range(0, len(input_ids), batch_size)):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids[i:i+batch_size].to('cuda'),
                attention_mask=attention_mask[i:i+batch_size].to('cuda'),
                max_new_tokens=64,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.9,
                top_p=0.9
            )
        for output in outputs:
            tweet = tokenizer.decode(output, skip_special_tokens=True)
            tweets.append(tweet)
    return tweets

#%%
def generate_posts(df, model, tokenizer):
    """
    For each row in the dataset, generate both a positive and negative tweet,
    tokenize the prompts, and process them with the model.
    Returns a DataFrame of cleaned results.
    """
    # Storage
    tweets = []
    positive_prompts, negative_prompts = [], []
    positive_metadata, negative_metadata = [], []
    positive_inputs, negative_inputs = [], []

    # Generate prompts and tokenize
    for index, row in tqdm(df.iterrows()):
        pos_prompt, pos_meta = generate_tweet_prompt(row, sentiment='positive')
        neg_prompt, neg_meta = generate_tweet_prompt(row, sentiment='negative')

        positive_prompts.append(pos_prompt)
        negative_prompts.append(neg_prompt)
        positive_metadata.append(pos_meta)
        negative_metadata.append(neg_meta)

        pos_input = tokenizer(pos_prompt, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')
        neg_input = tokenizer(neg_prompt, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')

        positive_inputs.append(pos_input)
        negative_inputs.append(neg_input)

    # Stack input tensors
    pos_input_ids = torch.stack([x['input_ids'] for x in positive_inputs]).squeeze(1).to('cuda')
    pos_attention = torch.stack([x['attention_mask'] for x in positive_inputs]).squeeze(1).to('cuda')
    neg_input_ids = torch.stack([x['input_ids'] for x in negative_inputs]).squeeze(1).to('cuda')
    neg_attention = torch.stack([x['attention_mask'] for x in negative_inputs]).squeeze(1).to('cuda')

    # Generate tweets
    pos_tweets = generate_tweets(model, tokenizer, pos_input_ids, pos_attention, batch_size=16)
    neg_tweets = generate_tweets(model, tokenizer, neg_input_ids, neg_attention, batch_size=16)

    # Clean outputs
    def clean(texts):
        return [
            x.split('Post: \n')[-1]
             .replace('assistant\n\n','')
             .replace('Post\n','')
             .replace('Post \n','')
             .replace('Post:','')
            for x in texts
        ]

    cleaned_positive = clean(pos_tweets)
    cleaned_negative = clean(neg_tweets)

    # Create output DataFrame
    tweets_df = pd.DataFrame({
        'positive': cleaned_positive,
        'positive_metadata': positive_metadata,
        'positive_prompt': positive_prompts,
        'negative': cleaned_negative,
        'negative_metadata': negative_metadata,
        'negative_prompt': negative_prompts
    })

    return tweets_df

#%%
# === Argument Parsing ===

def parse_args():
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate tweets using a pre-trained model.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID for the pre-trained model.')
    return parser.parse_args()

#%%
# === Main Program ===

def main():
    # Parse input arguments
    args = parse_args()
    file_path = args.file_path
    output_path = args.output_path
    model_id = args.model_id

    # Load input data
    df = pd.read_csv(file_path)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # Generate posts
    tweets_df = generate_posts(df, model, tokenizer)

    # Save to CSV
    tweets_df.to_csv(output_path, index=False)
    print(f"Tweets saved to {output_path}")

# Run script
if __name__ == "__main__":
    main()