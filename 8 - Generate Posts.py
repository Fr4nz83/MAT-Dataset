#!/usr/bin/env python3
#%%
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from resources.utils import generate_tweet_messages, clean_posts

# Configurazione
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

MODEL_ID = 'meta-llama/Llama-3.3-70B-Instruct'
BATCH_SIZE = 512
TRAJECTORIES_PATH = '/home/francomaria.nardini/raid/guidorocchietti/code/MAT-Dataset/data/enriched/'
MIN_OUTPUT_LENGTH = 5

# Caricamento modello
print("Caricamento modello...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
)

def generate_batch(input_ids, attention_mask, temperature=0.9):
    """Genera un singolo batch di output."""
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=64,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Rimuovi prompt e decodifica
    output_without_prompt = output[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(output_without_prompt, skip_special_tokens=True)

def generate_posts_with_smart_retry(inputs, key, max_retries=2):
    """Genera post con retry intelligente - prima batch, poi individuali solo per i falliti."""
    all_outputs = []
    
    for i in tqdm(range(0, len(inputs['input_ids']), BATCH_SIZE), desc=f'Generando per {key}'):
        batch_input_ids = inputs['input_ids'][i:i + BATCH_SIZE].to(model.device)
        batch_attention_mask = inputs['attention_mask'][i:i + BATCH_SIZE].to(model.device)
        
        # Prima prova: batch normale
        batch_outputs = generate_batch(batch_input_ids, batch_attention_mask)
        
        # Identifica output troppo corti
        failed_indices = []
        final_outputs = []
        
        for j, output in enumerate(batch_outputs):
            cleaned_output = output.strip()
            if len(cleaned_output) >= MIN_OUTPUT_LENGTH:
                final_outputs.append(cleaned_output)
            else:
                failed_indices.append(j)
                final_outputs.append(cleaned_output)  # Placeholder temporaneo
        
        # Retry solo per i falliti (se ce ne sono pochi)
        if failed_indices and len(failed_indices) <= BATCH_SIZE // 4:  # Solo se < 25% falliti
            print(f"Retry per {len(failed_indices)} output corti nel batch {i//BATCH_SIZE + 1}")
            
            for retry_attempt in range(max_retries):
                remaining_failed = []
                
                for idx in failed_indices:
                    single_input = batch_input_ids[idx:idx+1]
                    single_mask = batch_attention_mask[idx:idx+1]
                    
                    # Aumenta temperatura per retry
                    retry_output = generate_batch(single_input, single_mask, 
                                                temperature=0.9 + (retry_attempt + 1) * 0.15)
                    
                    cleaned_retry = retry_output[0].strip()
                    if len(cleaned_retry) >= MIN_OUTPUT_LENGTH:
                        final_outputs[idx] = cleaned_retry
                    else:
                        remaining_failed.append(idx)
                
                failed_indices = remaining_failed
                if not failed_indices:
                    break
            
            if failed_indices:
                print(f"Warning: {len(failed_indices)} output ancora troppo corti dopo retry")
        
        elif failed_indices:
            print(f"Troppi output corti ({len(failed_indices)}/{len(batch_outputs)}) - skipping retry per efficienza")
        
        all_outputs.extend(final_outputs)
    
    return all_outputs

def process_trajectory(key, df):
    """Processa una singola traiettoria."""
    print(f'Processing {key}... ({len(df)} righe)')
    
    # Genera messaggi
    positive_messages = [generate_tweet_messages(row, sentiment='positive') for _, row in df.iterrows()]
    negative_messages = [generate_tweet_messages(row, sentiment='negative') for _, row in df.iterrows()]
    
    # Prepara input
    def prepare_inputs(messages):
        chat_inputs = tokenizer.apply_chat_template(
            [msg[0] for msg in messages], return_tensors='pt', tokenize=False, 
            padding=True, truncation=True, max_length=1024, padding_side='left'
        )
        return tokenizer(chat_inputs, return_tensors='pt', padding=True, truncation=True, max_length=1024, padding_side='left')
    
    positive_inputs = prepare_inputs(positive_messages)
    negative_inputs = prepare_inputs(negative_messages)
    
    # Genera post con retry intelligente
    model.eval()
    positive_posts = generate_posts_with_smart_retry(positive_inputs, f"{key}-positive")
    negative_posts = generate_posts_with_smart_retry(negative_inputs, f"{key}-negative")
    
    print(f'Generati {len(positive_posts)} post positivi e {len(negative_posts)} negativi per {key}')
    
    return {
        'positive': {'posts': positive_posts, 'metadata': [msg[1] for msg in positive_messages]},
        'negative': {'posts': negative_posts, 'metadata': [msg[1] for msg in negative_messages]}
    }

def save_results(key, original_df, posts_data):
    """Salva i risultati in vari formati."""
    # Crea DataFrame
    posts_df = pd.DataFrame({
        'positive': posts_data['positive']['posts'],
        'negative': posts_data['negative']['posts'],
        'positive_metadata': posts_data['positive']['metadata'],
        'negative_metadata': posts_data['negative']['metadata']
    })
    
    # Pulisci post e combina
    posts_df['positive'] = posts_df['positive'].apply(clean_posts)
    posts_df['negative'] = posts_df['negative'].apply(clean_posts)
    final_df = pd.concat([original_df, posts_df], axis=1)
    
    # Rimuovi timezone
    for col in final_df.columns:
        if pd.api.types.is_datetime64tz_dtype(final_df[col]):
            final_df[col] = final_df[col].dt.tz_localize(None)
    
    # Salva
    base_path = os.path.join(TRAJECTORIES_PATH, f'{key}_posts')
    final_df.to_csv(f'{base_path}.csv', index=False)
    final_df.to_parquet(f'{base_path}.parquet', index=False)
    final_df.to_excel(f'{base_path}.xlsx', index=False)
    print(f'Salvato: {base_path}.*')

def select_pois(enriched_stops):
    enriched_stops_with_poi = enriched_stops.loc[~enriched_stops['osmid'].isna()]
    print(f'Number of stops associated with at least 1 POI: {enriched_stops_with_poi['stop_id'].nunique()}')
    
    # For each stop, select the POI with the smallest distance.
    idx = enriched_stops_with_poi.groupby("stop_id")["distance"].idxmin()
    result = enriched_stops_with_poi.loc[idx].reset_index(drop=True)
    #display(result)
    return result
#ny_stops =generate_right_files(ny)
#paris_stops = generate_right_files(paris)
# Main execution
def main():
    # Carica traiettorie
    trajectories_files = [f for f in os.listdir(TRAJECTORIES_PATH) 
                         if f.endswith('.parquet') and not f.endswith('_posts.parquet')]
    trajectories_df = {
        f.replace('.parquet', ''): select_pois(pd.read_parquet(os.path.join(TRAJECTORIES_PATH, f))) 
        for f in trajectories_files
    }

    
    print(f"Trovate {len(trajectories_df)} traiettorie")
    
    # Processa ogni traiettoria
    for key, df in trajectories_df.items():
        try:
            
            posts_data = process_trajectory(key, df)
            save_results(key, df, posts_data)
        except Exception as e:
            print(f"Errore processing {key}: {e}")
            continue
    
    print("Completato!")

if __name__ == "__main__":
    main()