import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Dict, List, Tuple
import numpy as np

# Configurazione ambiente
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Definizione dei modelli da testare
MODELS = {
    # Modelli originali
    'cardiffnlp': "cardiffnlp/twitter-roberta-base-sentiment",
    'nlptown': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'tabularisai': "tabularisai/multilingual-sentiment-analysis",
    
    # Nuovi modelli raccomandati
    'xlm_roberta': "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    'distilbert_multi': "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    'bertweet': "finiteautomata/bertweet-base-sentiment-analysis",
    'roberta_large': "siebert/sentiment-roberta-large-english",
    
    # Modelli per emozioni (opzionale)
    # 'emotions': "cardiffnlp/twitter-roberta-base-emotion",
}

# Definizione dei dataset
DATASETS = {
    'ny': '/home/francomaria.nardini/raid/guidorocchietti/code/MATDataset/trajectories/enriched_occasional_ny_posts.parquet',
    'paris': '/home/francomaria.nardini/raid/guidorocchietti/code/MATDataset/trajectories/enriched_occasional_paris_posts.parquet'
}

class SentimentAnalyzer:
    def __init__(self, models: Dict[str, str]):
        self.models = models
        self.pipelines = {}
        self.label_mappings = {}
        self._setup_models()
    
    def _setup_models(self):
        """Inizializza tutti i modelli e le pipeline"""
        for model_key, model_name in self.models.items():
            print(f"Caricamento modello: {model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model = model.to('cuda')
                
                pipeline_obj = pipeline(
                    "sentiment-analysis", 
                    model=model, 
                    tokenizer=tokenizer, 
                    device=0
                )
                
                self.pipelines[model_key] = pipeline_obj
                self.label_mappings[model_key] = self._get_label_mapping(model_key)
                print(f"✓ Modello {model_key} caricato con successo")
                
            except Exception as e:
                print(f"✗ Errore nel caricamento del modello {model_key}: {e}")
    
    def _get_label_mapping(self, model_key: str) -> Dict[str, str]:
        """Definisce il mapping delle label per ogni modello"""
        if 'cardiffnlp' in model_key.lower() or 'xlm_roberta' in model_key.lower():
            return {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
        elif 'nlptown' in model_key.lower():
            return {
                '1 star': 'very_negative',
                '2 stars': 'negative',
                '3 stars': 'neutral',
                '4 stars': 'positive',
                '5 stars': 'very_positive'
            }
        elif 'tabularisai' in model_key.lower():
            return {
                'Negative': 'negative',
                'Neutral': 'neutral',
                'Positive': 'positive'
            }
        elif 'distilbert' in model_key.lower():
            return {
                'NEGATIVE': 'negative',
                'POSITIVE': 'positive'
            }
        elif 'bertweet' in model_key.lower() or 'roberta_large' in model_key.lower():
            return {
                'NEG': 'negative',
                'NEU': 'neutral',
                'POS': 'positive'
            }
        else:
            # Mapping generico - adattalo in base ai modelli che usi
            return {
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral',
                'POS': 'positive'
            }
    
    def analyze_texts(self, texts: List[str], model_key: str) -> List[Dict]:
        """Analizza una lista di testi con un modello specifico"""
        if model_key not in self.pipelines:
            raise ValueError(f"Modello {model_key} non disponibile")
        
        try:
            results = self.pipelines[model_key](texts)
            return results
        except Exception as e:
            print(f"Errore nell'analisi con {model_key}: {e}")
            return []

def load_datasets(dataset_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Carica tutti i dataset"""
    datasets = {}
    for name, path in dataset_paths.items():
        try:
            df = pd.read_parquet(path)
            datasets[name] = df
            print(f"✓ Dataset {name} caricato: {len(df)} righe")
        except Exception as e:
            print(f"✗ Errore nel caricamento dataset {name}: {e}")
    return datasets

def calculate_metrics(true_positive: int, true_negative: int, 
                     false_positive: int, false_negative: int) -> Dict[str, float]:
    """Calcola le metriche di valutazione"""
    total = true_positive + true_negative + false_positive + false_negative
    
    if total == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_samples': total
    }

def evaluate_model_performance(results: Dict, model_key: str, analyzer: SentimentAnalyzer) -> Dict[str, float]:
    """Valuta le performance di un modello su tutti i dataset"""
    true_positive = true_negative = false_positive = false_negative = 0
    
    for dataset_name, dataset_results in results.items():
        for sentiment_type, predictions in dataset_results.items():
            for label, count in predictions.items():
                # Mappa la label al sentiment normalizzato
                normalized_label = analyzer.label_mappings[model_key].get(label, 'unknown')
                
                if sentiment_type == 'positive':
                    # Per testi positivi, le predizioni positive sono true positive
                    if normalized_label in ['positive', 'very_positive']:
                        true_positive += count
                    elif normalized_label in ['negative', 'very_negative']:
                        false_negative += count
                    # neutral viene considerato come falso positivo per semplicità
                    else:
                        false_positive += count
                        
                elif sentiment_type == 'negative':
                    # Per testi negativi, le predizioni negative sono true negative
                    if normalized_label in ['negative', 'very_negative']:
                        true_negative += count
                    elif normalized_label in ['positive', 'very_positive']:
                        false_positive += count
                    # neutral viene considerato come falso negativo per semplicità
                    else:
                        false_negative += count
    
    return calculate_metrics(true_positive, true_negative, false_positive, false_negative)

def main():
    print("Inizializzazione Sentiment Analysis Multi-Model...")
    
    # Inizializza l'analyzer
    analyzer = SentimentAnalyzer(MODELS)
    
    # Carica i dataset
    datasets = load_datasets(DATASETS)
    
    # Risultati per tutti i modelli
    all_results = {}
    
    # Analizza con ogni modello
    for model_key in analyzer.pipelines.keys():
        print(f"\n{'='*50}")
        print(f"Analisi con modello: {model_key}")
        print(f"{'='*50}")
        
        model_results = {}
        
        for dataset_name, df in datasets.items():
            print(f"\nAnalizzando dataset: {dataset_name}")
            
            # Estrai i testi positivi e negativi
            positive_texts = df['positive'].dropna().tolist()
            negative_texts = df['negative'].dropna().tolist()
            
            print(f"Testi positivi: {len(positive_texts)}")
            print(f"Testi negativi: {len(negative_texts)}")
            
            # Analizza i testi
            positive_results = analyzer.analyze_texts(positive_texts, model_key)
            negative_results = analyzer.analyze_texts(negative_texts, model_key)
            
            if positive_results and negative_results:
                # Conta i risultati
                positive_counts = pd.Series([r['label'] for r in positive_results]).value_counts().to_dict()
                negative_counts = pd.Series([r['label'] for r in negative_results]).value_counts().to_dict()
                
                model_results[dataset_name] = {
                    'positive': positive_counts,
                    'negative': negative_counts
                }
                
                # Stampa i risultati per questo dataset
                print(f"\nRisultati {dataset_name} - Testi Positivi:")
                for label, count in positive_counts.items():
                    normalized = analyzer.label_mappings[model_key].get(label, label)
                    print(f"  {label} ({normalized}): {count}")
                
                print(f"\nRisultati {dataset_name} - Testi Negativi:")
                for label, count in negative_counts.items():
                    normalized = analyzer.label_mappings[model_key].get(label, label)
                    print(f"  {label} ({normalized}): {count}")
        
        all_results[model_key] = model_results
        
        # Calcola le metriche per questo modello
        if model_results:
            metrics = evaluate_model_performance(model_results, model_key, analyzer)
            print(f"\n📊 METRICHE MODELLO {model_key.upper()}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"Campioni totali: {metrics['total_samples']}")
    
    # Confronto finale
    print(f"\n{'='*60}")
    print("CONFRONTO FINALE MODELLI")
    print(f"{'='*60}")
    
    comparison_results = []
    for model_key in all_results.keys():
        if all_results[model_key]:
            metrics = evaluate_model_performance(all_results[model_key], model_key, analyzer)
            comparison_results.append({
                'Model': model_key,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Total Samples': metrics['total_samples']
            })
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        print(f"\n🏆 Migliore modello per F1-Score: {comparison_df.iloc[0]['Model']}")
        print(f"🏆 Migliore modello per Accuracy: {comparison_df.sort_values('Accuracy', ascending=False).iloc[0]['Model']}")

if __name__ == "__main__":
    main()