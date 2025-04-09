import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from typing import List, Dict
import nltk
nltk.download('punkt')

class ModelBenchmark:
    def __init__(self):
        self.gemma = OllamaLLM(model="gemma3:4b")
        self.llama = OllamaLLM(model="llama3.2")
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Test cases with ground truth answers
        self.test_cases = [
            {
                "question": "What is the capital of France?",
                "ground_truth": "The capital of France is Paris."
            },
            {
                "question": "Explain how photosynthesis works briefly.",
                "ground_truth": "Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
            },
            {
                "question": "What is 2+2?",
                "ground_truth": "2+2 equals 4."
            },
            # Add more test cases as needed
        ]

    def evaluate_response(self, response: str, ground_truth: str) -> Dict[str, float]:
        # Calculate ROUGE scores
        rouge_scores = self.scorer.score(ground_truth, response)
        
        # Calculate BLEU score
        reference = [ground_truth.split()]
        candidate = response.split()
        bleu_score = sentence_bleu(reference, candidate)
        
        # Response time (will be set later)
        metrics = {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score,
        }
        
        return metrics

    def run_benchmark(self):
        results = {
            'gemma': [],
            'llama': []
        }
        
        for test_case in tqdm(self.test_cases, desc="Running benchmarks"):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            
            # Test Gemma
            start_time = time.time()
            gemma_response = self.gemma.invoke(question)
            gemma_time = time.time() - start_time
            gemma_metrics = self.evaluate_response(gemma_response, ground_truth)
            gemma_metrics['response_time'] = gemma_time
            results['gemma'].append(gemma_metrics)
            
            # Test Llama
            start_time = time.time()
            llama_response = self.llama.invoke(question)
            llama_time = time.time() - start_time
            llama_metrics = self.evaluate_response(llama_response, ground_truth)
            llama_metrics['response_time'] = llama_time
            results['llama'].append(llama_metrics)
            
        return results

    def plot_results(self, results):
        # Convert results to DataFrames
        gemma_df = pd.DataFrame(results['gemma'])
        llama_df = pd.DataFrame(results['llama'])
        
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'response_time']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison: Gemma vs Llama', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = {
                'Gemma': gemma_df[metric],
                'Llama': llama_df[metric]
            }
            
            sns.boxplot(data=data, ax=ax)
            ax.set_title(f'{metric.upper()} Score')
            ax.set_ylabel('Score')
            
        # Remove extra subplot
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Print average metrics
        print("\nAverage Metrics:")
        print("\nGemma:")
        for metric in metrics:
            print(f"{metric}: {gemma_df[metric].mean():.4f}")
        
        print("\nLlama:")
        for metric in metrics:
            print(f"{metric}: {llama_df[metric].mean():.4f}")

def main():
    benchmark = ModelBenchmark()
    print("Starting benchmark...")
    results = benchmark.run_benchmark()
    benchmark.plot_results(results)
    print("\nBenchmark complete! Results saved to 'model_comparison.png'")

if __name__ == "__main__":
    main() 