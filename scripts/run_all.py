from scripts.run_gpt2 import run_gpt2
from scripts.run_gptneo import run_gptneo
from scripts.run_openai_gpt import run_openai
from scripts.run_bert_variants import run_bert_models
from scripts.run_llama import run_llama
from scripts.run_openai_prescription import run_openai_prescription

def main():
    print("Running all model benchmarks...\n")

    print("==> Running GPT-2")
    run_gpt2()
    print("==> Completed GPT-2\n")

    print("==> Running GPT-Neo")
    run_gptneo()
    print("==> Completed GPT-Neo\n")

    print("==> Running OpenAI GPT")
    run_openai()
    print("==> Completed OpenAI GPT\n")

    print("==> Running BERT Variants")
    run_bert_models()
    print("==> Completed BERT Variants\n")

    print("==> Running LLaMA")
    run_llama()
    print("==> Completed LLaMA\n")

    print("==> Running OpenAI on Prescription Data")
    run_openai_prescription()
    print("==> Completed OpenAI on Prescription Data\n")
