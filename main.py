import argparse
import sys
import os

# Ensure subdirectories are on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

def run_script(script_name):
    if script_name == "gpt2":
        from scripts.run_gpt2 import run_gpt2
        run_gpt2()
    elif script_name == "gptneo":
        from scripts.run_gptneo import run_gptneo
        run_gptneo()
    elif script_name == "gptneo_pubmedqa":
        from scripts.run_gptneo_pubmedqa import run_gptneo_pubmedqa
        run_gptneo_pubmedqa()
    elif script_name == "openai":
        from scripts.run_openai_gpt import run_openai
        run_openai()
    elif script_name == "openai_prescription":
        from scripts.run_openai_prescription import run_openai_prescription
        run_openai_prescription()
    elif script_name == "openai_pubmedqa":
        from scripts.run_openai_pubmedqa import run_openai_pubmedqa
        run_openai_pubmedqa()
    elif script_name == "bert":
        from scripts.run_bert_variants import run_bert_models
        run_bert_models()
    elif script_name == "bert_prescription":
        from scripts.run_bert_prescription import run_bert_prescription
        run_bert_prescription()
    elif script_name == "llama":
        from scripts.run_llama import run_llama
        run_llama()
    elif script_name == "llama_prescription":
        from scripts.run_llama_prescription import run_llama_prescription
        run_llama_prescription()
    elif script_name == "gpt2_pubmedqa":
        from scripts.run_gpt2_pubmedqa import run_gpt2_pubmedqa
        run_gpt2_pubmedqa()
    elif script_name == "bert_pubmedqa":
        from scripts.run_bert_pubmedqa import run_bert_pubmedqa
        run_bert_pubmedqa()
    elif script_name == "llama_pubmedqa":
        from scripts.run_llama_pubmedqa import run_llama_pubmedqa
        run_llama_pubmedqa()
    elif script_name == "openai_temperature":
        from scripts.run_openai_temperature import run_openai_temperature
        run_openai_temperature()
    elif args.script == "llama_temperature":
        from scripts.run_llama_temperature import run_llama_temperature
        run_llama_temperature()

    elif script_name == "all":
        from scripts.run_all import main as run_all
        run_all()
    else:
        print(f"Unknown script: {script_name}. Please use one of: "
              "gpt2, gptneo, gptneo_pubmedqa, openai, openai_prescription, openai_pubmedqa, "
              "bert, bert_prescription, llama, llama_prescription, gpt2_pubmedqa, all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run uncertainty benchmarking scripts")
    parser.add_argument(
        "--script",
        type=str,
        default="all",
        help="Choose from: gpt2, gptneo, gptneo_pubmedqa, openai, openai_prescription, openai_pubmedqa, "
             "bert, bert_prescription, llama, llama_prescription, gpt2_pubmedqa, all"
    )
    args = parser.parse_args()
    run_script(args.script)
