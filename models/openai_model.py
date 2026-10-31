# models/openai_model.py

import numpy as np
import time
from openai import OpenAI

class OpenAIModel:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, input_text, temperature=1.0, max_tokens=50):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    def bayesian_estimate(self, input_text, n_samples=10):
        lengths = []
        for _ in range(n_samples):
            response = self.generate(input_text, temperature=np.random.uniform(0.7, 1.2))
            lengths.append(len(response.split()))
            time.sleep(1)
        mean_pred = np.mean(lengths)
        uncertainty = np.std(lengths) / (mean_pred + 1e-6)
        return mean_pred, uncertainty
