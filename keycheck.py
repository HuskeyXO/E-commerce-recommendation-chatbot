import openai

# Set your API key
openai.api_key = "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"

try:
    # Use the updated endpoint for listing models
    response = openai.models.list()
    print("API Key is valid. Models:", response)
except openai.InvalidRequestError:
    print("Invalid API Key or request.")
except openai.OpenAIError as e:
    print(f"OpenAI Error: {e}")
except Exception as e:
    print(f"General Error: {e}")
