from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    return tokenizer, model

def process_data(input_text, tokenizer, model):
    inputs = tokenizer('summarize: ' + input_text, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary

# function that takes a string and returns a summary of it
def summarize(input_text):
    # load the model and tokenizer
    tokenizer, model = load_model()

    # process the data
    summary = process_data(input_text, tokenizer, model)

    return summary

# Load the model and tokenizer
tokenizer, model = load_model()

# Process the data
summary = process_data(input_text, tokenizer, model)

print(summary)