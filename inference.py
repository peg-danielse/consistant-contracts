from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedModel

# Path to the converted Hugging Face model
model_path = "meta-llama/Llama-3.3-70B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

type(model)

# Generate text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids, max_length=50, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
