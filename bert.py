from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from itertools import pairwise
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = 'cpu'

def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two BERT embeddings.
    
    Args:
        embedding1 (torch.Tensor): The first embedding.
        embedding2 (torch.Tensor): The second embedding.
    
    Returns:
        float: The cosine similarity value.
    """
    # Normalize the embeddings
    embedding1_norm = embedding1 / embedding1.norm(dim=1, keepdim=True)
    embedding2_norm = embedding2 / embedding2.norm(dim=1, keepdim=True)
    
    # Compute cosine similarity
    cosine_sim = torch.mm(embedding1_norm, embedding2_norm.T).item()  # Single value
    return cosine_sim

def generate_bert_embeddings(input_text, model_name="bert-base-uncased", device="cpu"):
    """
    Generate BERT embeddings for a given input text.
    
    Args:
        input_text (str): The text to generate embeddings for.
        model_name (str): The name of the BERT model to use.
        device (str): The device to run the model on ("cpu" or "cuda").
    
    Returns:
        torch.Tensor: The BERT embeddings for the input text.
    """
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding='max_length', add_special_tokens=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

def score(string1: str, string2: str):
    emb1 = generate_bert_embeddings(string1, device=DEVICE)
    emb2 = generate_bert_embeddings(string2, device=DEVICE)

    return calculate_cosine_similarity(emb1, emb2)

if __name__ == "__main__":
    path = '/home/pager/Documents/consistant-contracts/results'
    files = list(os.listdir(path))
    files = sorted(files, key= lambda sub: sub[::-1])

    experiment_dict = defaultdict(lambda: [])
    for first_file, second_file in zip(files[::2], files[1::2]):
        exp_label = "_".join(first_file.split("_")[:3])
        
        experiment_dict[exp_label].append(
            pd.concat([pd.read_csv(os.path.join(path, first_file)), 
                       pd.read_csv(os.path.join(path, second_file))], 
                      ignore_index=True))
    
    for k in experiment_dict:
        experiment_score = 0
        
        # The experiment failed in one of the runs needing a stricter comparion please manually change it if you want to recreate our results.
        if k in "experiment_temperature_english8": 
            continue
        
        

        #print(len(experiment_dict[k][0]['answers']))
        #print(len(experiment_dict[k][2]['answers']))

        for i in range(0, len(experiment_dict[k][0])):
            
            p_score = 0
            p_score  += score(experiment_dict[k][0]['answers'][i], experiment_dict[k][1]['answers'][i])
            p_score += score(experiment_dict[k][1]['answers'][i], experiment_dict[k][2]['answers'][i])
            p_score += score(experiment_dict[k][0]['answers'][i], experiment_dict[k][2]['answers'][i])
            p_score /= 3
            experiment_score += p_score
            #print(f"{i}/{len(experiment_dict[k][0])} : {p_score}")

        experiment_score /= len(experiment_dict[k][0])
        print(k, ':', experiment_score)
        
