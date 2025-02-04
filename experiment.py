# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Optional

import pandas as pd

import fire, os, openpyxl, sys, torch, random

from llama_models.llama3.api.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama


CoG_paraphrase_prompt = """\
    Today I want you to learn the ways of paraphrasing a sentence. Below are few methods with examples. Go through them carefully.
    
    1. Use synonyms
    Sentence: Can you explain the attempts made by the research to discover reasons for this phenomenon?
    Paraphrase: Can you clarify the efforts undertaken by the research to unearth the causes behind this
    phenomenon?
    
    2. Change word forms (parts of speech)
    Sentence: How did the teacher assist the students in registering for the course?
    Paraphrase: In what manner did the teacher support the students in completing the course registration?
    
    3. Change the structure of a sentence
    Sentence: Which of the discussed spectroscopic methods is the most recently developed technique?
    Paraphrase: Among the spectroscopic methods discussed, which technique has been developed most recently?
    
    4. Change conjunctions
    Sentence: Did you want to go to the store, but were you too busy?
    Paraphrase: Although you were busy, did you still want to go to the store?
    Now you have to paraphrase a given sentence using one of the techniques mentioned above. I will provide you
    the number of the technique to use.
    
    Technique Number: {}
    Sentence: {}
    Paraphrase (Do not give any preamble, just give the paraphrased text)
"""

CoG_rank_prompt = """\
Question: {}

For the question above there are several options given, choose one among them which seems to be the most correct. 
Do not give any other output. Only the number.


"""

CoG_answer_options_str = "Option {}: {}\n"

def generate_answer(generator, prompt, max_gen_len, temperature, top_p):
    torch.cuda.empty_cache() # clear cache of GPU. Having issues with with GPU OOM 
                
    result = generator.chat_completion(prompt,
                                       max_gen_len=max_gen_len,
                                       temperature=temperature,
                                       top_p=top_p,
                                      )
    
    return result

def create_dialogue(language: str, context_doc: str, test_doc: str):
    system_add_agent_message = "Answer as if you are a compliance officer."
    system_add_context_message = "Add the following document to your context and use it to answer questions."
    system_add_test_message = "Add the following document to your context and only answer questions about the content of this document."
    system_restrict_message = "format your answers in a short paragraph of max 512 words starting with a difinitive statement about the compliance"

    dialogue = [
            RawMessage(role="system", content=system_add_agent_message),

            # RawMessage(role="system", content=system_add_context_message),
            # RawMessage(role="document", content=context_doc),

            RawMessage(role="system", content=system_add_test_message),
            RawMessage(role="document", content=test_doc),

            RawMessage(role="system", content=system_restrict_message),
    ]


    # add language experiment details
    if language != "english":
        system_add_language_message = f"write everything in {language}"
        dialogue.append(RawMessage(role="system", content=system_add_language_message))

    return dialogue


def read_xlsx_to_lists(file_path):
    questions = []
    answers = []

    try:
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active

        for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, max_col=2, values_only=True):
            if row[0] is not None and row[1] is not None:  # Ensure there are at least two columns with data
                questions.append(row[0])
                answers.append(row[1])

        return questions, answers

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def str_to_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default

def run_main(
    label: str,
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 15000,
    max_batch_size: int = 4,
    seed = -1,
    results_dir = "./results",
    language = "english",
    cog = False,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None
):
    # generate a random seed    
    if seed >= -1:
        seed=random.randint(1, 10000)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
        seed=seed
    )   

    context_documents = ["dora-chapter-V-art-28.txt", "dora-chapter-V-art-29.txt"] 
    question_documents = ["questions-art-28.xlsx", "questions-art-29.xlsx"] 
    test_documents = ["bench-Atlas-CA.txt"] # ,"bench-Micro-SA.txt","bench-SG-MSA.txt","./bench-TSP.txt","./bench-TTSP.txt"]
    
    for i, test in enumerate(test_documents):
        for j, quest in enumerate(question_documents):

            test_doc = ''
            context_doc = ''
            questions, _ = read_xlsx_to_lists(quest)

            with open(context_documents[j], "r") as file:
                context_doc = file.read()

            with open(test, "r") as file:
                test_doc = file.read()

            if cog:
                answers = []
                for n, q in enumerate(questions):

                    rank_prompt = CoG_rank_prompt.format(q)
                    para_answers = []
                    for k in (0, 1, 2, 3):
                        para_prompt = CoG_paraphrase_prompt.format(k, q)
                        d = create_dialogue(language, context_doc, test_doc)
                        d.append(RawMessage(role="user", content=para_prompt))
                        
                        p_result = generate_answer(generator, d, max_gen_len, temperature, top_p)
                        rank_prompt += CoG_answer_options_str.format(k, p_result.generation.content)
                        para_answers.append(p_result.generation.content)
                        print(f"{k} : \n", p_result.generation.content)

                    # select
                    r_d = create_dialogue(language, context_doc, '')
                    r_d.append(RawMessage(role="user", content=rank_prompt))
                    
                    r_result = generate_answer(generator, r_d, max_gen_len, temperature, top_p)
                    
                    choice_i = max(0, min(str_to_int(r_result.generation.content), len(para_answers)-1)) #clamp
                    answers.append(para_answers[choice_i])
                    
                    print(f"chosen {choice_i}", para_answers[choice_i])
                    print(f"\n================{n+1}/{len(questions)}==================\n")


            else:
                # generate experiments
                experiments = []
                for q in questions:
                    # create base dialogue
                    exp = create_dialogue(language, context_doc, test_doc)                 

                    # add the question
                    exp.append(RawMessage(role="user", content=q))
                    experiments.append(exp) 
            
                # run experiments
                answers = []
                for n, e in enumerate(experiments):
                    result = generate_answer(generator, e, max_gen_len, temperature, top_p)
                    answers.append(result.generation.content)
    
                    out_message = result.generation

                    print(f"> {out_message.role.capitalize()}: {out_message.content}")
                    print(f"\n================{n+1}/{len(experiments)}==================\n")

            df = pd.DataFrame()
            df["question"] = questions 
            df["answers"] = answers

            df.to_csv(f'{results_dir}/experiment_{label}_{language}_{i}_{j}_{seed}_{temperature}.csv', index=False)

def main():
    

    fire.Fire(run_main)


if __name__ == "__main__":
    main()
