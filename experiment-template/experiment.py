# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Optional

import fire, os, openpyxl, sys

from llama_models.llama3.api.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama


def generateParaphrasePrompt(method, sentence):
    prompt = f""":
    Today I want you to learn the ways of paraphrasing a sentence. Below are few methods with examples. Go through
    them carefully.
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
    Technique Number: {method}
    Sentence: {sentence}
    Paraphrase (Do not give any preamble, just give the paraphrased text):"""

    return prompt


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


# def generate_reply(generator, context, prompt, max_gen_len, temperature, top_p):

#     if context:
#         model_input = [RawMessage(role="system", content=context), RawMessage(role="user", content=prompt)]
#     else:
#         model_input = [RawMessage(role="user", content=prompt)]

#     result = generator.chat_completion(model_input, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    
#     #print("\n\n=== MODEL PROMPT ===")
#     #for msg in model_input:
#     #    print(f"{msg.role.capitalize()}: {msg.content}\n")

#     out_message = result.generation
#     #print("\n\n=== MODEL OUTPUT ===")
#     #print(f"> {out_message.role.capitalize()}: {out_message.content}")
#     #print("\n==================================\n")

#     return out_message.content
    

def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
        seed=42
    )

    # base. 

    # 1. Load in the benchmark documents with formatting.

    # 2. Load the prompts.

    # 3. Do the experiments.
    print("the current directory is: ", os.getcwd())

    context_documents = ["dora-chapter-V-art-28.txt", "dora-chapter-V-art-30.txt", "dora-chapter-V-art-30.txt"] 
    test_documents = ["bench-Atlas-CA.txt","bench-Micro-SA.txt","bench-SG-MSA.txt","bench-TSP.txt","bench-TTSP.txt"]
    
    for i, fn in enumerate(context_documents):
        with open(fn, "r") as file:
            context_documents[i] = file.read()

    for i, fn in enumerate(test_documents):
        with open(fn, "r") as file:
            test_documents[i] = file.read()

    questions_28, answers_28 = read_xlsx_to_lists("article_28_question_answer_pairs.xlsx")
    print(questions_28)

    system_add_context_message = "Answer as if you are a compliance officer."
    system_add_context_message = "add the following document to your context."
    system_add_test_message = "add the following document to your context and only answer questions about this document."    

    experiments = []

    for prompt in questions:
        exp = [
            RawMessage(role="System", content=system_add_context_message),
            RawMessage(role="Document", content=context_documents[0]),
            RawMessage(role="System", content=system_add_test_message),
            RawMessage(role="Document", content=test_documents[0]),
            RawMessage(role="User", content=prompt),
        ]

        experiments.append(exp)
    
    for dialog in experiments:
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        out_message = result.generation
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
