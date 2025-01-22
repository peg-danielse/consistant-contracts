# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Optional

import fire, os

from llama_models.llama3.api.datatypes import RawMessage, StopReason

from llama_models.llama3.reference_impl.generation import Llama


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
    print("the current directory is: ",os.getcwd())

    context_documents = ["dora-chapter-V-art-28.txt", "dora-chapter-V-art-30.txt", "dora-chapter-V-art-30.txt"] 
    test_documents = ["bench-Atlas-CA.txt","bench-Micro-SA.txt","bench-SG-MSA.txt","bench-TSP.txt","bench-TTSP.txt"]
    
    for i, fn in enumerate(context_documents):
        with open(fn, "r") as file:
            context_documents[i] = file.read()

    for i, fn in enumerate(test_documents):
        with open(fn, "r") as file:
            test_documents[i] = file.read()

    system_add_context_message = "add the following document to your context."
    system_add_test_message = "add the following document to your context and only answer questions about this document."

    dialog = [[
        RawMessage(role="System", content=system_add_context_message),
        RawMessage(role="Document", content=context_documents[0]),
        RawMessage(role="System", content=system_add_test_message),
        RawMessage(role="Document", content=text_documents[0]),

RawMessage(role="User", content="""\
Search for explicit clauses stating that the financial entity retains full responsibility for DORA compliance. Look specifically for:
 • Clear statements of retained responsibility
 • References to DORA obligations
 • Non-transfer of compliance obligations
 • Acknowledgment that outsourcing doesn't diminish entity's responsibility
 
Follow-up if Missing: Flag as critical compliance gap and recommend adding explicit responsibility retention clause citing DORA requirements."""),

RawMessage(role="User", content="""\
    Review contract for clear statements regarding the financial entity's retained responsibility for all applicable financial services laws. Look for:
 • Comprehensive coverage of all applicable regulations
 • Non-delegation of regulatory responsibilities
 • Clear acknowledgment of ongoing regulatory obligations
 • References to specific financial services laws
 ▪ 
Follow-up if Missing: Recommend adding explicit clause covering retention of regulatory responsibilities for all applicable financial services laws.
"""),

RawMessage(role="User", content="""\ 
Review contract for clear definition of roles and responsibilities in third-party risk management. Verify:
 • Specific role assignments
 • Responsibility matrices
 • Internal accountability structures
 • Reporting lines

Follow-up if Missing: Recommend adding detailed RACI matrix or equivalent responsibility definition structure."""),

RawMessage(role="User", content="""\ Search for provisions requiring senior management or board approval of ICT service agreements. Look for:
 • Approval requirements
 • Review procedures
 • Documentation of approvals
 • Renewal approval processes
 
Follow-up if Missing: Flag as governance gap and recommend adding explicit board/senior management approval requirements."""),

RawMessage(role="User", content="""\ 
Review for provisions requiring ongoing board/senior management monitoring of third-party arrangements. Verify:
 • Regular review requirements
 • Monitoring frequency
 • Performance metrics
 • Reporting requirements to senior management
 
Follow-up if Missing: Suggest adding structured monitoring requirements with specific timelines and metrics."""),

RawMessage(role="User", content="""\ 
Search for documented escalation paths for unresolved third-party risks. Look for:
 • Clear escalation triggers
 • Escalation levels
 • Time frames for escalation
 • Resolution requirements
 • Board notification procedures

Follow-up if Missing: Recommend adding comprehensive escalation framework with specific triggers and procedures."""),

RawMessage(role="User", content="""\ 
Review requirements for documenting escalation processes and outcomes. Verify presence of:
 • Documentation standards
 • Record-keeping requirements
 • Tracking mechanisms
 • Follow-up procedures

Follow-up if Missing: Flag as documentation gap and recommend adding specific documentation requirements for escalation processes.
"""),

RawMessage(role="User", content="""\ Review requirements for maintaining comprehensive governance documentation. Look for:
 • Documentation maintenance requirements
 • Update procedures
 • Accessibility requirements
 • Retention policies
 ▪ 
Follow-up if Missing: Recommend establishing comprehensive documentation framework for governance processes."""),
    ],
]
    
    for dialog in dialogs:
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
