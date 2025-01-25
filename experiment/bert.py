from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

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

def generate_bert_embeddings(input_text, model_name="bert-base-uncased", device="cuda"):
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
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, add_special_tokens=True, max_length=512, padding="max_length").to(device)
    
    # Pass the input through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
    
    # Optionally, aggregate embeddings (e.g., mean pooling)
    sentence_embedding = embeddings.mean(dim=1)  # Shape: (batch_size, hidden_size)
    
    return sentence_embedding

if __name__ == "__main__":


    input_text1 = """**Compliance Verification: Business Continuity and Disaster Recovery Plans**

A comprehensive Business Continuity and Disaster Recovery (BCDR) plan is essential for ensuring organizational resilience and minimizing business disruptions. Our review reveals that the current plan components, alignment with operational needs, update frequency, testing requirements, and documentation standards are insufficient, indicating a significant resilience gap.

**Plan Components:**

* The plan lacks a clear definition of business continuity and disaster recovery objectives, risk assessment, and impact analysis.
* It does not specify roles and responsibilities, communication protocols, and incident management procedures.
* The plan is missing critical components, such as data backup and recovery processes, incident response plans, and business continuity procedures.

**Alignment with Operational Needs:**

* The plan does not take into account the organization's operational needs, such as critical business processes, infrastructure, and technology dependencies.
* It does not consider the impact of disasters on key stakeholders, including employees, customers, and suppliers.

**Update Frequency:**

* The plan has not been updated in the past year, which is inadequate given the dynamic nature of business operations and changing risk landscape.
* The update frequency should be at least annually, with regular reviews and updates to ensure the plan remains relevant and effective.

**Testing Requirements:**

* The plan lacks a comprehensive testing schedule, which is critical for identifying weaknesses and ensuring the plan's effectiveness.
* Regular testing should be conducted at least twice a year, with a thorough review of results and corrective actions taken as needed.

**Documentation Standards:**

* The plan does not adhere to industry-recognized documentation standards, such as ISO 22301 or NIST SP 800-34.
* The plan should be documented in a clear, concise, and easily accessible format, with regular reviews and updates to ensure compliance with relevant standards.

**Resilience Gap:**

Given the deficiencies identified in the review, it is essential to flag this as a significant resilience gap and suggest adding comprehensive BCDR requirements to the plan. This includes:

1. Conducting a thorough risk assessment and impact analysis to identify critical business processes and infrastructure dependencies.
2. Developing a comprehensive incident response plan, including communication protocols and roles and responsibilities.
3. Establishing regular testing and update schedules to ensure the plan remains relevant and effective.
4. Adhering to industry-recognized documentation standards, such as ISO 22301 or NIST SP 800-34.
5. Ensuring the plan is regularly reviewed and updated to reflect changes in business operations and the risk landscape.

By addressing these gaps, the organization can develop a robust BCDR plan that ensures business continuity and minimizes disruptions in the event of a disaster."""

    input_text2 = """**Compliance Review Findings: Business Continuity and Disaster Recovery (BCDR) Plans**

The BCDR plans reviewed are inadequate in several areas, posing a significant resilience gap. To ensure compliance with industry standards and regulatory requirements, the following components and requirements need attention:

**Plan Components:**

* The current plans lack a clear statement of objectives, scope, and assumptions.
* Risk assessments and impact analyses are incomplete or missing.
* Recovery Time Objectives (RTOs) and Recovery Point Objectives (RPOs) are not defined for critical business functions.
* The plans do not include a comprehensive inventory of IT assets, systems, and data.

**Alignment with Operational Needs:**

* The plans do not adequately address the needs of key stakeholders, including customers, employees, and suppliers.
* Business continuity procedures are not aligned with operational processes and procedures.
* The plans do not account for changes in business operations, such as remote work arrangements.

**Update Frequency:**

* The plans have not been updated within the last 12 months, which is a significant gap.
* The review process does not ensure that the plans are reviewed and updated regularly.

**Testing Requirements:**

* The plans do not include a comprehensive testing strategy, including regular drills and exercises.
* The testing process does not evaluate the effectiveness of the BCDR plans.

**Documentation Standards:**

* The plans are not well-organized, and documentation is not consistent.
* The plans do not meet industry-standard documentation requirements.

**Resilience Gap:**

Based on the review findings, I recommend adding comprehensive BCDR requirements to the plans, including:

1. Conduct a thorough risk assessment and impact analysis to identify critical business functions and systems.
2. Develop a clear statement of objectives, scope, and assumptions.
3. Define RTOs and RPOs for critical business functions.
4. Create a comprehensive inventory of IT assets, systems, and data.
5. Develop business continuity procedures aligned with operational processes and procedures.
6. Establish a regular review and update process for the plans.
7. Develop a comprehensive testing strategy, including regular drills and exercises.
8. Ensure documentation meets industry-standard requirements.

By addressing these gaps, the organization can ensure that its BCDR plans are effective, comprehensive, and aligned with operational needs, ultimately reducing the risk of business disruption and ensuring continuity in the event of a disaster."""
    input_text3 = """Compliance Observation: The contract does not explicitly specify a process for evaluating changes in the subcontracting chain.

Upon reviewing the contract, I found that Section 20.11 (Subcontractors and Affiliates) mentions that Atlassian may use subcontractors or its Affiliates in the performance of its obligations under this Agreement, but it does not provide any specific requirements for notification or risk evaluation of changes in the subcontracting structure.

The contract does not specify any notification requirements for changes in the subcontracting chain, nor does it require a re-assessment of risks for new subcontractors or structural changes.

Recommendation:

To strengthen the compliance of the contract, I suggest including clauses that require timely notification and risk evaluation of any changes in the subcontracting structure. This can be achieved by adding provisions such as:

* Atlassian shall notify Customer in writing of any changes to its subcontracting structure, including the addition or removal of subcontractors, at least [X] days prior to the effective date of the change.
* Customer reserves the right to object to any changes in the subcontracting structure that may increase the risk to Customer's business or compromise the confidentiality or security of Customer Data.
* Atlassian shall conduct a risk assessment of any new subcontractor or structural change and provide the results to Customer upon request.

By including these provisions, the contract can better ensure that Atlassian's subcontracting changes do not compromise the relationship or the interests of Customer."
"Search for clauses specifying restrictions or conditions for subcontracting. Extract details about limitations on services, geography, or critical functions. If missing, suggest adding these restrictions.","**Compliance Note:** This Atlassian Customer Agreement contains inadequate subcontracting restrictions, exposing Atlassian to potential compliance risks and liability.

Upon reviewing the agreement, I found that clause 20.11 (Subcontractors and Affiliates) only mentions that Atlassian may use subcontractors or its Affiliates in the performance of its obligations, but it does not specify any restrictions or conditions on subcontracting. This lack of specificity may lead to unforeseen consequences, such as:

* Unclear limitations on services: The agreement does not specify which services Atlassian can subcontract or which services must be performed by Atlassian directly.
* Unrestricted geography: The agreement does not limit the geography where subcontractors can be based or where services can be performed, potentially exposing Atlassian to data protection and other regulatory risks.
* Unclear critical functions: The agreement does not specify which critical functions or services cannot be subcontracted, potentially undermining the reliability and security of the Products and related Support and Advisory Services.

To mitigate these risks, I recommend adding the following subcontracting restrictions to the agreement:

* Specify which services Atlassian can subcontract and which services must be performed by Atlassian directly.
* Limit subcontracting to specific geographic regions or countries, ensuring compliance with data protection and other regulatory requirements.
* Identify critical functions or services that cannot be subcontracted, ensuring that Atlassian maintains control over essential aspects of the Products and related Support and Advisory Services.

Example of revised clause:

20.11. Subcontractors and Affiliates.

(a) Atlassian may use subcontractors or its Affiliates in the performance of its obligations under this Agreement, but only for the following services: [list specific services, e.g., data processing, customer support, software development].
(b) Subcontractors must be based in countries that are members of the OECD or have equivalent data protection standards, and services must be performed within the European Economic Area (EEA) or the United States.
(c) The following critical functions or services cannot be subcontracted: [list specific functions or services, e.g., product development, security, data protection]."""


    baseline = [
        pd.read_csv("./experiment/baseline/experiment_0_0_4134.csv")["answers"].tolist() + pd.read_csv("./experiment/baseline/experiment_0_1_4134.csv")["answers"].tolist(),
        pd.read_csv("./experiment/baseline/experiment_0_0_5008.csv")["answers"].tolist() + pd.read_csv("./experiment/baseline/experiment_0_1_5008.csv")["answers"].tolist(),
        pd.read_csv("./experiment/baseline/experiment_0_0_5143.csv")["answers"].tolist() + pd.read_csv("./experiment/baseline/experiment_0_1_5143.csv")["answers"].tolist(),
    ]

    chineese = [
        pd.read_csv("./experiment/chineese/experiment_0_0_137.csv")["answers"].tolist() + pd.read_csv("./experiment/chineese/experiment_0_1_137.csv")["answers"].tolist(),
        pd.read_csv("./experiment/chineese/experiment_0_0_4496.csv")["answers"].tolist() + pd.read_csv("./experiment/chineese/experiment_0_1_4496.csv")["answers"].tolist(),
        pd.read_csv("./experiment/chineese/experiment_0_0_9040.csv")["answers"].tolist() + pd.read_csv("./experiment/chineese/experiment_0_1_9040.csv")["answers"].tolist(),
    ]

    temperature20 = [
        pd.read_csv("./experiment/temperature/experiment_0_0_1050_0.2.csv")["answers"].tolist() + pd.read_csv("./experiment/temperature/experiment_0_1_1050_0.2.csv")["answers"].tolist(),
        pd.read_csv("./experiment/temperature/experiment_0_0_6032_0.2.csv")["answers"].tolist() + pd.read_csv("./experiment/temperature/experiment_0_1_6032_0.2.csv")["answers"].tolist(),
        pd.read_csv("./experiment/temperature/experiment_0_0_6491_0.2.csv")["answers"].tolist() + pd.read_csv("./experiment/temperature/experiment_0_1_6491_0.2.csv")["answers"].tolist(),
    ]

    temperature80 = [
        pd.read_csv("./experiment/temperature/experiment_0_0_721_0.8.csv")["answers"].tolist() + pd.read_csv("./experiment/temperature/experiment_0_1_721_0.8.csv")["answers"].tolist(),
        pd.read_csv("./experiment/temperature/experiment_0_0_5699_0.8.csv")["answers"].tolist() + pd.read_csv("./experiment/temperature/experiment_0_1_5699_0.8.csv")["answers"].tolist(),
        pd.read_csv("./experiment/temperature/experiment_0_0_9200_0.8.csv")["answers"].tolist() + pd.read_csv("./experiment/temperature/experiment_0_1_9200_0.8.csv")["answers"].tolist(),
    ]

    # CoG = [
    #     pd.read_csv("./experiment/chineese/experiment_0_0_137.csv")["answers"].tolist() + pd.read_csv("./experiment/baseline/experiment_0_1_137.csv")["answers"].tolist(),
    #     pd.read_csv("./experiment/chineese/experiment_0_0_4496.csv")["answers"].tolist() + pd.read_csv("./experiment/baseline/experiment_0_1_4496.csv")["answers"].tolist(),
    #     pd.read_csv("./experiment/chineese/experiment_0_0_9040.csv")["answers"].tolist() + pd.read_csv("./experiment/baseline/experiment_0_1_9040.csv")["answers"].tolist(),
    # ]

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    experiment = temperature80
    comparison = []
    for i in range(len(experiment[0])):
        sim_1 = calculate_cosine_similarity(generate_bert_embeddings(experiment[0][i]), generate_bert_embeddings(experiment[1][i]))
        sim_2 = calculate_cosine_similarity(generate_bert_embeddings(experiment[1][i]), generate_bert_embeddings(experiment[2][i]))
        sim_3 = calculate_cosine_similarity(generate_bert_embeddings(experiment[2][i]), generate_bert_embeddings(experiment[0][i]))
        
        comparison.append((sim_1 + sim_2 + sim_3) / 3)


    # print(comparison)
    print(sum(comparison)/ len(comparison))

    # Generate BERT embeddings
    # embeddings1 = generate_bert_embeddings(input_text1)
    # embeddings2 = generate_bert_embeddings(input_text2)
    # embeddings3 = generate_bert_embeddings(input_text3)
    
    # print("BERT Embeddings Shape:", embeddings1.shape)
    # print("BERT Embeddings Shape:", embeddings2.shape)
    # print("BERT Embeddings Shape:", embeddings3.shape)

    # print(calculate_cosine_similarity(embeddings1, embeddings2))
    # print(calculate_cosine_similarity(embeddings2, embeddings3))
    # print(calculate_cosine_similarity(embeddings1, embeddings3))

