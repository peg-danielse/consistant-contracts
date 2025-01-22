# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Optional

import fire

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

    dora-chapter-V-document = """\
1. Financial entities shall manage ICT third-party risk as an integral component of ICT risk within their ICT risk  management framework as referred to in Article 6(1), and in accordance with the following principles: 
(a) financial entities that have in place contractual arrangements for the use of ICT services to run their business operations  shall, at all times, remain fully responsible for compliance with, and the discharge of, all obligations under this  Regulation and applicable financial services law; 
(b) financial entities’ management of ICT third-party risk shall be implemented in light of the principle of proportionality,  taking into account: 
(i) the nature, scale, complexity and importance of ICT-related dependencies, 
(ii) the risks arising from contractual arrangements on the use of ICT services concluded with ICT third-party service  providers, taking into account the criticality or importance of the respective service, process or function, and the  potential impact on the continuity and availability of financial services and activities, at individual and at group  level. 

2. As part of their ICT risk management framework, financial entities, other than entities referred to in Article 16(1),  first subparagraph, and other than microenterprises, shall adopt, and regularly review, a strategy on ICT third-party risk,  taking into account the multi-vendor strategy referred to in Article 6(9), where applicable. The strategy on ICT third-party  risk shall include a policy on the use of ICT services supporting critical or important functions provided by ICT third-party  service providers and shall apply on an individual basis and, where relevant, on a sub-consolidated and consolidated basis.  The management body shall, on the basis of an assessment of the overall risk profile of the financial entity and the scale  and complexity of the business services, regularly review the risks identified in respect to contractual arrangements on the  use of ICT services supporting critical or important functions. 

3. As part of their ICT risk management framework, financial entities shall maintain and update at entity level, and at  sub-consolidated and consolidated levels, a register of information in relation to all contractual arrangements on the use of  ICT services provided by ICT third-party service providers. 
The contractual arrangements referred to in the first subparagraph shall be appropriately documented, distinguishing  between those that cover ICT services supporting critical or important functions and those that do not. 
Financial entities shall report at least yearly to the competent authorities on the number of new arrangements on the use of  ICT services, the categories of ICT third-party service providers, the type of contractual arrangements and the ICT services  and functions which are being provided. 
Financial entities shall make available to the competent authority, upon its request, the full register of information or, as  requested, specified sections thereof, along with any information deemed necessary to enable the effective supervision of  the financial entity. 
Financial entities shall inform the competent authority in a timely manner about any planned contractual arrangement on  the use of ICT services supporting critical or important functions as well as when a function has become critical or  important. 

4. Before entering into a contractual arrangement on the use of ICT services, financial entities shall: 
(a) assess whether the contractual arrangement covers the use of ICT services supporting a critical or important function; (b) assess if supervisory conditions for contracting are met; 
(c) identify and assess all relevant risks in relation to the contractual arrangement, including the possibility that such  contractual arrangement may contribute to reinforcing ICT concentration risk as referred to in Article 29; 
(d) undertake all due diligence on prospective ICT third-party service providers and ensure throughout the selection and  assessment processes that the ICT third-party service provider is suitable; 
(e) identify and assess conflicts of interest that the contractual arrangement may cause. 

5. Financial entities may only enter into contractual arrangements with ICT third-party service providers that comply  with appropriate information security standards. When those contractual arrangements concern critical or important  functions, financial entities shall, prior to concluding the arrangements, take due consideration of the use, by ICT third party service providers, of the most up-to-date and highest quality information security standards. 

6. In exercising access, inspection and audit rights over the ICT third-party service provider, financial entities shall, on  the basis of a risk-based approach, pre-determine the frequency of audits and inspections as well as the areas to be audited  through adhering to commonly accepted audit standards in line with any supervisory instruction on the use and  incorporation of such audit standards. 
Where contractual arrangements concluded with ICT third-party service providers on the use of ICT services entail high  technical complexity, the financial entity shall verify that auditors, whether internal or external, or a pool of auditors,  possess appropriate skills and knowledge to effectively perform the relevant audits and assessments. 

7. Financial entities shall ensure that contractual arrangements on the use of ICT services may be terminated in any of  the following circumstances: 
(a) significant breach by the ICT third-party service provider of applicable laws, regulations or contractual terms; 
(b) circumstances identified throughout the monitoring of ICT third-party risk that are deemed capable of altering the  performance of the functions provided through the contractual arrangement, including material changes that affect  the arrangement or the situation of the ICT third-party service provider; 
(c) ICT third-party service provider’s evidenced weaknesses pertaining to its overall ICT risk management and in particular  in the way it ensures the availability, authenticity, integrity and, confidentiality, of data, whether personal or otherwise  sensitive data, or non-personal data; 
(d) where the competent authority can no longer effectively supervise the financial entity as a result of the conditions of, or  circumstances related to, the respective contractual arrangement. 

8. For ICT services supporting critical or important functions, financial entities shall put in place exit strategies. The exit  strategies shall take into account risks that may emerge at the level of ICT third-party service providers, in particular a  possible failure on their part, a deterioration of the quality of the ICT services provided, any business disruption due to  inappropriate or failed provision of ICT services or any material risk arising in relation to the appropriate and continuous  deployment of the respective ICT service, or the termination of contractual arrangements with ICT third-party service  providers under any of the circumstances listed in paragraph 7. 
Financial entities shall ensure that they are able to exit contractual arrangements without: 
(a) disruption to their business activities, 
(b) limiting compliance with regulatory requirements, 
(c) detriment to the continuity and quality of services provided to clients. 

Exit plans shall be comprehensive, documented and, in accordance with the criteria set out in Article 4(2), shall be  sufficiently tested and reviewed periodically. 
Financial entities shall identify alternative solutions and develop transition plans enabling them to remove the contracted  ICT services and the relevant data from the ICT third-party service provider and to securely and integrally transfer them to  alternative providers or reincorporate them in-house. 
Financial entities shall have appropriate contingency measures in place to maintain business continuity in the event of the  circumstances referred to in the first subparagraph. 

9. The ESAs shall, through the Joint Committee, develop draft implementing technical standards to establish the  standard templates for the purposes of the register of information referred to in paragraph 3, including information that is  common to all contractual arrangements on the use of ICT services. The ESAs shall submit those draft implementing  technical standards to the Commission by 17 January 2024. 
Power is conferred on the Commission to adopt the implementing technical standards referred to in the first subparagraph  in accordance with Article 15 of Regulations (EU) No 1093/2010, (EU) No 1094/2010 and (EU) No 1095/2010. 
EN 
 
10. The ESAs shall, through the Joint Committee, develop draft regulatory technical standards to further specify the  detailed content of the policy referred to in paragraph 2 in relation to the contractual arrangements on the use of ICT  services supporting critical or important functions provided by ICT third-party service providers. 
When developing those draft regulatory technical standards, the ESAs shall take into account the size and the overall risk  profile of the financial entity, and the nature, scale and complexity of its services, activities and operations. The ESAs shall  submit those draft regulatory technical standards to the Commission by 17 January 2024. 
Power is delegated to the Commission to supplement this Regulation by adopting the regulatory technical standards  referred to in the first subparagraph in accordance with Articles 10 to 14 of Regulations (EU) No 1093/2010, (EU)  No 1094/2010 and (EU) No 1095/2010. 

    """


    dialogs = [
        [RawMessage(role="System", content="Inform yourself of the chapter of the dora regulation"),
         RawMessage(role="Document", content="""\
"""
),
RawMessage(role="System", content="Prepair yourself to scan the provided document for compliance with the regulation and answer the tasks"),
RawMessage(role="Document",content="""\
    This Agreement is between Customer and Atlassian. “Customer” means the entity on behalf of which this Agreement is accepted or, if that  does not apply, the individual accepting this Agreement. “Atlassian” means the Atlassian entity that owns or operates the Products that  Customer uses or accesses listed at https://www.atlassian.com/legal/product-terms. 
If you (the person accepting this Agreement) are accepting this Agreement on behalf of your employer or another entity, you agree that:  (i) you have full legal authority to bind your employer or such entity to this Agreement, and (ii) you agree to this Agreement on behalf of  your employer or such entity.  
If you are accepting this Agreement using an email address from your employer or another entity, then: (i) you will be deemed to represent  that party, (ii) your acceptance of this Agreement will bind your employer or that entity to these terms, and (iii) the word “you” or  “Customer” in this Agreement will refer to your employer or that entity.  
By clicking on the “Agree” (or similar button or checkbox) that is presented to you at the time of placing an Order, downloading Products,  or by using or accessing the Products, you confirm you are bound by this Agreement. If you do not wish to be bound by this Agreement,  do not click “Agree” (or similar button or checkbox), download the Products, or use or access the Products. 
1. Overview. This Agreement applies to Customer’s Orders for Products and related Support and Advisory Services. The terms of this  Agreement apply to both Cloud Products and Software Products, although certain terms apply only to Cloud Products or Software Products, as specified below. In addition, some Products are subject to additional Product-Specific Terms, and Support and Advisory Services are subject  to the applicable Policies.  
2. Use of Products. 
2.1. Permitted Use. Subject to this Agreement and during the applicable Subscription Term, Atlassian grants Customer a non-exclusive,  worldwide right to use the Products and related Support and Advisory Services for its and its Affiliates’ internal business purposes, in  accordance with the Documentation and Customer’s Scope of Use. 
2.2. Restrictions. Except to the extent otherwise expressly permitted by this Agreement, Customer must not (and must not permit anyone  else to): (a) rent, lease, sell, distribute or sublicense the Products or (except for Affiliates) include them in a service bureau or outsourcing  offering, (b) provide access to the Products to a third party, other than to Users, (c) charge its customers a specific fee for use of the Products,  but Customer may charge an overall fee for its own offerings (of which the Products are ancillary), (d) use the Products to develop a similar  or competing product or service, (e) reverse engineer, decompile, disassemble or seek to access the source code or non-public APIs to the  Products, (f) modify or create derivative works of the Products, (g) interfere with or circumvent Product usage limits or Scope of Use  restrictions, (h) remove, obscure or modify in any way any proprietary or other notices or attributions in the Products, or (i) violate the  Acceptable Use Policy. 
2.3. DPA. The DPA applies to Customer’s use of Products and related Support and Advisory Services and forms part of this Agreement.  3. Users.  
3.1. Responsibility. Customer may authorize Users to access and use the Products, in accordance with the Documentation and Customer’s  Scope of Use. Customer is responsible for its Users’ compliance with this Agreement and all activities of its Users, including Orders they may  place, apps and Third Party-Products enabled, and how Users access and use Customer Data.  
3.2. Login Credentials. Customer must ensure that each User keeps its login credentials confidential and must promptly notify Atlassian if it  becomes aware of any unauthorized access to any User login credentials or other unauthorized access to or use of the Products.  3.3. Domain Ownership. Where a Cloud Product requires Customer to specify a domain (such as www.example.com) for the Cloud Product’s  or a feature’s operation, Atlassian may verify that Customer or an Affiliate owns or controls that domain. Atlassian has no obligation to  provide that Cloud Product or feature if Atlassian cannot verify that Customer or an Affiliate owns or controls the domain. Product  administrators appointed by Customer may also take over management of accounts previously registered using an email address belonging  to Customer’s domain, which become “managed accounts” (or similar term), as described in the Documentation. 3.4. Age Requirements. The Products are not intended for use by anyone under the age of 16. Customer is responsible for ensuring that all  Users are at least 16 years old. 
4. Cloud Products. This Section 4 only applies to Cloud Products. 
4.1. Customer Data. Atlassian may process Customer Data to provide the Cloud Products and related Support or Advisory Services in  accordance with this Agreement. 
4.2. Security Program. Atlassian has implemented and will maintain an information security program that uses appropriate physical,  technical and organizational measures designed to protect Customer Data from unauthorized access, destruction, use, modification or  disclosure, as described in its Security Measures. Atlassian will also maintain a compliance program that includes independent third-party  audits and certifications, as described in its Security Measures. Further information about Atlassian’s security program is available on the  Atlassian Trust Center at https://www.atlassian.com/trust, as updated from time to time.
Atlassian Customer Agreement 1 June 2024 version 
4.3. Service Levels. Where applicable, service level commitments for the Cloud Products are set out in the Service Level Agreement. 4.4. Data Retrieval. The Documentation describes how Customer may retrieve its Customer Data from the Cloud Products. 4.5. Removals and Suspension. Atlassian has no obligation to monitor Customer Data. Nonetheless, if Atlassian becomes aware that: (a)  Customer Data may violate Law, Section 2.2 (Restrictions), or the rights of others (including relating to a takedown request received following  the guidelines for Reporting Copyright and Trademark Violations at https://www.atlassian.com/legal/copyright-and-trademark-violations),  or (b) Customer’s use of the Cloud Products threatens the security or operation of the Cloud Products, then Atlassian may: (i) limit access to,  or remove, the relevant Customer Data, or (ii) suspend Customer’s or any User’s access to the relevant Cloud Products. Atlassian may also  take any such measures where required by Law, or at the request of a governmental authority. When practicable, Atlassian will give  Customer the opportunity to remedy the issue before taking any such measures. 
5. Software Products. This Section 5 only applies to Software Products. 
5.1. Modifications. Atlassian may provide some portions of the Software Products in source code form for Customer to use internally to  create bug fixes, configurations or other modifications of the Software Products, as permitted in the Documentation (“Modifications”).  Customer must keep such source code secure (on computer devices and online repositories controlled by Customer), confidential, and only  make it available to Customer’s employees who have a legitimate need to access and use the source code to create and maintain  Modifications. Customer may only use Modifications with the Software Products, and only in accordance with this Agreement, including the  Third-Party Code Policy, the Documentation, and Customer’s Scope of Use. Customer must not distribute source code or Modifications to  third parties. Customer must securely destroy the source code at the earliest of: (a) Customer no longer needing to use source code to create  or maintain Modifications, (b) termination or non-renewal of a relevant Subscription Term, or (c) Atlassian’s request for any reason.  Notwithstanding anything else in this Agreement, Atlassian has no support, warranty, indemnity or other responsibility for Modifications. 
5.2. License Verification. Upon Atlassian’s written request, Customer will promptly confirm in writing whether its use of the Software  Products is in compliance with the applicable Scope of Use. Atlassian or its authorized agents may audit Customer’s use of the Software  Products no more than once every twelve (12) months to confirm compliance with Customer’s Scope of Use, provided Atlassian gives  Customer reasonable advance notice and uses reasonable efforts to minimize disruption to Customer. If Customer exceeds its Scope of Use,  Atlassian may invoice for that excess use, and Customer will pay Atlassian promptly after invoice receipt.  
5.3. Number of Instances. Unless otherwise specified in the Order or the Product-Specific Terms, Customer may install up to one (1)  production instance of each Software Product included in an Order on systems owned or operated by Customer or its Users. 
6. Customer Obligations.  
6.1. Disclosures and Rights. Customer must ensure it has made all disclosures and obtained all rights and consents necessary for Atlassian  to use Customer Data and Customer Materials to provide the Cloud Products, Support or Advisory Services. 6.2. Product Assessment. Customer is responsible for determining whether the Products meet Customer’s requirements and any regulatory  obligations related to its intended use. 
6.3. Sensitive Health Information and HIPAA. Unless the parties have entered into a ‘Business Associate Agreement,’ Customer must not  (and must not permit anyone else to) upload to the Cloud Products (or use the Cloud Products to process) any patient, medical or other  protected health information regulated by the Health Insurance Portability and Accountability Act. 
7. Third-Party Code and Third-Party Products.  
7.1. Third-Party Code. This Agreement and the Third-Party Code Policy apply to open source software and commercial third-party software  Atlassian includes in the Products. 
7.2. Third-Party Products. Customer may choose to use the Products with third-party platforms, apps, add-ons, services or products,  including offerings made available through the Atlassian Marketplace (“Third-Party Products”). Use of such Third-Party Products with the  Products may require access to Customer Data and other data by the third-party provider, which, for Cloud Products Atlassian will permit on  Customer’s behalf if Customer has enabled that Third-Party Product. Customer’s use of Third-Party Products is subject to the relevant  provider’s terms of use, not this Agreement. Atlassian does not control and has no liability for Third-Party Products. 
8. Support and Advisory Services. Atlassian will provide Support and Advisory Services as described in the Order and applicable Policies.  Atlassian’s provision of Support or Advisory Services is subject to Customer providing timely access to Customer Materials and personnel  reasonably requested by Atlassian. 
9. Ordering Process and Delivery. No Order is binding until Atlassian provides its acceptance, including by sending a confirmation email,  providing access to the Products, or making license or access keys available to Customer. No terms of any purchase order or other business  form used by Customer will supersede, supplement, or otherwise apply to this Agreement or Atlassian. Atlassian will deliver login instructions  or license keys for Products electronically, to Customer’s account (or through other reasonable means) promptly upon receiving payment of  the fees. Customer is responsible for the installation of Software Products, and Atlassian has no further delivery obligations with respect to  the Software Products after delivery of license keys.  
10. Billing and Payment. 
10.1. Fees. 
Atlassian Customer Agreement 2 June 2024 version 
(a) Direct Purchases. If Customer purchases directly from Atlassian, fees and any payment terms are specified in Customer’s Order with  Atlassian. 
(b) Resellers. If Customer purchases through a Reseller, Customer must pay all applicable amounts directly to the Reseller, and Customer’s  order details (e.g., Products and Scope of Use) will be specified in the Order placed by the Reseller with Atlassian on Customer’s behalf. (c) Renewals. Unless otherwise specified in an Order and subject to the Product, Support or Advisory Services continuing to be generally  available, a Subscription Term will automatically renew at Atlassian’s then current rates for: (i) if Customer’s prior Subscription was for a  period less than twelve (12) months, another Subscription Term of a period equal to Customer’s prior Subscription Term, or (ii) if Customer’s  prior Subscription Team was for twelve (12) months or more, twelve (12) months. Either party may elect not to renew a Subscription Term  by giving notice to the other party before the end of the current Subscription Term. Customer must provide any notice of non-renewal  through account settings in the Products, by contacting Atlassian’s support team or by otherwise providing Atlassian notice. (d) Increased Scope of Use. Customer may increase its Scope of Use by placing a new Order or modifying (by mutual agreement with  Atlassian) an existing Order. Unless otherwise specified in the applicable Order, Atlassian will charge Customer for any increased Scope of  Use at Atlassian’s then-current rates, prorated for the remainder of the then-current Subscription Term. 
(e) Refunds. All fees and expenses are non-refundable, except as otherwise provided in this Agreement. For any purchases Customer  makes through a Reseller, any refunds from Atlassian payable to Customer relating to that purchase will be remitted by that Reseller, unless  Atlassian specifically notifies Customer otherwise at the time of refund. 
(f) Credit Cards. If Customer uses a credit card or similar online payment method for its initial Order, then Atlassian may bill that payment  method for renewals, additional Orders, overages to scopes of use, expenses, and unpaid fees, as applicable. 10.2. Taxes. 
(a) Taxes Generally. Fees and expenses are exclusive of any sales, use, GST, value-added, withholding or similar taxes or levies that apply  to Customer’s Orders. Other than taxes on Atlassian’s net income, Customer is responsible for any such taxes or levies and must pay those  taxes or levies, which Atlassian will itemize separately, in accordance with an applicable invoice. 
(b) Withholding Taxes. To the extent Customer is required to withhold tax from payment to Atlassian in certain jurisdictions, Customer  must provide valid documentation it receives from the taxing authority in such jurisdictions confirming remittance of withholding. This  documentation must be provided at the time of payment of the applicable invoice to Atlassian. 
(c) Exemptions. If Customer claims exemption from any sales tax, VAT, GST or similar taxes under this Agreement, Customer must provide  Atlassian a valid tax exemption certificate or tax ID at the time of Order, and after receipt of valid evidence of exemption, Atlassian will not  include applicable taxes on the relevant Customer invoice. 
10.3. Return Policy. Within thirty (30) days of its initial Order for a Product, Customer may terminate the Subscription Term for that  Product, for any or no reason, by providing notice to Atlassian. Following such termination, upon request (which may be made through  Customer’s Atlassian account), Atlassian will refund Customer the amount paid for that Product and any associated Support under the  applicable Order. Unless otherwise specified in the Policies or Product-Specific Terms, this return policy does not apply to Advisory Services. 
10.4. Suspension for Non-payment. Atlassian may suspend Customer’s rights to use Products or receive Support or Advisory Services if  payment is overdue, and Atlassian has given Customer no fewer than ten (10) days’ written notice. 
11. Atlassian Warranties. 
11.1. Performance Warranties. Atlassian warrants to Customer that: (a) the Products will operate in substantial conformity with the  applicable Documentation during the applicable Subscription Term, (b) Atlassian will not materially decrease the functionality or overall  security of the Products during the applicable Subscription Term, and (c) Atlassian will use reasonable efforts designed to ensure that the  Products, when and as provided by Atlassian, are free of any viruses, malware or similar malicious code (each, a “Performance Warranty”).  11.2. Performance Warranty Remedy. If Atlassian breaches a Performance Warranty and Customer makes a reasonably detailed warranty  claim within 30 days of discovering the issue, Atlassian will use reasonable efforts to correct the non-conformity. If Atlassian determines  such remedy to be impracticable, either party may terminate the affected Subscription Term. Atlassian will then refund to Customer any  pre-paid, unused fees for the terminated portion of the Subscription Term. These procedures are Customer’s exclusive remedy and  Atlassian’s entire liability for breach of a Performance Warranty. 
11.3. Exclusions. The warranties in this Section 11 (Atlassian Warranties) do not apply to: (a) the extent the issue or non-conformity is  caused by Customer’s unauthorized use or modification of the Products, (b) unsupported releases of Software Products or Cloud Clients, or  (c) Third-Party Products. 
11.4. Disclaimers. Except as expressly provided in this Section 11 (Atlassian Warranties), the Products, Support and Advisory Services  and all related Atlassian services and deliverables are provided “AS IS.” Atlassian makes no other warranties, whether express, implied,  statutory or otherwise, including warranties of merchantability, fitness for a particular purpose, title or non-infringement. Atlassian does  not warrant that Customer’s use of the Products will be uninterrupted or error-free. Atlassian is not liable for delays, failures or problems  inherent in use of the internet and electronic communications or other systems outside Atlassian’s control. 
12. Term and Termination. 
12.1. Term. This Agreement commences on the date Customer accepts it and expires when all Subscription Terms have ended.  12.2. Termination for Convenience. Customer may terminate this Agreement or a Subscription Term upon notice for any reason. Subject  to Section 10.3 (Return Policy), Customer will not be entitled to any refunds as a result of exercising its rights under this Section 12.2, and 
Atlassian Customer Agreement 3 June 2024 version 
any unpaid amounts for the then-current Subscription Terms and any related service periods will become due and payable immediately upon  such termination. 
12.3. Termination for Cause. Either party may terminate this Agreement or a Subscription Term if the other party: (a) fails to cure a material  breach of this Agreement (including a failure to pay fees) within 30 days after notice, (b) ceases operation without a successor, or (c) seeks protection under a bankruptcy, receivership, trust deed, creditors’ arrangement, composition or comparable proceeding, or if such a  proceeding is instituted against that party and not dismissed within 60 days. If Customer terminates this Agreement or a Subscription Term  in accordance with this Section 12.3, Atlassian will refund to Customer any pre-paid, unused fees for the terminated portion of the Agreement  or applicable Subscription Term. 
12.4. Effect of Termination. Upon expiration or termination of this Agreement or a Subscription Term: (a) Customer’s rights to use the  applicable Products, Support or Advisory Services will cease, (b) Customer must immediately cease accessing the Cloud Products and using  the applicable Software Products and Cloud Clients, and (c) Customer must delete (or, on request, return) all license keys, access keys and  any Product copies. Following expiration or termination, unless prohibited by Law, Atlassian will delete Customer Data in accordance with  the Documentation.  
12.5. Survival. These Sections survive expiration or termination of this Agreement: 2.2 (Restrictions), 4.2 (Security Program), 10.1 (Fees),  10.2 (Taxes), 11.4 (Disclaimers), 12.4 (Effect of Termination), 12.5 (Survival), 13 (Ownership), 14 (Limitations of Liability), 15 (Indemnification  by Atlassian), 16 (Confidentiality), 17.4 (Disclaimer), 18 (Feedback), 20 (General Terms) and 21 (Definitions). 
13. Ownership. Except as expressly set out in this Agreement, neither party grants the other any rights or licenses to its intellectual property  under this Agreement. As between the parties, Customer owns all intellectual property and other rights in Customer Data and Customer  Materials provided to Atlassian or used with the Products. Atlassian and its licensors retain all intellectual property and other rights in the  Products, any Support and Advisory Services deliverables and related source code, Atlassian technology, templates, formats and dashboards,  including any modifications or improvements. 
14. Limitations of Liability. 
14.1. Damages Waiver. Except for Excluded Claims or Special Claims, to the maximum extent permitted by Law, neither party will have  any liability arising out of or related to this Agreement for any loss of use, lost data, lost profits, interruption of business or any indirect,  special, incidental, reliance or consequential damages of any kind, even if informed of their possibility in advance. 14.2. General Liability Cap. Except for Excluded Claims or Special Claims, to the maximum extent permitted by Law, each party’s entire  liability arising out of or related to this Agreement will not exceed in aggregate the amounts paid to Atlassian for the Products, Support  and Advisory Services giving rise to the liability during the twelve (12) months preceding the first event out of which the liability arose.  Customer’s payment obligations under Sections 10.1 (Fees) and 10.2 (Taxes) are not limited by this Section 14.2. 14.3. Excluded Claims. “Excluded Claims” means: (a) Customer’s breach of Section 2.2 (Restrictions) or Section 6 (Customer Obligations),  (b) either party’s breach of Section 16 (Confidentiality) but excluding claims relating to Customer Data or Customer Materials, or (c) amounts  payable to third parties under Atlassian’s obligations in Section 15 (Indemnification by Atlassian). 
14.4. Special Claims. For Special Claims, Atlassian’s aggregate liability under this Agreement will be the lesser of: (a) two times (2x) the  amounts paid to Atlassian for the Products, Support and Advisory Services giving rise to the Special Claim during the twelve (12) months  preceding the first event out of which the Special Claim arose, and (b) US$5,000,000. “Special Claims” means any unauthorized disclosure  of Customer Data or Customer Materials caused by a breach by Atlassian of its obligations in Section 4.2 (Security Program). 
14.5. Nature of Claims and Failure of Essential Purpose. The exclusions and limitations in this Section 14 (Limitations of Liability) apply  regardless of the form of action, whether in contract, tort (including negligence), strict liability or otherwise and will survive and apply even  if any limited remedy in this Agreement fails of its essential purpose. 
15. Indemnification by Atlassian. 
15.1. IP Indemnification. Atlassian must: (a) defend Customer from and against any third-party claim to the extent alleging that the  Products, when used by Customer as authorized by this Agreement, infringe any intellectual property right of a third party (an “Infringement  Claim”), and (b) indemnify and hold harmless Customer against any damages, fines or costs finally awarded by a court of competent  jurisdiction (including reasonable attorneys’ fees) or agreed in settlement by Atlassian resulting from an Infringement Claim. 
15.2. Procedures. Atlassian’s obligations in Section 15.1 (IP Indemnification) are subject to Customer providing: (a) sufficient notice of the  Infringement Claim so as to not prejudice Atlassian’s defense of the Infringement Claim, (b) the exclusive right to control and direct the  investigation, defense and settlement of the Infringement Claim, and (c) all reasonably requested cooperation, at Atlassian’s expense for  reasonable out-of-pocket expenses. Customer may participate in the defense of an Infringement Claim with its own counsel at its own  expense. 
15.3. Settlement. Customer may not settle an Infringement Claim without Atlassian’s prior written consent. Atlassian may not settle an  Infringement Claim without Customer’s prior written consent if settlement would require Customer to admit fault or take or refrain from  taking any action (other than relating to use of the Products). 
15.4. Mitigation. In response to an actual or potential Infringement Claim, Atlassian may, at its option: (a) procure rights for Customer’s  continued use of the Products, (b) replace or modify the alleged infringing portion of the Products without reducing the overall functionality  of the Products, or (c) terminate the affected Subscription Term and refund to Customer any pre-paid, unused fees for the terminated portion  of the Subscription Term.
Atlassian Customer Agreement 4 June 2024 version 
15.5. Exceptions. Atlassian’s obligaeons in this Seceon 15 (Indemnificaeon by Atlassian) do not apply to the extent an Infringement Claim  arises from: (a) Customer’s modificaeon or unauthorized use of the Products, (b) use of the Products in combinaeon with items not provided  by Atlassian (including Third-Party Products), (c) any unsupported release of the Sogware Products or Cloud Clients, or (d) Third-Party  Products, Customer Data or Customer Materials. 
15.6. Exclusive Remedy. This Section 15 (Indemnification by Atlassian) sets out Customer’s exclusive remedy and Atlassian’s entire  liability regarding infringement of third-party intellectual property rights. 
16. Confidentiality.  
16.1. Definition. “Confidential Information” means information disclosed by one party to the other under or in connection with this  Agreement that: (a) is designated by the disclosing party as proprietary or confidential, or (b) should be reasonably understood to be  proprietary or confidential due to its nature and the circumstances of its disclosure. Atlassian’s Confidential Information includes any source  code and technical or performance information about the Products. Customer’s Confidential Information includes Customer Data and  Customer Materials. 
16.2. Obligations. Unless expressly permitted by the disclosing party in writing, the receiving party must: (a) hold the disclosing party’s  Confidential Information in confidence and not disclose it to third parties except as permitted in this Agreement, and (b) only use such  Confidential Information to fulfill its obligations and exercise its rights in this Agreement. The receiving party may disclose such Confidential  Information to its employees, agents, contractors and other representatives having a legitimate need to know (including, for Atlassian, the  subcontractors referenced in Section 20.11 (Subcontractors and Affiliates)), provided the receiving party remains responsible for their  compliance with this Section 16 (Confidentiality) and they are bound to confidentiality obligations no less protective than this Section 16 (Confidentiality). 
16.3. Exclusions. These confidentiality obligations do not apply to information that the receiving party can demonstrate: (a) is or becomes  publicly available through no fault of the receiving party, (b) it knew or possessed prior to receipt under this Agreement without breach of  confidentiality obligations, (c) it received from a third party without breach of confidentiality obligations, or (d) it independently developed  without using the disclosing party’s Confidential Information. The receiving party may disclose Confidential Information if required by Law,  subpoena or court order, provided (if permitted by Law) it notifies the disclosing party in advance and cooperates, at the disclosing party’s  cost, in any reasonable effort to obtain confidential treatment. 
16.4. Remedies. Unauthorized use or disclosure of Confidential Information may cause substantial harm for which damages alone are an  insufficient remedy. Each party may seek appropriate equitable relief, in addition to other available remedies, for breach or anticipated  breach of this Section 16 (Confidentiality). 
17. Free or Beta Products.  
17.1. Access. Customer may receive access to certain Products or Product features on a free, fully discounted or trial basis, or as an alpha,  beta or early access offering (“Free or Beta Products”). Use of Free or Beta Products is subject to this Agreement and any additional terms  specified by Atlassian, such as the applicable scope and term of use. 
17.2. Termination or Modification. At any time, Atlassian may terminate or modify Customer’s use of (including applicable terms) Free or  Beta Products or modify Free or Beta Products, without any liability to Customer. For modifications to Free or Beta Products or Customer’s  use, Customer must accept those modifications to continue accessing or using the Free or Beta Products. 
17.3. Pre GA. Free or Beta Products may be inoperable, incomplete or include errors and bugs or features that Atlassian may never release,  and their features and performance information are Atlassian’s Confidential Information. 
17.4. Disclaimer. Notwithstanding anything else in this Agreement, to the maximum extent permitted by Law, Atlassian provides no  warranty, indemnity, service level agreement or support for Free or Beta Products and its aggregate liability for Free or Beta Products is  limited to US$100. 
18. Feedback. If Customer provides Atlassian with feedback or suggestions regarding the Products or other Atlassian offerings, Atlassian  may use the feedback or suggestions without restriction or obligation. 
19. Publicity. Atlassian may identify Customer as a customer of Atlassian in its promotional materials. Atlassian will promptly stop doing  so upon Customer request sent to sales@atlassian.com. 
20. General Terms. 
20.1. Compliance with Laws. Each party must comply with all Laws applicable to its business in its performance of obligations or exercise  of rights under this Agreement. 
20.2. Code of Conduct. Atlassian must comply with its Code of Conduct in its performance of obligations or exercise of rights under this  Agreement. 
20.3. Assignment.  
(a) Customer may not assign or transfer any of its rights or obligations under this Agreement or an Order without Atlassian’s prior written  consent. However, Customer may assign this Agreement in its entirety (including all Orders) to its successor resulting from a merger,  acquisition, or sale of all or substantially all of Customer’s assets or voting securities, provided that Customer provides Atlassian with prompt 
Atlassian Customer Agreement 5 June 2024 version 
written notice of the assignment and the assignee agrees in writing to assume all of Customer’s obligations under this Agreement and  complies with Atlassian’s procedural and documentation requirements to give effect to the assignment.  
(b) Any attempt by Customer to transfer or assign this Agreement or an Order, except as expressly authorized above, will be null and void. (c) Atlassian may assign its rights and obligations under this Agreement (in whole or in part) without Customer’s consent. 20.4. Governing Law, Jurisdiction and Venue. 
(a) If Customer is domiciled: (i) in Europe, the Middle East, or Africa, this Agreement is governed by the laws of the Republic of Ireland,  with the jurisdiction and venue for actions related to this Agreement in the courts of the Republic of Ireland, or (ii) elsewhere, this Agreement  is governed by the laws of the State of California, with the jurisdiction and venue for actions related to this Agreement in the state and United  States federal courts located in San Francisco, California.  
(b) This Agreement will be governed by such laws without regard to conflicts of laws provisions, and both parties submit to the personal  jurisdiction of the applicable courts. The United Nations Convention on the International Sale of Goods does not apply to this Agreement. 20.5. Notices.  
(a) Except as specified elsewhere in this Agreement, notices under this Agreement must be in writing and are deemed given on: (i) personal  delivery, (ii) when received by the addressee if sent by a recognized overnight courier with receipt request, (iii) the third business day after  mailing, or (iv) the first business day after sending by email, except that email will not be sufficient for notices regarding Infringement Claims,  alleging breach of this Agreement by Atlassian, or of Customer’s termination of this Agreement in accordance with Section 12.3 (Termination  for Cause).  
(b) Notices to Atlassian must be provided according to the details provided at https://www.atlassian.com/legal#how-do-i-provide-legal notices-to-atlassian, as may be updated from time to time. 
(c) Notices to Customer must be provided to the billing or technical contact provided to Atlassian, which may be updated by Customer  from time to time in Customer’s account portal. However, Atlassian may provide general or operational notices via email, on its website or  through the Products. Customer may subscribe to receive email notice of updates to this Agreement, as described at  https://www.atlassian.com/legal#notification-of-updates-in-terms-and-policies. 
20.6. Entire Agreement. This Agreement is the parties’ entire agreement regarding its subject matter and supersedes any prior or  contemporaneous agreements regarding its subject matter. In the event of a conflict among the documents making up this Agreement, the  main body of this Agreement (i.e., Sections 1 through 21, inclusive) will control, except that the Policies, Product-Specific Terms and DPA will  control for their specific subject matter. 
20.7. Other Atlassian Offerings. Atlassian makes available other offerings that can be used with the Products which, in some cases, are  subject to separate terms and conditions, available at https://www.atlassian.com/legal. These other offerings include training services,  developer tools and the Atlassian Marketplace. For clarity, this Agreement controls over any such terms and conditions with respect to  Customer’s use of the Products (including any Atlassian Apps). 
20.8. Interpretation, Waivers and Severability. In this Agreement, headings are for convenience only and “including” and similar terms are  to be construed without limitation. Waivers must be granted in writing and signed by the waiving party’s authorized representative. If any  provision of this Agreement is held invalid, illegal or unenforceable, it will be limited to the minimum extent necessary so the rest of this  Agreement remains in effect. 
20.9. Changes to this Agreement. 
(a) Atlassian may modify this Agreement (which includes the Policies, Product-Specific Terms and DPA) from time to time, by posting the  modified portion(s) of this Agreement on Atlassian’s website. Atlassian must use commercially reasonable efforts to post any such  modification at least thirty (30) days prior to its effective date. 
(b) For free subscriptions, modifications become effective during the then current Subscription Term, in accordance with Atlassian’s notice. (c) For paid subscriptions: 
(i) except as specified below, modifications to this Agreement will take effect at the next Order or renewal unless either party elects to  not renew pursuant to Section 10.1(c) (Renewals), and 
(ii)Atlassian may specify that modifications will become effective during a then-current Subscription Term if: (A) required to address  compliance with Law, or (B) required to reflect updates to Product functionality or introduction of new Product features. If Customer  objects, Customer may terminate the remainder of the then-current Subscription Term for the affected Products as its exclusive  remedy. To exercise this right, Customer must notify Atlassian of its termination under this Section 20.9(c) within thirty (30) days of  the modification notice, and Atlassian will refund any pre-paid fees for the terminated portion of the applicable Subscription Term. 
20.10. Force Majeure. Neither party is liable for any delay or failure to perform any obligation under this Agreement (except for a failure to  pay fees) due to events beyond its reasonable control and occurring without that party’s fault or negligence. 20.11. Subcontractors and Affiliates. Atlassian may use subcontractors or its Affiliates in the performance of its obligations under this  Agreement, but Atlassian remains responsible for its overall performance under this Agreement and for having appropriate written  agreements in place with its subcontractors to enable Atlassian to meet its obligations under this Agreement.  20.12. Independent Contractors. The parties are independent contractors, not agents, partners or joint venturers. 20.13. Export Restrictions. The Products may be subject to U.S. export restrictions and import restrictions of other jurisdictions. Customer  must comply with all applicable export and import Laws in its access to, use of, and download of the Products or any content or records  entered into the Products. Customer must not (and must not allow anyone else to) export, re-export, transfer or disclose the Products or 
Atlassian Customer Agreement 6 June 2024 version 
any direct product of the Products: (a) to (or to a national or resident of) any U.S. embargoed jurisdiction, (b) to anyone on any U.S. or  applicable non-U.S. restricted- or denied-party list, or (c) to any party that Customer has reason to know will use the Products in violation of  U.S. export Law, or for any restricted end user under U.S. export Law.  
20.14. Government End-Users. If Customer is a United States federal, state or local government customer, this Agreement is subject to, and  is varied by, the Government Amendment available at https://www.atlassian.com/legal/government-amendment. 
20.15. No Contingencies. The Products, Support and Advisory Services in each Order are purchased separately and not contingent on  purchase or use of other Atlassian products and services, even if listed in the same Order. Customer’s purchases are not contingent on  delivery of any future functionality or features. 
21. Definitions. 
“Acceptable Use Policy” means Atlassian’s acceptable use policy available at https://www.atlassian.com/legal/acceptable-use-policy.  “Advisory Services” means advisory services as described in the Advisory Services Policy. 
“Advisory Services Policy” means Atlassian’s advisory services policy available at https://www.atlassian.com/legal/advisory-services-policy.  “Affiliate” means an entity that, directly or indirectly, owns or controls, is owned or is controlled by or is under common ownership or control  with a party, where “ownership” means the beneficial ownership of more than fifty percent (50%) of an entity’s voting equity securities or  other equivalent voting interests and “control” means the power to direct the management or affairs of an entity. 
“Agreement” means this Atlassian Customer Agreement, as well as the Product-Specific Terms, the DPA and the Policies. “Atlassian Apps” means apps developed by Atlassian for use with Cloud Products or Software Products, as designated by Atlassian in the  Atlassian Marketplace.  
“Atlassian Marketplace” means the online platform to purchase apps for Atlassian products currently branded the Atlassian Marketplace  and accessible at https://marketplace.atlassian.com/. 
“Cloud Products” means Atlassian’s cloud products, including client software for its cloud products (“Cloud Clients”). “Code of Conduct” means the Atlassian Code of Business Conduct & Ethics, available at  https://investors.atlassian.com/governance/governance-documents/default.aspx. 
“Customer Data” means any data, content or materials provided to Atlassian by or at the direction of Customer or its Users via the Cloud  Products, including from Third-Party Products. 
“Customer Materials” means materials and other resources that Customer provides to Atlassian in connection with Support or Advisory  Services. 
“Documentation” means Atlassian’s usage guidelines and standard technical documentation for the applicable Product, available at  https://support.atlassian.com/, unless otherwise specified in the Product-Specific Terms. 
“DPA” means the Atlassian data processing addendum available at https://www.atlassian.com/legal/data-processing-addendum. “Laws” means all applicable laws, regulations, conventions, decrees, decisions, orders, judgments, codes and requirements of any  government authority (federal, state, local or international) having jurisdiction. 
“Order” means Atlassian’s ordering document or online order specifying the Products, Support or Advisory Services to be provided under  this Agreement, accepted by Atlassian in accordance with Section 9 (Ordering Process and Delivery).  
“Policies” means the Acceptable Use Policy, Advisory Services Policy, guidelines for Reporting Copyright and Trademark Violations, Privacy  Policy, Security Measures, Service Level Agreement, Support Policy, Third-Party Code Policy and any additional Atlassian policies specified in  Product-Specific Terms. 
“Privacy Policy” means Atlassian’s privacy policy available at https://www.atlassian.com/legal/privacy-policy. “Products” means the applicable Cloud Products or Software Products made available by Atlassian in connection with an Order. Products  also include Atlassian Apps.  
“Product-Specific Terms” means product-specific terms that apply only to certain Products, available at  https://www.atlassian.com/legal/product-terms. 
“Reseller” means a partner authorized by Atlassian to resell Atlassian’s Products, Support and Advisory Services to customers.  “Scope of Use” means Customer’s entitlements to the Products specified in an Order, which may include: (a) number and type of Users, (b)  numbers of licenses, copies or instances, or (c) entity, division, business unit, website, field of use or other restrictions or billable units. “Security Measures” means Atlassian’s security practices available at https://www.atlassian.com/legal/security-measures.  “Service Level Agreement” means the service level commitments, if any, for a Cloud Product as described at  https://www.atlassian.com/legal/sla. 
“Software Products” means Atlassian’s installed software products and any generally-available bug fixes, updates and upgrades it provides  to Customer, including through Support.  
“Subscription Term” means the term for Customer’s use of or access to the Products and related Support and Advisory Services as identified  in an Order. 
“Support” means the level of support for the Products corresponding to Customer’s Scope of Use, as identified in the Support Policy. “Support Policy” means the Atlassian support offerings documentation available at https://confluence.atlassian.com/support/atlassian-
Atlassian Customer Agreement 7 June 2024 version 
support-offerings-193299636.html. 
“Third-Party Code Policy” means Atlassian’s third-party code policy available at https://www.atlassian.com/legal/third-party-code-policy.  “User” means any individual that Customer authorizes to use the Products. Users may include: (i) Customer’s and its Affiliates’ employees,  consultants, contractors and agents (ii) third parties with which Customer or its Affiliates transact business (iii) individuals invited by  Customer’s users (iv) individuals under managed accounts, or (v) individuals interacting with a Product as Customer’s customer.
"""),

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
