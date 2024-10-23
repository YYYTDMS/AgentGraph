import re
import numpy as np


def get_expert_introduction(expert_name):
    expert_introduction=""
    if "One-Hop" in expert_name:
        expert_introduction="One-Hop Neighbors Expert specializes in one-hop neighbor relationships in graph mining. Its task is to analyze the graph based on one-hop neighbor relationships within the provided dataset."
    elif "Random Walks" in expert_name:
        expert_introduction="Random Walks Expert specializes in random walk methods within graph mining. Its task is to analyze the graph using random walk methodologies based on the provided dataset."
    elif "Centrality" in expert_name:
        expert_introduction = "Centrality Expert specializes in centrality measures in graph mining. Its task is to evaluate the graph based on centrality metrics such as degree, closeness, and betweenness, and to identify the most important nodes for classification."
    elif "Categories" in expert_name:
        expert_introduction = "Categories Expert specializes in category analysis in graph mining. Its task is to evaluate the given question by examining its relevance and correctness in relation to the provided categories within the dataset."
    elif "Multi-domains" in expert_name:
        expert_introduction="Multi-domains Expert specializes in multi-domain analysis within graph mining. Its task is to integrate multiple methodologies to comprehensively analyze the graph based on the provided dataset."
    elif expert_name=="GCN":
        expert_introduction="GCN is a graph mining expert, depending on the key operation of neighborhood aggregation to aggregate messages from the neighboring nodes to form the node representations."
    elif expert_name=="GAT":
        expert_introduction="GAT is a graph mining expert, integrating attention mechanisms, assigning varying weights to nodes in a neighborhood, thereby sharpening the model’s ability to focus on significant nodes."
    elif expert_name=="TAPE (GCN)":
        expert_introduction="TAPE (GCN) is a graph mining expert, using LLM to generate informative node features via encoding the augmented text data, boosting the performance of GCN."
    elif expert_name=="R-GCN":
        expert_introduction="R-GCN is a graph mining expert, extending traditional GCN to handle multiple edge types by first performing mean aggregation on each edge type and then aggregating the representations across all edge types."
    return expert_introduction


def replace_label_list(data):
    instruction=str(data['instruction'])
    if 'Arxiv' in instruction:
        options_str = ("{cs.NA; cs.MM; cs.LO; cs.CY; cs.CR; cs.DC; cs.HC; cs.CE; cs.NI; cs.CC; cs.AI; cs.MA; cs.GL; "
                       "cs.NE; cs.SC; cs.AR; cs.CV; cs.GR; cs.ET; cs.SY; cs.CG; cs.OH; cs.PL; cs.SE; cs.LG; cs.SD; "
                       "cs.SI; cs.RO; cs.IT; cs.PF; cs.CL; cs.IR; cs.MS; cs.FL; cs.DS; cs.OS; cs.GT; cs.DB; cs.DL; "
                       "cs.DM}")
        old_str=(
            "{1. cs.NA 2. cs.MM 3. cs.LO 4. cs.CY 5. cs.CR 6. cs.DC 7. cs.HC 8. cs.CE 9. cs.NI 10. cs.CC 11. cs.AI "
            "12. cs.MA 13. cs.GL 14. cs.NE 15. cs.SC 16. cs.AR 17. cs.CV 18. cs.GR 19. cs.ET 20. cs.SY 21. cs.CG 22. "
            "cs.OH 23. cs.PL 24. cs.SE 25. cs.LG 26. cs.SD 27. cs.SI 28. cs.RO 29. cs.IT 30. cs.PF 31. cs.CL 32. "
            "cs.IR 33. cs.MS 34. cs.FL 35. cs.DS 36. cs.OS 37. cs.GT 38. cs.DB 39. cs.DL 40. cs.DM}")

    elif 'IMDB' in instruction:
        options_str = "{Romance; Thriller; Comedy; Action; Drama}"
        old_str = "{1. Romance 2. Thriller 3. Comedy 4. Action 5. Drama}"
    elif "Cora" in instruction:
        options_str = ("{theory; reinforcement learning; genetic algorithms; neural networks; probabilistic methods; "
                       "case based; rule learning}")
        old_str = (
            "{1. theory 2. reinforcement learning 3. genetic algorithms 4. neural networks 5. probabilistic methods 6. case based 7. rule learning}")

    instruction = instruction.replace(old_str, options_str)
    new_data=data
    new_data['instruction']=instruction
    return new_data

def replace_instruction_format(data):
    instruction = str(data['instruction'])
    instruction=instruction.replace(" Using the following format: Answer: [Answer]\nReason: [Reason].","")
    new_data=data
    new_data['instruction']=instruction
    return new_data

def get_question_analysis_one_hop_role():
    question_analyzer = ("You are a graph mining expert specializing in one-hop neighbors. Analyze the graph based on one-hop neighbor relationships within the provided dataset.")
    return question_analyzer

def get_question_analysis_random_walk_role():
    question_analyzer = ("You are a graph mining expert specializing in random walks. Analyze the graph focusing on random walk methodologies within the provided dataset.")
    return question_analyzer

def get_question_analysis_centrality_role():
    question_analyzer = ("You are a graph mining expert specializing in centrality measures. Your task is to evaluate the graph based on centrality metrics such as degree, closeness, and betweenness, and identify the most important nodes for classification.")
    return question_analyzer

def get_question_analysis_categories_role():
    question_analyzer = ("You are a graph mining expert specializing in category analysis. Evaluate the given question by analyzing its relevance and correctness in relation to the provided categories.")
    return question_analyzer

def get_question_analysis_multi_domains_role():
    question_analyzer = ("You are a graph mining expert specializing in various graph-related tasks. Your task is to analyze the given dataset, applying your extensive knowledge of graph mining.")
    return question_analyzer

def get_question_analysis_one_hop_prompt(question):
    prompt_get_question_analysis = (f"Consider the following question:\n\n'''{question}'''\n\nUtilizing your expertise in one-hop neighbors, interpret the graph's conditions and emphasize key aspects related to one-hop neighbors. Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.")
    prompt_get_question_analysis+=" "+"Ensure that your explanation aligns with the answer you provide. Please present your answer and reason in the following format:\n'''\nAnswer: \n[Answer]\nReason: \n[Reason]\n'''"
    return prompt_get_question_analysis

def get_question_analysis_random_walk_prompt(question):
    prompt_get_question_analysis = (f"Consider the following question:\n\n'''{question}'''\n\nUsing your expertise in random walks, interpret the graph's conditions and highlight key findings related to random walks. Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.")
    prompt_get_question_analysis+=" "+"Please present your answer and reason in the following format:\n'''\nAnswer: \n[Answer]\nReason: \n[Reason]\n'''"
    return prompt_get_question_analysis

def get_question_analysis_centrality_prompt(question):
    prompt_get_question_analysis = (f"Consider the following question:\n\n'''{question}'''\n\nUsing your expertise in centrality measures, identify the most influential nodes based on metrics like degree, closeness, and betweenness. Highlight key nodes that play a critical role in the graph's structure, which could significantly impact node classification. Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.")
    prompt_get_question_analysis+=" "+"Ensure that your explanation aligns with the answer you provide. Please present your answer and reason in the following format:\n'''\nAnswer: \n[Answer]\nReason: \n[Reason]\n'''"
    return prompt_get_question_analysis

def get_question_analysis_categories_prompt(question):
    prompt_get_question_analysis = (f"Consider the following question:\n\n'''{question}'''\n\nUtilize your expertise in logical reasoning and graph structure analysis to assess the plausibility and applicability of each category. Ensure that your analysis explicitly states the specific relevance of each category, including supporting or opposing arguments, and provide appropriate explanations for categories. Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.")
    prompt_get_question_analysis+=" "+"Ensure that your explanation aligns with the answer you provide. Please present your answer and reason in the following format:\n'''\nAnswer: \n[Answer]\nReason: \n[Reason]\n'''"
    return prompt_get_question_analysis

def get_question_analysis_multi_domains_prompt(question):
    prompt_get_question_analysis = (f"Consider the following question:\n\n'''{question}'''\n\nUsing your expertise in graph mining to understand and analyze the provided graph. Identify and summarize the key findings relevant to this question. Please present three probable categories in a comma-separated list, ordered from most likely to least likely. For each category, explain your reasoning.")
    prompt_get_question_analysis+=" "+"Ensure that your explanation aligns with the answer you provide. Please present your answer and reason in the following format:\n'''\nAnswer: \n[Answer]\nReason: \n[Reason]\n'''"
    return prompt_get_question_analysis

#--------------------------------------------------------------------------------------

def extract_answer(analysis,cs_list):
    ans = str(analysis).split("Reason:")[0].replace("Answer:", "").replace(
        "[Category]", "").lower()

    def find_first_second_third_label(final_report):

        first_label = None
        second_label = None
        third_label = None
        for label in cs_list:
            index = final_report.find(label)
            if index != -1:
                if first_label is None or index < final_report.find(first_label):
                    first_label = label
        if first_label is None:
            return [None, None, None]

        first_index = final_report.find(first_label)
        final_report = final_report[first_index + len(first_label):]

        for label in cs_list:
            index = final_report.find(label)
            if index != -1:
                if second_label is None or index < final_report.find(second_label):
                    second_label = label
        if second_label is None:
            return [first_label, None, None]

        second_index = final_report.find(second_label)
        final_report = final_report[second_index + len(second_label):]
        for label in cs_list:
            index = final_report.find(label)
            if index != -1:
                if third_label is None or index < final_report.find(third_label):
                    third_label = label
        if third_label is None:
            if first_label == second_label:
                second_label = None
            return [first_label, second_label, None]
        else:
            if first_label == second_label:
                second_label = third_label
                third_label = None
                if first_label == second_label:
                    second_label = None
            elif second_label == third_label:
                third_label = None
            return [first_label, second_label, third_label]

    three_label = find_first_second_third_label(ans)
    if three_label[2] is not None:
        temp = three_label
    elif three_label[1] is not None:
        temp = [three_label[0],three_label[1]]
    elif three_label[0] is not None:
        temp = [three_label[0]]
    else:
        temp = []
    return temp

def calculate_answers_distribution(labels_dict):
    total_labels = sum(labels_dict.values())
    probabilities = np.array(list(labels_dict.values())) / total_labels

    entropy = -np.sum(probabilities * np.log(probabilities))
    return round(entropy, 2)

def calculate_answers_entropy(analysis_dict,options_list):
    labels_dict = {key: 0 for key in options_list}
    for key,value in analysis_dict.items():
        answer_list=extract_answer(value,options_list)
        for label in answer_list:
            labels_dict[label]+=1
    return calculate_answers_distribution({key: value for key, value in labels_dict.items() if value != 0})

#--------------------------------------------------------------------------------------

def get_reflection_one_hop_role():
    question_analyzer = ("You are a graph mining expert specializing in one-hop neighbors. Your task is to reflect on your initial analysis of the Question, taking into account the Question and the insights provided by other experts. Reassess your previous analysis and consider how it could be improved or adjusted based on these new perspectives.")
    return question_analyzer

def get_reflection_random_walk_role():
    question_analyzer = ("You are a graph mining expert specializing in random walks. Your task is to reflect on your initial analysis of the Question, taking into account the Question and the insights provided by other experts. Reassess your previous analysis and consider how it could be improved or adjusted based on these new perspectives.")
    return question_analyzer

def get_reflection_centrality_role():
    question_analyzer = ("You are a graph mining expert specializing in centrality measures. Your task is to reflect on your initial analysis of the Question, taking into account the Question and the insights provided by other experts. Reassess your previous analysis and consider how it could be improved or adjusted based on these new perspectives.")
    return question_analyzer

def get_reflection_categories_role():
    question_analyzer = ("You are a graph mining expert specializing in category analysis. Your task is to reflect on your initial analysis of the Question, taking into account the Question and the insights provided by other experts. Reassess your previous analysis and consider how it could be improved or adjusted based on these new perspectives.")
    return question_analyzer

def get_reflection_multi_domains_role():
    question_analyzer = ("You are a graph mining expert specializing in various graph-related tasks. Your task is to reflect on your initial analysis of the Question, taking into account the Question and the insights provided by other experts. Reassess your previous analysis and consider how it could be improved or adjusted based on these new perspectives.")
    return question_analyzer

def get_reflection_one_hop_prompt(question, sys_report, initial_analysis, random_walks_initial_analysis, centrality_initial_analysis,categories_initial_analysis,multi_domains_initial_analysis,
                                  GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis):
    prompt_get_question_analysis = f"Question:\n\n'''\n{question}\n'''\n\nBased on the Question, you have prepared a preliminary analysis:\n\n'''\n{initial_analysis}\n'''\n\n"

    expert_information=get_expert_introduction("Random Walks Expert")
    prompt_get_question_analysis += f"Random Walks Expert:\n{expert_information}"+"\n{"+ random_walks_initial_analysis+"}\n"


    expert_information = get_expert_introduction("Centrality Expert")
    prompt_get_question_analysis +=(f"Centrality Expert:\n{expert_information}"+"\n{"+ centrality_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Categories Expert")
    prompt_get_question_analysis +=(f"Categories Expert:\n{expert_information}"+"\n{"+ categories_initial_analysis+"}\n")

    expert_information = get_expert_introduction("Multi-domains Expert")
    prompt_get_question_analysis +=(f"Multi-domains Expert:\n{expert_information}"+"\n{"+ multi_domains_initial_analysis+"}\n")

    prompt=""
    if len(GCN_Analysis) != 0:
        expert_information = get_expert_introduction("GCN")
        prompt_get_question_temp = (
                    f"GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(GAT_Analysis) != 0:
        expert_information = get_expert_introduction("GAT")
        prompt_get_question_temp = (
                    f"GAT Expert:\n{expert_information}" + "\n- Analysis:\n{" + GAT_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(TAPE_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("TAPE (GCN)")
        prompt_get_question_temp = (
                    f"TAPE (GCN) Expert:\n{expert_information}" + "\n- Analysis:\n{" + TAPE_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(R_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("R-GCN")
        prompt_get_question_temp = (
                    f"R-GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + R_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp

    prompt_get_question_analysis +="1. You will receive preliminary analysis from other experts and synthesized report. Critically review and analyze these insights.\n2. If you find aspects of other experts' analysis that are more rational than yours, incorporate these into your analysis for improvement.\n3. If you believe your analysis is more scientifically sound compared to others, maintain your stance.\n"

    prompt_get_question_analysis += prompt

    prompt_get_question_analysis += (f"\nPlease review the following synthesized report:\n{sys_report}\n\n")

    prompt_get_question_analysis += ("Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.\n\nPlease present your reflection in the following format:\n\n'''\nAnswer:\n[Answer]\nReason: \n[Reason]\n'''")
    return prompt_get_question_analysis

def get_reflection_random_walk_prompt(question, sys_report,initial_analysis,one_hop_initial_analysis, centrality_initial_analysis,categories_initial_analysis,multi_domains_initial_analysis,
                                      GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis):
    prompt_get_question_analysis = f"Question:\n\n'''\n{question}\n'''\n\nBased on the Question, you have prepared a preliminary analysis:\n\n'''\n{initial_analysis}\n'''\n\n"

    expert_information = get_expert_introduction("One-Hop Neighbors Expert")
    prompt_get_question_analysis +=(f"One-hop Neighbors Expert:\n{expert_information}"+"\n{"+ one_hop_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Centrality Expert")
    prompt_get_question_analysis +=(f"Centrality Expert:\n{expert_information}"+"\n{"+ centrality_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Categories Expert")
    prompt_get_question_analysis +=(f"Categories Expert:\n{expert_information}"+"\n{"+ categories_initial_analysis+"}\n")

    expert_information = get_expert_introduction("Multi-domains Expert")
    prompt_get_question_analysis +=(f"Multi-domains Expert:\n{expert_information}"+"\n{"+ multi_domains_initial_analysis+"}\n")

    prompt=""
    if len(GCN_Analysis) != 0:
        expert_information = get_expert_introduction("GCN")
        prompt_get_question_temp = (
                    f"GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(GAT_Analysis) != 0:
        expert_information = get_expert_introduction("GAT")
        prompt_get_question_temp = (
                    f"GAT Expert:\n{expert_information}" + "\n- Analysis:\n{" + GAT_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(TAPE_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("TAPE (GCN)")
        prompt_get_question_temp = (
                    f"TAPE (GCN) Expert:\n{expert_information}" + "\n- Analysis:\n{" + TAPE_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(R_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("R-GCN")
        prompt_get_question_temp = (
                    f"R-GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + R_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp

    prompt_get_question_analysis +="1. You will receive preliminary analysis from other experts and synthesized report. Critically review and analyze these insights.\n2. If you find aspects of other experts' analysis that are more rational than yours, incorporate these into your analysis for improvement.\n3. If you believe your analysis is more scientifically sound compared to others, maintain your stance.\n"
    prompt_get_question_analysis+=prompt

    prompt_get_question_analysis +=(f"\nPlease review the following synthesized report:\n{sys_report}\n\n")

    prompt_get_question_analysis += ("Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.\n\nPlease present your reflection in the following format:\n\n'''\nAnswer:\n[Answer]\nReason: \n[Reason]\n'''")
    return prompt_get_question_analysis

def get_reflection_centrality_prompt(question, sys_report,initial_analysis,one_hop_initial_analysis, random_walks_initial_analysis,categories_initial_analysis, multi_domains_initial_analysis,
                                     GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis):
    prompt_get_question_analysis = f"Question:\n\n'''\n{question}\n'''\n\nBased on the Question, you have prepared a preliminary analysis:\n\n'''\n{initial_analysis}\n'''\n\n"

    expert_information = get_expert_introduction("One-Hop Neighbors Expert")
    prompt_get_question_analysis +=(f"One-hop Neighbors Expert:\n{expert_information}"+"\n{"+ one_hop_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Random Walks Expert")
    prompt_get_question_analysis +=(f"Random Walks Expert:\n{expert_information}"+"\n{"+ random_walks_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Categories Expert")
    prompt_get_question_analysis +=(f"Categories Expert:\n{expert_information}"+"\n{"+ categories_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Multi-domains Expert")
    prompt_get_question_analysis +=(f"Multi-domains Expert:\n{expert_information}"+"\n{"+ multi_domains_initial_analysis+"}\n")

    prompt=""
    if len(GCN_Analysis) != 0:
        expert_information = get_expert_introduction("GCN")
        prompt_get_question_temp = (
                    f"GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(GAT_Analysis) != 0:
        expert_information = get_expert_introduction("GAT")
        prompt_get_question_temp = (
                    f"GAT Expert:\n{expert_information}" + "\n- Analysis:\n{" + GAT_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(TAPE_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("TAPE (GCN)")
        prompt_get_question_temp = (
                    f"TAPE (GCN) Expert:\n{expert_information}" + "\n- Analysis:\n{" + TAPE_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(R_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("R-GCN")
        prompt_get_question_temp = (
                    f"R-GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + R_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp

    prompt_get_question_analysis +=("1. You will receive preliminary analysis from other experts and synthesized report. Critically review and analyze these insights.\n"
                                    "2. If you find aspects of other experts' analysis that are more rational than yours, incorporate these into your analysis for improvement.\n"
                                    "3. If you believe your analysis is more scientifically sound compared to others, maintain your stance.\n")

    prompt_get_question_analysis+=prompt

    prompt_get_question_analysis +=(f"\nPlease review the following synthesized report:\n{sys_report}\n\n")

    prompt_get_question_analysis += ("Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.\n\nPlease present your reflection in the following format:\n\n'''\nAnswer: \n[Answer]\nReason: \n[Reason]\n'''")
    return prompt_get_question_analysis

def get_reflection_categories_prompt(question, sys_report,initial_analysis,one_hop_initial_analysis, random_walks_initial_analysis,centrality_initial_analysis,multi_domains_initial_analysis,
                                     GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis):
    prompt_get_question_analysis = f"Question:\n\n'''\n{question}\n'''\n\nBased on the Question, you have prepared a preliminary analysis:\n\n'''\n{initial_analysis}\n'''\n\n"

    expert_information = get_expert_introduction("One-Hop Neighbors Expert")
    prompt_get_question_analysis +=(f"One-hop Neighbors Expert:\n{expert_information}"+"\n- Analysis:\n{"+ one_hop_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Random Walks Expert")
    prompt_get_question_analysis +=(f"Random Walks Expert:\n{expert_information}"+"\n- Analysis:\n{"+ random_walks_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Centrality Expert")
    prompt_get_question_analysis +=(f"Centrality Expert:\n{expert_information}"+"\n{"+ centrality_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Multi-domains Expert")
    prompt_get_question_analysis +=(f"Multi-domains Expert:\n{expert_information}"+"\n{"+ multi_domains_initial_analysis+"}\n")

    prompt=""
    if len(GCN_Analysis) != 0:
        expert_information = get_expert_introduction("GCN")
        prompt_get_question_temp = (
                    f"GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(GAT_Analysis) != 0:
        expert_information = get_expert_introduction("GAT")
        prompt_get_question_temp = (
                    f"GAT Expert:\n{expert_information}" + "\n- Analysis:\n{" + GAT_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(TAPE_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("TAPE (GCN)")
        prompt_get_question_temp = (
                    f"TAPE (GCN) Expert:\n{expert_information}" + "\n- Analysis:\n{" + TAPE_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(R_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("R-GCN")
        prompt_get_question_temp = (
                    f"R-GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + R_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp


    prompt_get_question_analysis +="1. You will receive preliminary analysis from other experts and synthesized report. Critically review and analyze these insights.\n2. If you find aspects of other experts' analysis that are more rational than yours, incorporate these into your analysis for improvement.\n3. If you believe your analysis is more scientifically sound compared to others, maintain your stance.\n"

    prompt_get_question_analysis+=prompt

    prompt_get_question_analysis +=(f"\nPlease review the following synthesized report:\n{sys_report}\n\n")

    prompt_get_question_analysis += ("Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.\n\nPlease present your reflection in the following format:\n\n'''\nAnswer:\n[Answer]\nReason: \n[Reason]\n'''")
    return prompt_get_question_analysis

def get_reflection_multi_domains_prompt(question, sys_report,initial_analysis,one_hop_initial_analysis, random_walks_initial_analysis,centrality_initial_analysis,categories_initial_analysis,
                                        GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis):
    prompt_get_question_analysis = f"Question:\n\n'''\n{question}\n'''\n\nBased on the Question, you have prepared a preliminary analysis:\n\n'''\n{initial_analysis}\n'''\n\n"

    expert_information = get_expert_introduction("One-Hop Neighbors Expert")
    prompt_get_question_analysis +=(f"One-hop Neighbors Expert:\n{expert_information}"+"\n- Analysis:\n{"+ one_hop_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Random Walks Expert")
    prompt_get_question_analysis +=(f"Random Walks Expert:\n{expert_information}"+"\n- Analysis:\n{"+ random_walks_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Centrality Expert")
    prompt_get_question_analysis +=(f"Centrality Expert:\n{expert_information}"+"\n{"+ centrality_initial_analysis+"}\n")


    expert_information = get_expert_introduction("Categories Expert")
    prompt_get_question_analysis +=(f"Categories Expert:\n{expert_information}"+"\n{"+ categories_initial_analysis+"}\n")

    prompt=""
    if len(GCN_Analysis) != 0:
        expert_information = get_expert_introduction("GCN")
        prompt_get_question_temp = (
                    f"GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(GAT_Analysis) != 0:
        expert_information = get_expert_introduction("GAT")
        prompt_get_question_temp = (
                    f"GAT Expert:\n{expert_information}" + "\n- Analysis:\n{" + GAT_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(TAPE_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("TAPE (GCN)")
        prompt_get_question_temp = (
                    f"TAPE (GCN) Expert:\n{expert_information}" + "\n- Analysis:\n{" + TAPE_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp
    if len(R_GCN_Analysis) != 0:
        expert_information = get_expert_introduction("R-GCN")
        prompt_get_question_temp = (
                    f"R-GCN Expert:\n{expert_information}" + "\n- Analysis:\n{" + R_GCN_Analysis + "}\n\n")
        prompt += prompt_get_question_temp


    prompt_get_question_analysis +="1. You will receive preliminary analysis from other experts and synthesized report. Critically review and analyze these insights.\n2. If you find aspects of other experts' analysis that are more rational than yours, incorporate these into your analysis for improvement.\n3. If you believe your analysis is more scientifically sound compared to others, maintain your stance.\n"

    prompt_get_question_analysis+=prompt

    prompt_get_question_analysis +=(f"\nPlease review the following synthesized report:\n{sys_report}\n\n")

    prompt_get_question_analysis += ("Please provide three likely categories as a comma-separated list, arranged from most likely to least likely. For each category, explain your reasoning.\n\nPlease present your reflection in the following format:\n\n'''\nAnswer:\n[Answer]\nReason: \n[Reason]\n'''")
    return prompt_get_question_analysis

#--------------------------------------------------------------------------------------

def get_summary_assistant_initial_role():
    question_analyzer = ("You are a graph mining summary assistant who excels at synthesizing information from multiple expert reports.")
    return question_analyzer

def get_summary_assistant_initial_reports_prompt(compact,analysis_dict):
    expert_answer = ""
    expert_analyses = ""
    count = 0
    for key, value in analysis_dict.items():
        count += 1
        if "Reason" in value: split_str="Reason"
        elif "Reflection" in value: split_str = "Reflection"
        else: split_str = "\\n"
        cleaned_text = re.sub(r'\d+\.\s*', '', str(value).split(split_str)[0])
        cleaned_text = cleaned_text.replace("Answer:", "")
        cleaned_text = cleaned_text.replace("\n", "").strip()
        expert_answer += f"({count}) {key}: {cleaned_text}\n"
        expert_introduction=get_expert_introduction(key)
        expert_analyses += f"- {key}:\n{expert_introduction}\n{value}\n"
    # prompt = f"Question:\n{question}\n\n"
    prompt = f"The Compact Graph:\n{compact}\n\n"
    prompt += "Based on the following expert reports, please perform these tasks:\n\n"
    prompt += "1. Carefully examine all the expert analyses, considering the specific domain expertise of each expert.\n"
    prompt += "2. Identify and summarize the most significant insights from each expert’s report, focusing on their relevance to the central question.\n"
    prompt += "3. Combine the insights into a unified and concise analysis, ensuring that the central node is accurately categorized.\n"
    prompt += "4. List up to three critical and potentially contentious points raised by the experts, prioritized by their importance for further discussion.\n"
    prompt += "5. List up to three critical common points agreed upon by the experts, prioritized by their importance for further discussion.\n"
    prompt += "6. You will find (Score:n) at the end of each category. The maximum (best) score is 10, which means that this category is 100% correct (and 0% incorrect). The minimum (worst) score is 0, which means that this category is 100% incorrect (and 0% correct). Pay attention to the category Score and conduct the corresponding analysis.\n\n"
    prompt += "Please output your response in the following format:\n\n"
    prompt += f"'''\nKey Insights:\n[extracted key insights]\n\nSynthesized Analysis:\n[combined analysis]\n\nEach Expert Answer:\n{expert_answer}"
    prompt +="\nKey controversial points:\n(1) xxx\n(2) xxx\n(3) xxx\n"
    prompt +="\nKey common points:\n(1) xxx\n(2) xxx\n(3) xxx\n'''\n\n"
    prompt += f"Here are the expert analyses:\n\n{expert_analyses}"
    return prompt

#--------------------------------------------------------------------------------------

def get_decision_agent_role():
    question_analyzer = ("You are a graph mining decision-maker skilled in making informed decisions based on the question and the synthesized report.")
    return question_analyzer

def get_decision_agent_prompt(question,final_report):
    question_analyzer = (f"Question:\n{question}\n\nSynthesized Report:\n{final_report}\n"
                         f"\nBased on the report, select the best category to answer the question.\n\nConsiderations:\n")
    question_analyzer += "1. Use the candidate categories to inform your decision.\n"
    question_analyzer += "2. Exclude any categories with incorrect or misleading information.\n"
    question_analyzer +=("3. If multiple categories seem correct, analyze their distinctions based on the graph's structure and attributes, and choose the one that best aligns with the analysis.\n")
    question_analyzer += "\nProvide your final answer with brief reasoning.\n"
    question_analyzer += "Using the following format: \n'''\nAnswer: [Answer]\nReason: [Reason]\n'''"

    return question_analyzer
