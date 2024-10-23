import json
from openai import OpenAI
from tqdm import tqdm
from prompt_generator import *


def get_gpt_summary_prompt(compact_graph_description, One_Hop_Neighbors_Expert_Analysis, Random_Walks_Expert_Analysis,
                           Centrality_Expert_Analysis, Categories_Expert_Analysis, Multi_domains_Expert_Analysis):
    prompt = ""
    prompt += "Analyze the expert reports related to the target node graph. Your main goal is to evaluate each expert’s overall answer based on a holistic confidence score and provide reasoning, leading to a cohesive global summary.\n"
    prompt += "1. Confidence Analysis: Evaluate each expert’s entire analysis by assigning a single confidence score on a scale of 1 to 5: \n- 5: Strong confidence \n- 4: High confidence \n- 3: Moderate confidence \n- 2: Low confidence \n- 1: Poor confidence \n For each score, provide a clear and concise justification that reflects the expert's overall depth of knowledge, accuracy, and reliability in their analysis. Ensure that the score represents the expert’s report as a whole and not specific sections or categories.\n"
    prompt += "2. Extract Key Insights: Summarize significant insights relevant to the target node from each expert’s report.\n"
    prompt += "3. Global Summary: Create a unified summary that synthesizes the insights and highlights any critical agreements or controversies."
    prompt += " " + "Response Format:\n\n```Confidence Analyses:\n(1) Random Walks Expert: {Whole score and reasoning}\n(2) One-Hop Neighbors Expert: {Whole score and reasoning}\n(3) Centrality Expert: {Whole score and reasoning}\n(4) Categories Expert: {Whole score and reasoning}\n(5) Multi-domains Expert: {Whole score and reasoning}"

    prompt += "\n\nKey Insights:\n[Summarized insights from all experts]\n\nGlobal Summary:\n[A cohesive analysis combining insights and highlighting critical points]\n```\n\n"
    prompt += f"Compact Graph for the Target Node:\n{compact_graph_description}\n\n"
    prompt += f"Expert Reports:\n\n"
    if len(Random_Walks_Expert_Analysis)!=0:
        expert_information = get_expert_introduction("Random Walks Expert")
        prompt_get_question_analysis = (
                    f"Random Walks Expert:\n{expert_information}" + "\n- Analysis:\n{" + Random_Walks_Expert_Analysis + "}\n\n")
        prompt += prompt_get_question_analysis
    if len(One_Hop_Neighbors_Expert_Analysis) != 0:
        expert_information = get_expert_introduction("One-Hop Neighbors Expert")
        prompt_get_question_analysis = (
                    f"One-hop Neighbors Expert:\n{expert_information}" + "\n- Analysis:\n{" + One_Hop_Neighbors_Expert_Analysis + "}\n\n")
        prompt += prompt_get_question_analysis
    if len(Centrality_Expert_Analysis) != 0:
        expert_information = get_expert_introduction("Centrality Expert")
        prompt_get_question_analysis = (
                    f"Centrality Expert:\n{expert_information}" + "\n- Analysis:\n{" + Centrality_Expert_Analysis + "}\n\n")
        prompt += prompt_get_question_analysis
    if len(Categories_Expert_Analysis) != 0:
        expert_information = get_expert_introduction("Categories Expert")
        prompt_get_question_analysis = (
                    f"Categories Expert:\n{expert_information}" + "\n- Analysis:\n{" + Categories_Expert_Analysis + "}\n\n")
        prompt += prompt_get_question_analysis
    if len(Multi_domains_Expert_Analysis) != 0:
        expert_information = get_expert_introduction("Multi-domains Expert")
        prompt_get_question_analysis = (
                    f"Multi-domains Expert:\n{expert_information}" + "\n- Analysis:\n{" + Multi_domains_Expert_Analysis + "}\n\n")
        prompt += prompt_get_question_analysis
    return prompt


def main(file_name, cur_index,diss):
    with open(f"{file_name}.json", "r") as file:
        datas = json.load(file)
    datas = datas[cur_index:]
    result = []
    for item in tqdm(datas):
        if diss==0:
           key="initial"
        else:
            key=f"Discussion {diss}"
        One_Hop_Neighbors_Expert_Analysis = item[f"{key}: One-Hop Neighbors Expert Analysis"]
        Random_Walks_Expert_Analysis = item[f"{key}: Random Walks Expert Analysis"]
        Centrality_Expert_Analysis = item[f"{key}: Centrality Expert Analysis"]
        Categories_Expert_Analysis = item[f"{key}: Categories Expert Analysis"]
        Multi_domains_Expert_Analysis = item[f"{key}: Multi-domains Expert Analysis"]


        compact_graph_description = item["input"]

        prompt = get_gpt_summary_prompt(compact_graph_description, One_Hop_Neighbors_Expert_Analysis,
                                        Random_Walks_Expert_Analysis, Centrality_Expert_Analysis,
                                        Categories_Expert_Analysis, Multi_domains_Expert_Analysis)


        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o",
            # model="gpt-4",

            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        original_string = str(completion.choices[0].message)
        # print(prompt)
        start_index = original_string.find("content='") + len("content='")
        content_part = original_string[start_index:]
        item[f"{key}: gpt_4o_syn_report"] = content_part
        result.append(item)
        with open(f"{file_name}_gpt4o.json", "a") as f:
            json.dump(result[:], f, indent=2)
        result = []


if __name__ == "__main__":
    cur_index = 0
    diss=1
    for file_name in ['test_nc_TPAEGCN_GAT_arxiv_initial_gpt4o_1diss_complete',]:
        main(file_name, cur_index, diss)
