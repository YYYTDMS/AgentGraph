import json
from tqdm import tqdm
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from prompt_generator import *


config_musegraph_qwen2 = {
    'model': 'agent_Qwen2-7B-Instruct',
    "base_url": "http://localhost:8000/v1/",
    "api_key": "NULL",
}




initializer = UserProxyAgent(
    name="Init",
    code_execution_config=False,
    human_input_mode="NEVER"
)


def problem_analysis(data):
    one_hop_expert = AssistantAgent(
        name="One-Hop_Neighbors_Expert",
        system_message=get_question_analysis_one_hop_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    random_walk_expert = AssistantAgent(
        name="Random_Walks_Expert",
        system_message=get_question_analysis_random_walk_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    centrality_expert = AssistantAgent(
        name="Centrality_Expert",
        system_message=get_question_analysis_centrality_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    categories_expert = AssistantAgent(
        name="Categories_Expert",
        system_message=get_question_analysis_categories_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    multi_domains_expert = AssistantAgent(
        name="Multi_domains_Expert",
        system_message=get_question_analysis_multi_domains_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    one_hop_prompt = get_question_analysis_one_hop_prompt(data["instruction"] + "\n" + data["input"])
    random_walk_prompt = get_question_analysis_random_walk_prompt(data["instruction"] + "\n" + data["input"])
    centrality_prompt = get_question_analysis_centrality_prompt(data["instruction"] + "\n" + data["input"])
    categories_prompt = get_question_analysis_categories_prompt(data["instruction"] + "\n" + data["input"])
    multi_domains_prompt = get_question_analysis_multi_domains_prompt(data["instruction"] + "\n" + data["input"])
    problem_analysis_initializer = UserProxyAgent(
        name="Problem_Analysis_Init",
        code_execution_config=False,
        human_input_mode="NEVER",
        # default_auto_reply=get_reflection_analysis(),
    )
    one_hop_analysis = problem_analysis_initializer.initiate_chat(
        recipient=one_hop_expert,
        message=one_hop_prompt,
        max_turns=1,
    )
    random_walk_analysis = problem_analysis_initializer.initiate_chat(
        recipient=random_walk_expert,
        message=random_walk_prompt,
        max_turns=1,
    )
    centrality_analysis = problem_analysis_initializer.initiate_chat(
        recipient=centrality_expert,
        message=centrality_prompt,
        max_turns=1,
    )
    categories_analysis = problem_analysis_initializer.initiate_chat(
        recipient=categories_expert,
        message=categories_prompt,
        max_turns=1,
    )
    multi_domains_analysis = problem_analysis_initializer.initiate_chat(
        recipient=multi_domains_expert,
        message=multi_domains_prompt,
        max_turns=1,
    )
    one_hop_expert_analysis = one_hop_analysis.chat_history[1]['content']
    random_walk_expert_analysis = random_walk_analysis.chat_history[1]['content']
    centrality_expert_analysis = centrality_analysis.chat_history[1]['content']
    categories_expert_analysis = categories_analysis.chat_history[1]['content']
    multi_domains_expert_analysis = multi_domains_analysis.chat_history[1]['content']
    analysis_dict = {"initial: One-Hop Neighbors Expert Analysis": one_hop_expert_analysis,
                     "initial: Random Walks Expert Analysis": random_walk_expert_analysis,
                     "initial: Centrality Expert Analysis": centrality_expert_analysis,
                     "initial: Categories Expert Analysis": categories_expert_analysis,
                     "initial: Multi-domains Expert Analysis": multi_domains_expert_analysis,}
    return analysis_dict

def main(file_name):
    with open(f"data/{file_name}.json", "r") as file:
        datas = json.load(file)
    result = []
    for data in tqdm(datas):
        temp = data
        if 'Arxiv' in data['instruction']:
            options_list = [
                "cs.na", "cs.mm", "cs.lo", "cs.cy", "cs.cr", "cs.dc", "cs.hc", "cs.ce", "cs.ni", "cs.cc",
                "cs.ai", "cs.ma", "cs.gl", "cs.ne", "cs.sc", "cs.ar", "cs.cv", "cs.gr", "cs.et", "cs.sy",
                "cs.cg", "cs.oh", "cs.pl", "cs.se", "cs.lg", "cs.sd", "cs.si", "cs.ro", "cs.it", "cs.pf",
                "cs.cl", "cs.ir", "cs.ms", "cs.fl", "cs.ds", "cs.os", "cs.gt", "cs.db", "cs.dl", "cs.dm"
            ]
        elif "Cora" in data['instruction']:
            options_list = ['theory','reinforcement learning','genetic algorithms','neural networks',
                            'probabilistic methods','case based','rule learning']

        analysis_dict = problem_analysis(data)
        for key, value in analysis_dict.items(): temp[key] = value
        initial_analysis_dict = {}
        for key in ["initial: One-Hop Neighbors Expert Analysis",
                    "initial: Random Walks Expert Analysis",
                    "initial: Centrality Expert Analysis",
                    "initial: Categories Expert Analysis",
                    "initial: Multi-domains Expert Analysis"]:
            initial_analysis_dict[key.replace("initial: ", "")] = data[key]
            initial_entropy = calculate_answers_entropy(initial_analysis_dict, options_list)
            temp["initial_entropy"] = initial_entropy

        result.append(temp)

        with open(f"{file_name}_initial_1009.json", "w") as f:
            json.dump(result[:], f, indent=2)


# 入口点
if __name__ == "__main__":
    for file_name in ["test_nc_TPAEGCN_GAT_arxiv"]:
        main(file_name)
