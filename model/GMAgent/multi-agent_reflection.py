import json
from tqdm import tqdm
import re
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from prompt_generator import *


config_musegraph_qwen2 = {
    'model': 'agent_Qwen2-7B-Instruct',
    "base_url": "http://localhost:8001/v1/",
    "api_key": "NULL",
}


initializer = UserProxyAgent(
    name="Init",
    code_execution_config=False,
    human_input_mode="NEVER"
)


def collaborative_discussion(data,initial_analysis_dict,sys_report,GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis):
    new_analysis_dict = {"Random Walks Expert Analysis": "",
                         "One-Hop Neighbors Expert Analysis": "",
                         "Centrality Expert Analysis": "",
                         "Categories Expert Analysis": "",
                         "Multi-domains Expert Analysis": ""}
    question = data["instruction"] + "\n" + data["input"]
    random_walk_expert = AssistantAgent(
        name="Random Walks Expert",
        system_message=get_reflection_random_walk_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    one_hop_expert = AssistantAgent(
        name="One-Hop Neighbors Expert",
        system_message=get_reflection_one_hop_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    centrality_expert = AssistantAgent(
        name="Centrality Expert",
        system_message=get_reflection_centrality_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    categories_expert = AssistantAgent(
        name="Categories Expert",
        system_message=get_reflection_categories_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    multi_domains_expert = AssistantAgent(
        name="Multi_domains_Expert",
        system_message=get_reflection_multi_domains_role(),
        llm_config=config_musegraph_qwen2,
        human_input_mode="NEVER",
    )
    analysis_dict=initial_analysis_dict
    for key, value in analysis_dict.items():
        if "One-Hop Neighbors Expert Analysis" in key:
            random_walks_initial_analysis = analysis_dict["Random Walks Expert Analysis"]
            centrality_initial_analysis = analysis_dict["Centrality Expert Analysis"]
            categories_initial_analysis = analysis_dict["Categories Expert Analysis"]
            multi_domains_initial_analysis = analysis_dict["Multi-domains Expert Analysis"]
            analysis_chat = initializer.initiate_chat(
                recipient=one_hop_expert,
                message=get_reflection_one_hop_prompt(question, sys_report, value,
                                                      random_walks_initial_analysis,
                                                      centrality_initial_analysis,
                                                      categories_initial_analysis,
                                                      multi_domains_initial_analysis,
                                                      GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis),
                max_turns=1,
            )
        elif "Random Walks Expert Analysis" in key:
            one_hop_initial_analysis = analysis_dict["One-Hop Neighbors Expert Analysis"]
            centrality_initial_analysis = analysis_dict["Centrality Expert Analysis"]
            categories_initial_analysis = analysis_dict["Categories Expert Analysis"]
            multi_domains_initial_analysis = analysis_dict["Multi-domains Expert Analysis"]
            analysis_chat = initializer.initiate_chat(
                recipient=random_walk_expert,
                message=get_reflection_random_walk_prompt(question, sys_report, value,
                                                          one_hop_initial_analysis,
                                                          centrality_initial_analysis,
                                                          categories_initial_analysis,
                                                          multi_domains_initial_analysis,
                                                          GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis),
                max_turns=1,
            )
        elif "Centrality Expert Analysis" in key:
            one_hop_initial_analysis = analysis_dict["One-Hop Neighbors Expert Analysis"]
            random_walks_initial_analysis = analysis_dict["Random Walks Expert Analysis"]
            categories_initial_analysis = analysis_dict["Categories Expert Analysis"]
            multi_domains_initial_analysis = analysis_dict["Multi-domains Expert Analysis"]
            analysis_chat = initializer.initiate_chat(
                recipient=centrality_expert,
                message=get_reflection_centrality_prompt(question, sys_report, value,
                                                         one_hop_initial_analysis,
                                                         random_walks_initial_analysis,
                                                         categories_initial_analysis,
                                                         multi_domains_initial_analysis,
                                                         GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis),
                max_turns=1,
            )
        elif "Categories Expert Analysis" in key:
            one_hop_initial_analysis = analysis_dict["One-Hop Neighbors Expert Analysis"]
            random_walks_initial_analysis = analysis_dict["Random Walks Expert Analysis"]
            centrality_initial_analysis = analysis_dict["Centrality Expert Analysis"]
            multi_domains_initial_analysis = analysis_dict["Multi-domains Expert Analysis"]
            analysis_chat = initializer.initiate_chat(
                recipient=categories_expert,
                message=get_reflection_categories_prompt(question, sys_report, value,
                                                         one_hop_initial_analysis,
                                                         random_walks_initial_analysis,
                                                         centrality_initial_analysis,
                                                         multi_domains_initial_analysis,
                                                         GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis),
                max_turns=1,
            )
        elif "Multi-domains Expert Analysis" in key:
            one_hop_initial_analysis = analysis_dict["One-Hop Neighbors Expert Analysis"]
            random_walks_initial_analysis = analysis_dict["Random Walks Expert Analysis"]
            centrality_initial_analysis = analysis_dict["Centrality Expert Analysis"]
            categories_initial_analysis = analysis_dict["Categories Expert Analysis"]
            analysis_chat = initializer.initiate_chat(
                recipient=multi_domains_expert,
                message=get_reflection_multi_domains_prompt(question, sys_report, value,
                                                            one_hop_initial_analysis,
                                                            random_walks_initial_analysis,
                                                            centrality_initial_analysis,
                                                            categories_initial_analysis,
                                                            GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis),
                max_turns=1,
            )
        new_analysis_dict[key] = analysis_chat.chat_history[1]['content']
    return new_analysis_dict



# 主函数
def main(file_name,cout):
    with open(f"{file_name}.json", "r") as file:
        datas = json.load(file)
    result = []
    complete = []
    check = []
    for data in tqdm(datas):
        if 'Arxiv' in data['instruction']:
            options_list = [
                "cs.na", "cs.mm", "cs.lo", "cs.cy", "cs.cr", "cs.dc", "cs.hc", "cs.ce", "cs.ni", "cs.cc",
                "cs.ai", "cs.ma", "cs.gl", "cs.ne", "cs.sc", "cs.ar", "cs.cv", "cs.gr", "cs.et", "cs.sy",
                "cs.cg", "cs.oh", "cs.pl", "cs.se", "cs.lg", "cs.sd", "cs.si", "cs.ro", "cs.it", "cs.pf",
                "cs.cl", "cs.ir", "cs.ms", "cs.fl", "cs.ds", "cs.os", "cs.gt", "cs.db", "cs.dl", "cs.dm"
            ]
            GCN_Analysis = ""
            GAT_Analysis = data["initial: GNN Agent Answer (GAT)"]
            TAPE_GCN_Analysis = data["initial: GNN Agent Answer (TAPE_GCN)"]
            R_GCN_Analysis = ""
        elif "Cora" in data['instruction']:
            options_list = ['theory','reinforcement learning','genetic algorithms','neural networks','probabilistic methods','case based','rule learning']
            GCN_Analysis = data["initial: GNN Agent Answer (GCN)"]
            GAT_Analysis = data["initial: GNN Agent Answer (GAT)"]
            TAPE_GCN_Analysis = ""
            R_GCN_Analysis = ""
        if cout==0:
            sys_report=data["initial: gpt_4o_syn_report"]
        else:
            sys_report = data[f"Discussion {cout}: gpt_4o_syn_report"]
        initial_analysis_dict = {}
        if cout==0:
            for key in ["initial: One-Hop Neighbors Expert Analysis","initial: Random Walks Expert Analysis",
                        "initial: Centrality Expert Analysis","initial: Categories Expert Analysis",f"initial: Multi-domains Expert Analysis"]:
                initial_analysis_dict[key.replace("initial: ","")]=data[key]
            initial_entropy = calculate_answers_entropy(initial_analysis_dict, options_list)
        else:
            for key in [f"Discussion {cout}: One-Hop Neighbors Expert Analysis",f"Discussion {cout}: Random Walks Expert Analysis",
                        f"Discussion {cout}: Centrality Expert Analysis",f"Discussion {cout}: Categories Expert Analysis",
                        f"Discussion {cout}: Multi-domains Expert Analysis"]:
                initial_analysis_dict[key.replace(f"Discussion {cout}: ","")]=data[key]
            initial_entropy = calculate_answers_entropy(initial_analysis_dict, options_list)
        temp = data
        # cout=1
        # before_entropy=initial_entropy
        # update_analysis_dict=initial_analysis_dict
        # while(True):
        #     update_analysis_dict = collaborative_discussion(data,update_analysis_dict,sys_report)
        #     current_entropy = calculate_answers_entropy(update_analysis_dict,options_list)
        #
        #     # Recording
        #     for key, value in update_analysis_dict.items():
        #         temp["Discussion "+str(cout)+": "+key] = value
        #     temp["Discussion "+str(cout)+": "+"current entropy"] = current_entropy
        #     cout += 1
        #     if before_entropy > current_entropy:
        #         before_entropy = current_entropy
        #     else:
        #         break

        update_analysis_dict = initial_analysis_dict
        update_analysis_dict = collaborative_discussion(data,update_analysis_dict,sys_report,
                                                        GCN_Analysis, GAT_Analysis, TAPE_GCN_Analysis, R_GCN_Analysis)
        current_entropy = calculate_answers_entropy(update_analysis_dict,options_list)
        for key, value in update_analysis_dict.items():
            temp["Discussion "+str(cout+1)+": "+key] = value
        temp["Discussion "+str(cout+1)+": "+"current entropy"] = current_entropy
        before_entropy = initial_entropy
        if before_entropy > current_entropy:
            check.append(temp)
        else:
            complete.append(temp)

        # Recording
        with open(f"saves/agent/{file_name}_{cout+1}diss_complete.json", "w") as f:
            json.dump(complete[:], f, indent=2)

        with open(f"saves/agent/{file_name}_{cout+1}diss_check.json", "w") as f:
            json.dump(check[:], f, indent=2)

# 入口点
if __name__ == "__main__":
    cout=0
    for file_name in ["test_nc_TPAEGCN_GAT_arxiv_initial_1009_gpt4o"]:
        main(file_name,cout)
