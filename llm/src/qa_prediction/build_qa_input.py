import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable

import json
with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}

import re 
import string
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

class PromptBuilder(object):
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    #SAQ_RULE_INSTRUCTION = """Based on the provided knowledge, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    #SAQ_RULE_INSTRUCTION = """Your tasks is to use the following facts and answer the question. Make sure that you use the information from the facts provided. Please keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    #GRAPH_CONTEXT = """The facts are the following:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""
    def __init__(self, prompt_path, encrypt=False, add_rule = False, use_true = False, cot = False, explain = False, use_random = False, each_line = False, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line

        self.encrypt=encrypt
        
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        
        entities = question_dict['q_entity']
        skip_ents = []
        
        graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)

        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        
        if not question.endswith('?'):
            question += '?'
        
        lists_of_paths = []
        if self.add_rule:
            entities = question_dict['q_entity']
            #graph = utils.build_graph(question_dict['graph'], entities, self.encrypt)
            skip_ents = []
            
            graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                
                # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
            else:
                lists_of_paths = []
            #input += self.GRAPH_CONTEXT.format(context = context)
        #lists_of_paths = []
        if question_dict['cand'] is not None:
            if not self.add_rule:
                skip_ents = []
                graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)
            lists_of_paths2 = []
            #print(question_dict['cand'])
            reasoning_paths = utils.get_truth_paths(question_dict['q_entity'], question_dict['cand'], graph)
            for p in reasoning_paths:
                if utils.path_to_string(p) not in lists_of_paths:
                    lists_of_paths.append(utils.path_to_string(p))
            
            for p in reasoning_paths:
                if utils.path_to_string(p) not in lists_of_paths2:
                    lists_of_paths2.append(utils.path_to_string(p))
           
        input = self.QUESTION.format(question = question)
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule or question_dict['cand'] is not None:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_rule or question_dict['cand'] is not None:
                instruction = self.SAQ_RULE_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule or question_dict['cand'] is not None:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            
            input = self.GRAPH_CONTEXT.format(context = context) + input
        
        input = self.prompt_template.format(instruction = instruction, input = input)
            
        return input
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)
            