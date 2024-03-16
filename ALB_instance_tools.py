import numpy as np
import re
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import glob
import copy
import yaml
import warnings
import csv

class MixedModelInstance:
    def __init__(self, model_dicts=None, model_yaml=None, init_type= "model_dicts", name = None, cycle_time = None):
        self.model_dicts = model_dicts
        self.name = name
        self.cycle_time = cycle_time
        self.no_models = None
        self.all_tasks = None
        #MixedModelInstance stats
        self.no_tasks = None
        self.flexibility_measures = None
        if init_type == 'model_dicts':
            self.data = create_instance_pair_stochastic(model_dicts)
            self.format_data()
            self.no_models = len(model_dicts)
            
        elif init_type == 'yaml':
            #Reads config data from a yaml then model data from the model file 
            self.model_data_from_yaml(model_yaml)
            self.config_check()
        else:
            raise ValueError('init_type must be either model_dicts or yaml ')
        self.model_mixtures = self.get_model_mixtures()
        
        if not self.name:
            self.name = self.generate_name()
        
    def get_model_mixtures(self):
        model_mixtures = {}
        for model in self.data:
            model_mixtures[model] = self.data[model]['probability']
        return model_mixtures

    
    def generate_name(self):
        '''generates name from the filename of the .albp problem. The name is from the two numbers before the .albp extension'''
        name = ''
        for model in self.model_dicts:
            name += model['fp'].split('/')[-1].split('.')[0].split('_')[-2]+ '_'+ model['fp'].split('/')[-1].split('.')[0].split('_')[-1] + '_'
        #remove last character of name string
        name = name[:-1]
        return name
    
    def model_data_to_yaml(self, fp=None, name=None):
        '''This function writes a MultiModelInstance object to a yaml file'''
        mmi_dict = {'name':self.name, 
                    'model_data':self.data, 
                    'cycle_time':self.cycle_time, 
                    'no_tasks' : self.no_tasks, 
                    'flexibility_measures':self.flexibility_measures,
                    'similarity_measure':self.similarity_measure}
        if not name:
            name = self.name
        my_yaml = open(fp + f'{name}.yaml', 'w')
        yaml.dump(mmi_dict, my_yaml)

    def model_data_from_yaml(self, model_yaml_file):
        '''This function reads a model_yaml_file and uses it to fill in the model 
        data for the MultiModelInstance object'''
        print('using this file', model_yaml_file)
        with open(model_yaml_file) as file:
            mm_yaml = yaml.load(file, Loader=yaml.FullLoader)
            self.cycle_time = mm_yaml['cycle_time']
            self.data = mm_yaml['model_data']
            self.no_models = len(mm_yaml['model_data'])
            self.no_tasks = mm_yaml['no_tasks']
            self.name = mm_yaml['name']
            self.all_tasks = get_task_union(self.data, *self.data.keys())

    def calculate_flexibility_measure(self):
        '''Calculates the flexibility measure for a mixed model instance'''
        #for each model in the mixed model instance, calculates the flexibility measure
        #The flexibility measure is number of edges in the precedence graph divided by the total number of possible edges
        self.flexibility_measures = {}
        for model in self.data:
            print(model)
            print(self.data[model]['precedence_relations'])
            print('num_of tasks', self.data[model]['num_tasks'])
            self.flexibility_measures[model] = 1 - len(self.data[model]['precedence_relations'])/(self.data[model]['num_tasks']*(self.data[model]['num_tasks']-1)/2)

        #calculates the average flexibility measure for the mixed model instance
        self.flexibility_measures['average']= sum(self.flexibility_measures.values())/self.no_models

    def calculate_similarity_measure(self):
        '''Calculates the similarity between models in a mixed model instance'''
        #for each model in the mixed model instance, calculates the similarity measure with all other models
        #The similarity measure is the difference in task times between two models divided by the cycle time
        model_keys = list(self.data.keys())
        diff_measure = 0
        for i in range(len(model_keys)-1):
            for j in range(i+1, len(model_keys)):
                for task in self.all_tasks:
                    #if task is not in model i, then its time will be zero
                    if task not in self.data[model_keys[i]]['task_times'][1]:
                        task_i_time = 0
                    else:
                        task_i_time = self.data[model_keys[i]]['task_times'][1][task]
                    #if task is not in model j, then its time will be zero
                    if task not in self.data[model_keys[j]]['task_times'][1]:
                        task_j_time = 0
                    else:
                        task_j_time = self.data[model_keys[j]]['task_times'][1][task]
                    diff_measure += abs(task_i_time - task_j_time)/self.cycle_time
        diff_measure = 2 * diff_measure/ (self.no_tasks * self.no_models * (self.no_models - 1))
        self.similarity_measure = 1- diff_measure



    def calculate_stats(self):
        self.all_tasks = get_task_union(self.data, *self.data.keys())
        print('the data', self.data)
        print('all tasks', self.all_tasks)
        print('len of all tasks', len(self.all_tasks))
        self.no_tasks = len(self.all_tasks)
        print('no tasks', self.no_tasks)
        self.calculate_flexibility_measure()
        self.calculate_similarity_measure()

    def format_data(self):
        '''Formats the data so that it is in the correct format for the MultiModelInstance class'''
        self.all_tasks = set()
        for model in self.data:
                task_times = self.data[model]['task_times'].copy()
                self.all_tasks.update(task_times.keys())
                self.data[model]['task_times'] = {}
                self.data[model]['task_times'][1] = task_times
        self.no_tasks = len(self.all_tasks)

    def config_check(self):
        #Check that the total probability of entering is equal to 1
        total_probability = sum_prob(self.data)
        if total_probability != 1:
            warnings.warn('Model probabilities do not sum to 1: ', total_probability)





def sum_prob(sequences):
    '''function for sanity checking that the probabilities sum to 1'''
    total = 0
    for seq in sequences:
        total += sequences[seq]['probability']
    return total



def pair_instances(instance_list, MODEL_MIXTURES):
      '''returns a list of lists of multi-model instances, where each list of instances is a list of instances that will be run together'''
      instance_groups = []
      for i in range(len(instance_list)-len(MODEL_MIXTURES)+1):
         instance_group= []
         for j in range(i, i+ len(MODEL_MIXTURES)):
            model_name = list(MODEL_MIXTURES.keys())[j-i]
            instance_group.append({'fp':instance_list[j], 'name':model_name, 'probability':MODEL_MIXTURES[model_name]})
         instance_groups.append(instance_group)
      return instance_groups

def make_instance_pair(instance_list, MODEL_MIXTURES):
    '''returns a list of lists of multi-model instances, where each list of instances is a list of instances that will be run together'''
    instance_group= []
    for j in range(len(MODEL_MIXTURES)):
        model_name = list(MODEL_MIXTURES.keys())[j]
        instance_group.append({'fp':instance_list[j], 'name':model_name, 'probability':MODEL_MIXTURES[model_name]})
    return instance_group

def read_instance_folder(folder_loc):
   '''looks in folder_loc for all .alb files and returns a list of filepaths to the .alb files'''
   instance_list = []
   for file in glob.glob(f"{folder_loc}*.alb"):
      instance_list.append(file)
   instance_list.sort(key = lambda file: int(file.split("_")[-1].split(".")[0]))
   return instance_list

def parse_alb(alb_file_name):
    """Reads assembly line balancing instance .alb file, returns dictionary with the information"""
    parse_dict = {}
    alb_file = open(alb_file_name).read()
    # Get number of tasks
    num_tasks = re.search("<number of tasks>\n(\d*)", alb_file)
    parse_dict["num_tasks"] = int(num_tasks.group(1))

    # Get cycle time
    cycle_time = re.search("<cycle time>\n(\d*)", alb_file)
    parse_dict["cycle_time"] = int(cycle_time.group(1))

    # Order Strength
    order_strength = re.search("<order strength>\n(\d*,\d*)", alb_file)
    
    if order_strength:
        parse_dict["order_strength"] = float(order_strength.group(1).replace(",", "."))
    else:
        order_strength = re.search("<order strength>\n(\d*.\d*)", alb_file)
        parse_dict["order_strength"] = float(order_strength.group(1))

    # Task_times
    task_times = re.search("<task times>(.|\n)+?<", alb_file)

    # Get lines in this regex ignoring the first and last 2
    task_times = task_times.group(0).split("\n")[1:-2]
    task_times = {task.split()[0]: int(task.split()[1]) for task in task_times}
    parse_dict["task_times"] = task_times

    # Precedence relations
    precedence_relations = re.search("<precedence relations>(.|\n)+?<", alb_file)
    precedence_relations = precedence_relations.group(0).split("\n")[1:-2]
    precedence_relations = [task.split(",") for task in precedence_relations]
    parse_dict["precedence_relations"] = precedence_relations
    return parse_dict


#function that returns names of all files in a directory with a given extension
def get_instance_list(directory, keep_directory_location = True,  extension='.alb'):
    if keep_directory_location:
        return [ directory + '/' + f for f in os.listdir(directory) if f.endswith(extension)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(extension)]

def rand_pert_precedence(p_graph_orig, seed=None):
    # randomly change at least 1 edge in the precedence graph
    # Seed random number generators
    while True:
        p_graph = p_graph_orig.copy()
        random.seed(seed)
        rng = np.random.default_rng(seed=seed)
        # calculate number of edges to change
        num_edges = 1 + rng.poisson(lam=4)
        # nx.swap.directed_edge_swap( p_graph, nswap=num_edges, seed=seed)
        edges_to_remove = random.sample(list(p_graph.edges()), num_edges)
        edges_to_add = random.sample(list(nx.non_edges(p_graph)), num_edges)
        for index, edge in enumerate(edges_to_remove):
            p_graph.remove_edge(edge[0], edge[1])
            p_graph.add_edge(edges_to_add[index][0], edges_to_add[index][1])
        pos = nx.spring_layout(p_graph_orig, k=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # nx.draw(p_graph_orig, ax=ax1, pos= pos, with_labels=True)
        # nx.draw(p_graph, ax=ax2, pos=pos, with_labels=True)
        simple_cycles = list(nx.simple_cycles(p_graph))
        if not simple_cycles:
            return list(p_graph.edges())

def eliminate_tasks_different_root(instance, elim_dict, seed=None):
    '''eliminates tasks from different models in the same mixed model instance. They may have different tasks eliminated from the same original precedence graph.'''
    rng = np.random.default_rng(seed=seed)
    for idx, model in enumerate(instance.data):
        for variant in instance.data[model]["task_times"]:
            to_remove = rng.choice(
                list(instance.data[model]["task_times"][1].keys()),
                size= elim_dict[model],
                replace=False,
            )
            entries_to_remove(to_remove, instance.data[model]["task_times"][variant])
            instance.data[model]["precedence_relations"] = reconstruct_precedence_constraints(
                instance.data[model]["precedence_relations"], to_remove
            )
            instance.data[model]["num_tasks"] = len(instance.data[model]["task_times"][1])

    print("instance data", instance.data)
    return instance

def eliminate_tasks_shared_root(instance, elim_dict, seed=None):
    '''eliminates tasks from different models in the same mixed model instance. 
    We assume there is a "base model" and all other models just have extra
    tasks added to it'''
    rng = np.random.default_rng(seed=seed)
    #sorts the elim dict by the number of tasks to eliminate
    elim_dict = dict(sorted(elim_dict.items(), key=lambda item: item[1]))
    print("elim dict", elim_dict)
    #gets the first model from instance.data
    model = list(instance.data.keys())[0]
    remaining_tasks  = list(instance.data[model]["task_times"][1].keys())
    print("remaining tasks", remaining_tasks)
    tasks_to_remove = []
    for idx, model in enumerate(instance.data):
        to_remove = rng.choice(
            remaining_tasks,
            size=(int(elim_dict[model])),
            replace=False,
        )
        tasks_to_remove += list(to_remove)
        print("tasks to remove", tasks_to_remove)
        #subtracts from each entry of the elim dict the number of the tasks to remove
        elim_dict = {key: value - elim_dict[model] for key, value in elim_dict.items()}
        print("elim dict", elim_dict)
        #removes the to_remove tasks from the remaining tasks
        remaining_tasks = [task for task in remaining_tasks if task not in to_remove]
        #deletes the task from the dictionary
        entries_to_remove(tasks_to_remove, instance.data[model]["task_times"][1])
        instance.data[model]["precedence_relations"] = reconstruct_precedence_constraints(
            instance.data[model]["precedence_relations"], tasks_to_remove
        )
        instance.data[model]["num_tasks"] = len(instance.data[model]["task_times"][1])
    return instance

                                                


# this function actually removes tasks from the precedence graph and task times
def eliminate_tasks(old_instance, elim_interval=(0.6, 0.8), seed=None):
    instance = copy.deepcopy(old_instance)
    rng = np.random.default_rng(seed=seed)
    interval_choice = rng.uniform(low=elim_interval[0], high=elim_interval[1])
    first_key = list(instance.data.keys())[0]
    # Instance must have same number of tasks and task numbering
    to_remove = rng.choice(
        list(instance.data[first_key]["task_times"][1].keys()),
        size=(int(instance.data[first_key]["num_tasks"] * (interval_choice))),
        replace=False,
    )
    # nx.draw_planar(p_graph, with_labels=True)
    for model in instance.data:
        for worker in instance.data[model]["task_times"]:
            # Remove node from task times list
            #print(instance.data[model]["task_times"])
            remaining_entries = entries_to_remove(
                to_remove, 
                instance.data[model]["task_times"][worker]
            )  
            # change precedence graph
            instance.data[model]["precedence_relations"] = reconstruct_precedence_constraints(
                instance.data[model]["precedence_relations"], 
                to_remove
            )  
            #update number of tasks
            instance.data[model]["num_tasks"] = len(
                instance.data[model]["task_times"][worker]
            )  
            
    return instance


def reconstruct_precedence_constraints(precedence_relations, to_remove):
    """Removes given tasks from precedence constraint, relinks preceding and succeeding tasks in the precedence  contraints"""
    p_graph = nx.DiGraph()
    p_graph.add_edges_from(precedence_relations)
    for node in to_remove:
        if node in p_graph.nodes():
            for parent in p_graph.predecessors(node):
                for child in p_graph.successors(node):
                    p_graph.add_edge(parent, child)
            p_graph.remove_node(node)
    return [list(edge) for edge in p_graph.edges()]


def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]
    return the_dict


def change_task_times(instance, perc_reduct_interval=(0.40, 0.60), seed=None):
    # this function creates new task times based on the original task times takes original task times and how much they need to be reduced
    new_task_times = instance["task_times"]
    print("old task times", new_task_times)
    rng = np.random.default_rng(seed=seed)
    for key in new_task_times:
        new_task_times[key] = int(
            new_task_times[key]
            * rng.uniform(low=perc_reduct_interval[0], high=perc_reduct_interval[1])
        )
    print(new_task_times)
    return new_task_times

def get_task_intersection(test_instance, model_1, model_2):
    '''Returns the intersection of tasks between two models'''
    return  set(test_instance[model_1]['task_times']).intersection(set(test_instance[model_2]['task_times']))

def get_task_union(test_instance, *args):
    '''Returns the union of tasks between all models, input is a series of models to check'''
    for index, model in enumerate(args):
        if index == 0:
            task_union = set(test_instance[model]['task_times'][1])
        else:
            task_union = task_union.union(set(test_instance[model]['task_times'][1]))
    return  task_union
    
def construct_precedence_matrix(instance):
    '''constructs a precedence matrix representation of a model's precedence relations'''
    precedence_matrix = np.zeros((len(instance['task_times'].keys()), len(instance['task_times'].keys())))
    for precedence in instance['precedence_relations']:
        precedence_matrix[int(precedence[0]) - 1][int(precedence[1]) - 1] = 1
    return precedence_matrix

def create_instance_pair_stochastic(instance_dicts):
    '''read .alb files, create a dictionary for each model, and include model name and probability
     input: list of dictionaries with keys 'name' 'location' and probability '''
    parsed_instances = {}
    for instance in instance_dicts:
        parsed_instances[instance['name']] = {}
        parsed_instance = parse_alb(instance['fp'])
        #Cycle time will be set to the same for all models
        del parsed_instance['cycle_time']
        parsed_instances[instance['name']].update(parsed_instance)
        parsed_instances[instance['name']]['probability'] = instance['probability']
    return parsed_instances


def list_all_tasks(instance):
    """Generates the set O of all tasks from a list of models"""
    tasks = []
    for index, model in enumerate(instance):
        tasks += model["task_times"].keys()
    return list(set(tasks))


def linear_reduction(old_task_times, number_of_workers):
    """Divides time of task by number of workers.
    INPUT: task times dictionary, number_of_workers int
    OUTPUT: new task_times dictonary with reduced times
    """
    if number_of_workers == 0:
        return old_task_times
    task_times = old_task_times.copy()
    for key, values in task_times.items():
        task_times[key] = values / number_of_workers
    return task_times



def dict_list_from_csv(file_name):
    '''reads a list of dictionaries from a csv file'''
    with open(file_name, newline='') as input_file:
        reader = csv.DictReader(input_file)
        dict_list = []
        for row in reader:
            dict_list.append(row)
    return dict_list
