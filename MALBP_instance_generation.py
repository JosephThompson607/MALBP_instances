import sys  
sys.path.insert(1, '..')
from ALB_instance_tools import *
import random

def random_model_mixture(model_names, seed = None):
    '''Creates a dictionary where the model name are the keys and the probability is the values. The probabilities sum to 1.
        parameters: model_names: a list of model names'''
    model_mixture = {}
    random.seed(seed)
    for model_name in model_names:
        model_mixture[model_name] = random.random()
    total = sum(model_mixture.values())
    for model_name in model_mixture.keys():
        model_mixture[model_name] = model_mixture[model_name]/total
    return model_mixture

def make_reduced_instances(filepath, 
                           SALBP_instance_list, 
                           model_names, 
                           cycle_time, 
                           interval_choice = (0.7,0.8), 
                           seed = None,
                           parent_set = "Otto"):
    '''Deletes the same tasks from two different SALBP models in the same mixed model instance.
        parameters: filepath: the path to the file where the mixed model instances will be written
                    SALBP_instance_list: a list of SALBP_instances
                    model_names: a list of model names
                    cycle_time: the cycle time of the mixed model instance
                    interval_choice: a tuple of two floats that represent the interval of the number of tasks to delete. (0.3, 0.4) means delete 30%-40% of the tasks
                    seed: a seed for the random number generator
                    parent_set: a string that will be added to the name of the mixed model instances, to differentiate them from other instance sets'''
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        instance_name = '_'.join([instance.split("/")[-1].split(".")[0].split("=")[-1] for instance in instances])
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        new_instance = eliminate_tasks(mm_instance, interval_choice, seed=seed)
        new_instance.calculate_stats()
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in new_instance.data.items()])
        new_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        new_instance.model_data_to_yaml(filepath)

def make_reduced_from_one_instance(filepath, SALBP_instance_list, model_names, cycle_time, to_reduce, shared_root=True, seed = None, parent_set = "Otto"):
    '''Deletes tasks randomly from the same SALBP_instance to create two or more distinct models.
        parameters: filepath: the path to the file where the mixed model instances will be written
                    SALBP_instance_list: a list of SALBP_instances
                    model_names: a list of model names
                    cycle_time: the cycle time of the mixed model instance
                    to_reduce: the number of tasks to delete, a dictionary with the keys being the models and the values the number of tasks to remove
                    shared_root: a boolean indicating whether the tasks to delete should share a root, i.e. should 
                                    it have the same base tasks with different tasks added on top, or should it eliminate different tasks from the original precedence graph for each model
                    seed: a seed for the random number generator
    '''
    for instance in SALBP_instance_list:
        model_mixture = random_model_mixture(model_names, seed)
        instances = [instance for i in range(len(model_names))]
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        if shared_root:
            new_instance = eliminate_tasks_subgraph(mm_instance, to_reduce, seed=seed)
        else:
            new_instance = eliminate_tasks_different_graphs(mm_instance, to_reduce, seed=seed)
        instance_name = instance.split("/")[-1].split(".")[0].split("=")[-1]
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in new_instance.data.items()])
        new_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        new_instance.calculate_stats()
        new_instance.model_data_to_yaml(filepath)

def make_reduced_from_one_instance_task_time_perturbation(filepath, SALBP_instance_list, model_names, cycle_time, to_reduce, perturbation_amount, shared_root=True, seed = None, parent_set = "Otto"):
    '''Deletes tasks randomly from the same SALBP_instance to create two or more distinct models. It then perturbs the task times of the tasks that are not deleted.
        parameters: filepath: the path to the file where the mixed model instances will be written
                    SALBP_instance_list: a list of SALBP_instances
                    model_names: a list of model names
                    cycle_time: the cycle time of the mixed model instance
                    to_reduce: the number of tasks to delete, a dictionary with the keys being the models and the values the number of tasks to remove
                    shared_root: a boolean indicating whether the tasks to delete should share a root, i.e. should 
                                    it have the same base tasks with different tasks added on top, or should it eliminate different tasks from the original precedence graph for each model
                    perturbation_amount: A dictionary of dictionaries. For each model, it has a dictionary with one key being the number of tasks to randomly perturb,
                                         and the other being  the percentage of the perturbation (positive or negative)
                    seed: a seed for the random number generator
    '''
    for instance in SALBP_instance_list:
        model_mixture = random_model_mixture(model_names, seed)
        instances = [instance for i in range(len(model_names))]
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        if shared_root:
            new_instance = eliminate_tasks_subgraph(mm_instance, to_reduce, seed=seed)
        else:
            new_instance = eliminate_tasks_different_graphs(mm_instance, to_reduce, seed=seed)
        #gets the text from instance that is before the period and after the last slash
        instance_name = instance.split("/")[-1].split(".")[0].split("=")[-1]
        new_instance = perturb_task_times(new_instance, perturbation_amount, seed)
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in new_instance.data.items()])
        new_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        new_instance.calculate_stats()
        new_instance.model_data_to_yaml(filepath)


        
def make_instances(filepath,SALBP_instance_list,model_names,cycle_time, seed = None):
    '''Creates mixed model instances from a list of SALBP_instances. It procedes down the list of instances, 
    taking the first n instances where n is the number of models in model_names. 
    It then creates a model mixture and creates a mixed model instance from the instances and the model mixture. 
    It then writes the mixed model instance to a file. This is repeated until the end of the list of instances is reached.'''
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        mm_instance.generate_name()
        mm_instance.calculate_stats()
        mm_instance.model_data_to_yaml(filepath)




if __name__ == "__main__":
    print("This is the MALBP_instance_generation.py file")