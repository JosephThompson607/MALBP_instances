import sys  
sys.path.insert(1, '..')
from ALB_instance_tools import *
import random

def random_model_mixture(model_names, seed = None):
    '''Creates a dictionary where the model name are the keys and the probability is the values. The probabilities sum to 1'''
    model_mixture = {}
    random.seed(seed)
    for model_name in model_names:
        model_mixture[model_name] = random.random()
    total = sum(model_mixture.values())
    for model_name in model_mixture.keys():
        model_mixture[model_name] = model_mixture[model_name]/total
    return model_mixture

def make_reduced_instances(filepath, SALBP_instance_list, model_names, cycle_time, interval_choice = (0.7,0.8), seed = None):
    '''Deletes the same tasks from two different models in the same mixed model instance.'''
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        print("These are the model_dicts")
        print(model_dicts)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        print("This is the data")
        new_instance = eliminate_tasks(mm_instance, interval_choice, seed=seed)
        new_instance.generate_name()
        new_instance.calculate_stats()
        new_instance.model_data_to_yaml(filepath)

def make_reduced_from_one_instance(filepath, SALBP_instance_list, model_names, cycle_time, to_reduce, shared_root=True, seed = None):
    '''Deletes tasks randomly from the same SALBP_instance to create two or more distinct models.'''
    for instance in SALBP_instance_list:
        model_mixture = random_model_mixture(model_names, seed)
        instances = [instance for i in range(len(model_names))]
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        if shared_root:
            new_instance = eliminate_tasks_shared_root(mm_instance, to_reduce, seed=seed)
        else:
            new_instance = eliminate_tasks_different_root(mm_instance, to_reduce, seed=seed)
        new_instance.generate_name()
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