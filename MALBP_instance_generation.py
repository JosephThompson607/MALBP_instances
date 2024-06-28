import sys  
sys.path.insert(1, '..')
from ALB_instance_tools import *
import random
from scipy.stats import bernoulli

def random_model_mixture(model_names, seed=None):
    '''Creates a dictionary where the model name are the keys and the probability is the values. The probabilities sum to 1.
        parameters: model_names: a list of model names'''
    #if seed is a rng object, then it will use that rng object, otherwise it will create a new one
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    while True:
        model_mixture = {}
        for model_name in model_names:
            model_mixture[model_name] = rng.uniform(0, 1)
        total = sum(model_mixture.values())

        for model_name in model_mixture.keys():
            model_mixture[model_name] = round(model_mixture[model_name]/total, 2)

        #makes sure the sum of the probabilities is 1 by rounding the last value
        model_mixture[model_names[-1]] = 1 - sum([value for value in model_mixture.values() if value != model_mixture[model_names[-1]]])
        if model_mixture[model_names[-1]] >= 0:
            break
    return model_mixture

def make_reduced_instances(out_fp, 
                           SALBP_instance_list, 
                           model_names, 
                           cycle_time, 
                           interval_choice = (0.7,0.8), 
                           seed = None,
                           parent_set = "Otto"):
    '''Deletes the same tasks from two different SALBP models in the same mixed model instance.
        parameters: out_fp: the path to the file where the mixed model instances will be written
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
        new_instance.model_data_to_yaml(out_fp)

def make_reduced_from_one_instance(out_fp, SALBP_instance_list, model_names, cycle_time, to_reduce, shared_root=True, seed = None, parent_set = "Otto"):
    '''Deletes tasks randomly from the same SALBP_instance to create two or more distinct models.
        parameters: out_fp: the path to the file where the mixed model instances will be written
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
        new_instance.model_data_to_yaml(out_fp)

def make_reduced_from_one_instance_task_time_perturbation(out_fp, SALBP_instance_list, model_names, cycle_time, to_reduce, perturbation_amount, shared_root=True, seed = None, parent_set = "Otto"):
    '''Deletes tasks randomly from the same SALBP_instance to create two or more distinct models. It then perturbs the task times of the tasks that are not deleted.
        parameters: out_fp: the path to the file where the mixed model instances will be written
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
        new_instance = perturb_task_times(new_instance, perturbation_amount, seed)
        instance_name = instance.split("/")[-1].split(".")[0].split("=")[-1]
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in new_instance.data.items()])
        new_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        new_instance.calculate_stats()
        new_instance.model_data_to_yaml(out_fp)


def make_reduced_multi_instance(out_fp, SALBP_instance_list, model_names, cycle_time, to_reduce, perturbation_amount, shared_root=True, seed = None, parent_set = "Otto"):
    '''Creates mixed model instances from a list of SALBP_instances. It procedes down the list of instances,
    taking the first n instances where n is the number of models in model_names. It then eliminates tasks from the instances.
    parameters: out_fp: the path to the file where the mixed model instances will be written
                    SALBP_instance_list: a list of SALBP_instances
                    model_names: a list of model names
                    cycle_time: the cycle time of the mixed model instance
                    to_reduce: the number of tasks to delete, a dictionary with the keys being the models and the values the number of tasks to remove
                    shared_root: a boolean indicating whether the tasks to delete should share a root, i.e. should 
                                    it have the same base tasks with different tasks added on top, or should it eliminate different tasks from the original precedence graph for each model
                    perturbation_amount: A dictionary of dictionaries. For each model, it has a dictionary with one key being the number of tasks to randomly perturb,
                                         and the other being  the percentage of the perturbation (positive or negative)
                    seed: a seed for the random number generator'''
    for i in range(len(SALBP_instance_list)-len(model_names)+1):
        instances = SALBP_instance_list[i:i+len(model_names)]
        model_mixture = random_model_mixture(model_names, seed)
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        if shared_root:
            new_instance = eliminate_tasks_subgraph(mm_instance, to_reduce, seed=seed)
        else:
            new_instance = eliminate_tasks_different_graphs(mm_instance, to_reduce, seed=seed)
        new_instance = perturb_task_times(new_instance, perturbation_amount, seed)
        instance_name = '_'.join([instance.split("/")[-1].split(".")[0].split("=")[-1] for instance in instances])
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in new_instance.data.items()])
        new_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        new_instance.calculate_stats()
        new_instance.model_data_to_yaml(out_fp)

def make_instances(out_fp,SALBP_instance_list,model_names,cycle_time, seed = None, parent_set = "Otto"):
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
        instance_name = '_'.join([instance.split("/")[-1].split(".")[0].split("=")[-1] for instance in instances])
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in mm_instance.data.items()])
        mm_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        mm_instance.model_data_to_yaml(out_fp)


def make_sikora_like_instances(out_fp, instance_list,model_names, elimination_prob,  perturbation_prob,  n_tasks, perturb_range = (-0.5, 0.5), cycle_time=1000, seed=None):
    '''Creates instances that are similar to the ones in Sikora 2024. It eliminates a random amount of tasks from the instances and then perturbs the task times of the remaining tasks. Note that for Sikora's actual instances,
    the different models are created from selecting different "options" from the tasks. So there would be a huge number of possible models. This creates a few models with similar properties.
    parameters: out_fp: the path to the file where the mixed model instances will be written
                instance_list: a list of SALBP_instances
                model_names: a list of model names
                elimination_prob: the probability of eliminating a task
                perturbation_prob: the probability of perturbing a task
                n_tasks: the number of tasks in the instance
                perturb_range: a tuple of two floats that represent the range of the perturbation. (-0.5, 0.5) means the perturbation will be between -50% and 50%
                cycle_time: the cycle time of the mixed model instance
                seed: a seed for the random number generator'''
    #elim amount is the amount of tasks to eliminate, based on a bernoulli distribution
    #sets the seed
    rng = np.random.default_rng(seed)
    perturb_dict = {}
    to_reduce = {}
    for instance in instance_list:
        for model in model_names:
            np.random.seed()
            elim_amount = sum(bernoulli.rvs(elimination_prob, size=n_tasks, random_state=rng))
            to_reduce[model] = elim_amount
            remaining_tasks = n_tasks - elim_amount
            perturb_seed = random.randint(0, 1000)
            perturb_number = sum(bernoulli.rvs(perturbation_prob, size=remaining_tasks, random_state=rng))
            perturb_dict[model] = {"num_tasks":perturb_number, "lower_bound": perturb_range[0], "upper_bound": perturb_range[1]}
        instance_seed = rng.integers(1,100000).item()
        make_reduced_from_one_instance_task_time_perturbation(out_fp, [instance], model_names, perturbation_amount=perturb_dict, cycle_time=cycle_time, to_reduce=to_reduce, seed=instance_seed, shared_root=False)


def make_reduced_multi_instance_targeted_instances(out_fp, SALBP_instance_list, target_instances, model_names, cycle_time, to_reduce, perturbation_amount, shared_root=True, seed = None, parent_set = "Otto"):
    '''Creates mixed model instances from a list of SALBP_instances. It uses the instances around a target instance to generate the sequence. It procedes down the list of instances,
    taking the first n instances where n is the number of models in model_names. It then eliminates tasks from the instances.
    parameters: out_fp: the path to the file where the mixed model instances will be written
                    SALBP_instance_list: a list of SALBP_instances
                    target_instances: a list of indices of the target instances
                    model_names: a list of model names
                    cycle_time: the cycle time of the mixed model instance
                    to_reduce: the number of tasks to delete, a dictionary with the keys being the models and the values the number of tasks to remove
                    shared_root: a boolean indicating whether the tasks to delete should share a root, i.e. should 
                                    it have the same base tasks with different tasks added on top, or should it eliminate different tasks from the original precedence graph for each model
                    perturbation_amount: A dictionary of dictionaries. For each model, it has a dictionary with one key being the number of tasks to randomly perturb,
                                         and the other being  the percentage of the perturbation (positive or negative)
                    seed: a seed for the random number generator'''
    used_instances = []
    rng = np.random.default_rng(seed)
    for i in target_instances:
        #if to_reduce has the the key 'Random', then it will randomly select the number of tasks to reduce for each model
        if 'Random' in to_reduce.keys():
            value = to_reduce['Random']
            reductions = rng.integers(value[0], value[1], size=len(model_names)).tolist()
            print("reductions", reductions)
            model_reductions = {model: reduction for model, reduction in zip(model_names, reductions)}
        else:
            model_reductions = to_reduce
            
        #if the target instance plus the other instances is greater than the length of the SALBP_instance_list, then uses the instances before it
        if i + len(model_names) > len(SALBP_instance_list):
            extra_instances = i + len(model_names) - len(SALBP_instance_list)
            instances = SALBP_instance_list[i-extra_instances:i+len(model_names)-extra_instances]
        else:
            instances = SALBP_instance_list[i:i+len(model_names)]
        print("instances", instances)
        used_instances.append(instances)
        model_mixture = random_model_mixture(model_names, seed=rng) 
        model_dicts = make_instance_pair(instances, model_mixture)
        mm_instance = MixedModelInstance(model_dicts=model_dicts, cycle_time=cycle_time)
        if shared_root:
            new_instance = eliminate_tasks_subgraph(mm_instance, model_reductions, seed=rng)
        else:
            new_instance = eliminate_tasks_different_graphs(mm_instance, model_reductions, seed=rng)
        new_instance = perturb_task_times(new_instance, perturbation_amount, rng)
        instance_name = '_'.join([instance.split("/")[-1].split(".")[0].split("=")[-1] for instance in instances])
        name_tasks = '_'.join([ str(key) + str(value['num_tasks']) for (key, value) in new_instance.data.items()])
        new_instance.name = parent_set + "_" +  instance_name + "_" + "MODELtasks" + "_" +  name_tasks
        new_instance.calculate_stats()
        new_instance.model_data_to_yaml(out_fp)
    return used_instances
if __name__ == "__main__":
    print("This is the MALBP_instance_generation.py file")