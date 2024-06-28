import pandas as pd
import re

def instance_list_from_detail_excel(in_fp, instance_folder, instance_numbers, out_fp = None, sheet_name='Summary', extension='.alb', name='filtered_instances'):
    df = pd.read_excel(in_fp, sheet_name=sheet_name, skiprows=1)
    selected_df = df[df['<No>'].isin(instance_numbers)]
    selected_df.columns = selected_df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.replace('\n', ' ')
    if out_fp:
        selected_df.to_csv(out_fp + name + '.csv')
    instances_names = selected_df['Filename'].to_list()
    instances = [instance_folder + name + extension for name in instances_names]
    return instances

def instance_list_from_detail_csv(detail_csv, instance_folder, extension=".alb"):
    df = pd.read_csv(detail_csv)
    instances_names = df['Filename'].to_list()
    instances = [instance_folder + name + extension for name in instances_names]
    return instances

def parse_alb(alb_file_name):
    """Reads assembly line balancing instance .alb file, returns dictionary with the information"""
    parse_dict = {}
    alb_file = open(alb_file_name).read()
    # Get number of tasks
    num_tasks = re.search("<number of tasks>\n(\\d*)", alb_file)
    parse_dict["num_tasks"] = int(num_tasks.group(1))

    # Get cycle time
    cycle_time = re.search("<cycle time>\n(\\d*)", alb_file)
    parse_dict["cycle_time"] = int(cycle_time.group(1))

    # Order Strength
    order_strength = re.search("<order strength>\n(\\d*,\\d*)", alb_file)
    
    if order_strength:
        parse_dict["original_order_strength"] = float(order_strength.group(1).replace(",", "."))
    else:
        order_strength = re.search("<order strength>\n(\\d*.\\d*)", alb_file)
        parse_dict["original_order_strength"] = float(order_strength.group(1))

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