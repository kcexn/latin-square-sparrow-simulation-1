from simulation import Simulation
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import csv
import configparser
import multiprocessing

experiment_name = "10000 servers"

matplotlib.set_loglevel("critical")
def histogram(times, num_bins, title, save_file = None, y_height=8.3, x_width=11.7):
    """y_height and x_widht are both given in inches."""
    fig,ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('P(T < time)')
    ax.set_yticks(np.arange(0,1.21,0.1))
    ax.set_yticks(np.arange(0,1.21,0.02), minor=True)
    # max_latencies = [max(samples) for samples in times]
    max_latency = max(times)
    ax.set_xticks(np.arange(0,max_latency+0.01,max([max_latency//13,1])))
    ax.set_xticks(np.arange(0,max_latency+0.01,max([max_latency/65,0.2])), minor=True)
    # labels=["Latin Square","Sparrow","Round Robin","Random"]
    ax.ecdf(times)
    # for idx,samples in enumerate(times):
        # ax.ecdf(samples, label=f"{labels[idx]}")
    # ax.legend()
    fig.set_figwidth(x_width) #inches
    fig.set_figheight(y_height) #inches
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(f'{save_file}.pdf', format='pdf', bbox_inches = 'tight')
        plt.savefig(f'{save_file}.eps', format='eps', bbox_inches = 'tight', backend='PS')
        plt.close()

def box_whisker(times, title, save_file = None, y_height = 8.3, x_width = 11.7):
    """y_height and x_width are both given in inches."""
    fig,ax = plt.subplots(1,1)
    ax.boxplot(times)
    ax.set_ylabel('time')
    ax.set_yticks(np.arange(0,max(times)+0.01,1))
    ax.set_yticks(np.arange(0,max(times)+0.01,0.2), minor=True)
    ax.set_title(title)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(f'{save_file}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{save_file}.eps', format='eps', bbox_inches='tight', backend='PS')
        plt.close()

def experiment(index, configuration, experiment_name = experiment_name):
    import math
    HISTOGRAM = True

    sim = Simulation(configuration)
    sim.run()

    jobs = [job for job in sim.work]
    job_finish_times = [job.finish_time for job in jobs]
    job_start_times = [job.start_time for job in jobs]
    job_latencies = [finish_time - start_time for (finish_time, start_time) in zip(job_finish_times,job_start_times)]
    mean_job_latency = sum(job_latencies)/len(job_latencies)
    sorted_job_latencies = sorted(job_latencies)
    median_job_latency = sorted_job_latencies[math.ceil(len(sorted_job_latencies)/2)]
    job_latency_quartile_3 = sorted_job_latencies[math.ceil(len(sorted_job_latencies)*(3/4))]
    job_latency_quartile_1 = sorted_job_latencies[math.ceil(len(sorted_job_latencies)*(1/4))]
    job_latency_iqr = job_latency_quartile_3 - job_latency_quartile_1

    with open(f'./experiments/{experiment_name}/{index}/data/job-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        data_heading = ['job_start_times', 'job_finish_times', 'job_latencies']
        writer.writerow(data_heading)
        data = [[start_time, finish_time, latency] for (start_time, finish_time, latency) in zip(job_start_times, job_finish_times, job_latencies)]
        writer.writerows(data)

    tasks = [task for job in jobs for task in job.tasks]
    task_finish_times = [task.finish_time for task in tasks]
    task_start_times = [task.start_time for task in tasks]
    task_latencies = [finish_time - start_time for (finish_time, start_time) in zip(task_finish_times, task_start_times)]

    with open(f'./experiments/{experiment_name}/{index}/data/task-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        data_heading = ['task_start_times', 'task_finish_times', 'task_latencies']
        writer.writerow(data_heading)
        data = [[start_time, finish_time, latency] for (start_time, finish_time, latency) in zip(task_start_times, task_finish_times, task_latencies)]
        writer.writerows(data)

    sim_finish_time = max(task_finish_times)
    sim_start_time = min(task_start_times)
    total_sim_time = sim_finish_time - sim_start_time

    servers = [server for server in sim.cluster.servers]
    cum_idle_times = [server.cumulative_idle_time for server in servers]
    cum_busy_times = [server.cumulative_busy_time for server in servers]
    mean_availability = sum(cum_idle_times)/(sum(cum_busy_times) + sum(cum_idle_times))

    with open(f'./experiments/{experiment_name}/{index}/data/summary-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        data_heading = [
            'mean job latency', 
            'median job latency', 
            'quartile 3 job latency',
            'quartile 1 job latency',
            'job latency iqr'
        ]
        writer.writerow(data_heading)
        data = [mean_job_latency, median_job_latency, job_latency_quartile_3, job_latency_quartile_1, job_latency_iqr]
        writer.writerow(data)

    print(f'experiment {index}: {mean_job_latency}, '
          +f'{job_latency_quartile_1}, {median_job_latency}, {job_latency_quartile_3}, {job_latency_iqr}, '
          +f'{mean_availability}, {total_sim_time}')

    if HISTOGRAM:
        num_bins=50
        histogram(job_latencies, num_bins, f'Experiment {index}: Job Latency Distribution', save_file=f'./experiments/{experiment_name}/{index}/figures/job_latencies')

        # num_bins=50
        # histogram(task_latencies, num_bins, 'Distribution of Task Latencies', save_file=f'./experiments/{experiment_name}/{index}/figures/task_latencies')

        box_whisker(job_latencies, f'Experiment {index}: Job Latencies', save_file=f'./experiments/{experiment_name}/{index}/figures/job_latencies_box')

def postprocessing_histogram(times, title, save_file = None, y_height=8.3, x_width=11.7):
    """y_height and x_widht are both given in inches."""
    fig,ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('P(T < time)')
    ax.set_yticks(np.arange(0,1.21,0.1))
    ax.set_yticks(np.arange(0,1.21,0.02), minor=True)
    max_latencies = [max(samples) for samples in times]
    max_latency = max(max_latencies)
    ax.set_xticks(np.arange(0,max_latency+0.01,max([max_latency//13,1])))
    ax.set_xticks(np.arange(0,max_latency+0.01,max([max_latency/65,0.2])), minor=True)
    labels=["Latin Square","Sparrow","Round Robin","Random"]
    # ax.ecdf(times)
    for idx,samples in enumerate(times):
        ax.ecdf(samples, label=f"{labels[idx]}")
    ax.legend()
    fig.set_figwidth(x_width) #inches
    fig.set_figheight(y_height) #inches
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(f'{save_file}.pdf', format='pdf', bbox_inches = 'tight')
        plt.savefig(f'{save_file}.eps', format='eps', bbox_inches = 'tight', backend='PS')
        plt.close()
        
def postprocessing_box_plot(times, title, save_file = None, y_height = 8.3, x_width = 11.7):
    """y_height and x_width are both given in inches."""
    fig,ax = plt.subplots(1,1)
    ax.boxplot(times,
        showfliers=False
    )
    ax.set_ylabel('time')
    ax.set_xticks(range(1,len(times)+1), labels=["Correlated - Homogeneous Servicing", 
            "Correlated - Heterogeneous Servicing", 
            "Uncorrelated - Homogeneous Servicing", 
            "Uncorrelated - Heterogeneous Servicing"],
            rotation=20,
            horizontalalignment='right')
    ax.set_title(title)
    fig.set_figwidth(x_width)
    fig.set_figheight(y_height)
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(f'{save_file}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{save_file}.eps', format='eps', bbox_inches='tight', backend='PS')
        plt.close()
        
def process_probability_function_data(experiment_path, num_experiments):
    titles=["Correlated Processing of Homogeneous Tasks", 
        "Correlated Processing of Heterogeneous Tasks", 
        "Uncorrelated Processing of Homogeneous Tasks", 
        "Uncorrelated Processing of Heterogeneous Tasks"]
    policies=["Latin Square", "Sparrow", "Round Robin", "Random"]
    for i,title in enumerate(titles):
        job_latencies = []
        for j,policy in enumerate(policies):
            with open(f"{experiment_path}/{j*len(policies)+i+1}/data/job-data.csv", newline='') as f:
                reader = csv.DictReader(f)
                latencies = [float(row["job_finish_times"]) - float(row["job_start_times"]) for row in reader]
                job_latencies.append(latencies)
        postprocessing_histogram(job_latencies, title, save_file=f"{experiment_path}/post_processed_figures/{title}")
    job_latencies = []
    for j,policy in enumerate(policies):
        policy_latencies = []
        for i,title in enumerate(titles):
            with open(f"{experiment_path}/{j*len(policies)+i+1}/data/job-data.csv", newline='') as f:
                reader = csv.DictReader(f)
                latencies = [float(row["job_finish_times"]) - float(row["job_start_times"]) for row in reader]
                policy_latencies.append(latencies)
        job_latencies.append([sum(latency/4 for latency in latencies) for latencies in zip(*policy_latencies)])
    title = "Uniform Mixture Processing"
    postprocessing_histogram(job_latencies, title, save_file=f"{experiment_path}/post_processed_figures/{title}")
        
def process_histogram_data(experiment_path, num_experiments):
    titles = ["Latin Square Schedule", "Sparrow Schedule", "Round Robin Schedule", "Random Schedule"]
    processing_models = ["Correlated - Homogeneous Servicing", "Correlated - Heterogeneous Servicing", "Uncorrelated - Homogeneous Servicing", "Uncorrelated - Heterogeneous Servicing"]
    for i,title in enumerate(titles):
        job_latencies = []
        for j,models in enumerate(processing_models):
            with open(f"{experiment_path}/{i*len(processing_models)+j+1}/data/job-data.csv", newline='') as f:
                reader = csv.DictReader(f)
                latencies = [float(row["job_finish_times"]) - float(row["job_start_times"]) for row in reader]
                job_latencies.append(latencies)
        postprocessing_box_plot(job_latencies, title, save_file=f"{experiment_path}/post_processed_figures/{title}")

def post_process_data(experiment_path, num_experiments):
    process_probability_function_data(experiment_path, num_experiments)
    process_histogram_data(experiment_path, num_experiments)

if __name__ == '__main__':
    from enum import Enum
    from functools import reduce
    import itertools
    import datetime
    import os
    scheduling_policies = ['LatinSquare', 'Sparrow', 'RoundRobin', 'CompletelyRandom']
    # (Correlated, Homogeneous)
    task_service_time_config = [(True, True), (True, False), (False, True), (False, False)]
    # job_arrival_policies = ['Erlang','Exponential']
    job_arrival_policies=['Exponential']
    num_tasks_per_job = 100
    class LoadEnum(Enum):
        """
        Erlang job arrival policy task arrival scale parameters.
        multiply by num_tasks_per_job to get the equivalent Exponential job arrival policy
        scale factor.
        """
        HIGH = 0.0001111111
        MEDIUM = 0.0002
        LOW = 0.001
    # task_arrival_scale_factors = [LoadEnum.LOW.value, LoadEnum.MEDIUM.value, LoadEnum.HIGH.value]
    task_arrival_scale_factors = [LoadEnum.MEDIUM.value]
    

    parameters = [
        scheduling_policies,
        task_service_time_config,
        job_arrival_policies,
        task_arrival_scale_factors
    ]
    num_experiments = reduce(
        lambda x,y: x*y,
        (len(param) for param in parameters)
    )
    for i in range(num_experiments):
        try:
            os.makedirs(f'experiments/{experiment_name}/{i+1}/data/')
            os.makedirs(f'experiments/{experiment_name}/{i+1}/figures/')
            os.makedirs(f'experiments/{experiment_name}/post_processed_figures/')
        except os.error:
            if not (os.access(f'experiments/{experiment_name}/{i+1}/data/', os.F_OK) 
                and os.access(f'experiments/{experiment_name}/{i+1}/figures/', os.F_OK) 
                and os.access(f'experiments/{experiment_name}/post_processed_figures/', os.F_OK)):
                raise Exception("File error can't make paths, and paths don't exist.")
    experiment_params = [param for param in itertools.product(*parameters)]
    configs = [configparser.ConfigParser() for _ in range(num_experiments)]
    for idx,config in enumerate(configs):
        config.read('./configuration.ini')
        match experiment_params[idx]:
            case [scheduler_policy, task_service_time_config, arrival_policy, task_arrival_scale]:
                print(f'experiment {idx+1}: scheduler: {scheduler_policy}, arrivals: {arrival_policy}, '
                      + f'correlated task servicing: {task_service_time_config[0]}, homogeneous task servicing: {task_service_time_config[1]}, '
                      + f'arrival rate: {1/float(task_arrival_scale) if arrival_policy == "Erlang" else 1/(num_tasks_per_job*float(task_arrival_scale))}')
                config['Computer.Scheduler']['POLICY'] = scheduler_policy
                if arrival_policy == 'Erlang':
                    config['Processes.Arrival']['SCALE'] = str(task_arrival_scale)
                else:
                    config['Processes.Arrival']['SCALE'] = str(num_tasks_per_job*float(task_arrival_scale))
                config['Processes.Arrival.Job']['POLICY'] = arrival_policy
                config['Processes.Completion.Task']['CORRELATED_TASKS'] = str(task_service_time_config[0])
                config['Processes.Completion.Task']['HOMOGENEOUS_TASKS'] = str(task_service_time_config[1])
    start_time = datetime.datetime.now()
    print(f'start time: {start_time}')
    print(f"experiment index: mean, q1, median, q3, iqr, mean availability, total simulation time")

    with multiprocessing.Pool(maxtasksperchild=1) as p:
        p.starmap(experiment, [(idx+1, configs[idx]) for idx in range(num_experiments)], 1)
    end_time = datetime.datetime.now()
    print(f'End time: {end_time}. Duration: {end_time - start_time}.')
    
    post_process_data(f'experiments/{experiment_name}', len(experiment_params))