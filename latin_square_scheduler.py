from simulation import Simulation
from matplotlib import pyplot as plt
import numpy as np
import csv
import configparser
import multiprocessing

def histogram(times, num_bins, title, save_file = None, y_height=8.3, x_width=11.7):
    """y_height and x_widht are both given in inches."""
    fig,ax = plt.subplots(1,1)
    ax.hist(times, bins = num_bins, density=True, cumulative=True, histtype='step', label='cdf')
    ax.hist(times, bins = num_bins, density=True, histtype='step', label='densities')
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('frequencies')
    ax.set_yticks(np.arange(0,1.21,0.1))
    ax.set_yticks(np.arange(0,1.21,0.02),minor=True)
    ax.set_xticks(np.arange(0,60.01,1))
    ax.set_xticks(np.arange(0,60.01,0.2), minor=True)
    ax.legend()
    fig.set_figwidth(x_width) #inches
    fig.set_figheight(y_height) #inches
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, format='pdf')

def experiment(index, configuration, experiment_name = '20 servers'):
    HISTOGRAM = True

    sim = Simulation(configuration)
    sim.run()

    jobs = [job for job in sim.work]
    job_finish_times = [job.finish_time for job in jobs]
    job_start_times = [job.start_time for job in jobs]
    job_latencies = [finish_time - start_time for (finish_time, start_time) in zip(job_finish_times,job_start_times)]
    mean_job_latency = sum(job_latencies)/len(job_latencies)

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
    mean_task_latency = sum(task_latencies)/len(task_latencies)

    with open(f'./experiments/{experiment_name}/{index}/data/task-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        data_heading = ['job_start_times', 'job_finish_times', 'job_latencies']
        writer.writerow(data_heading)
        data = [[start_time, finish_time, latency] for (start_time, finish_time, latency) in zip(task_start_times, task_finish_times, task_latencies)]
        writer.writerows(data)

    sim_finish_time = max(task_finish_times)
    sim_start_time = min(task_start_times)
    total_sim_time = sim_finish_time - sim_start_time

    servers = [server for server in sim.scheduler.cluster.servers]
    cum_idle_times = [server.cumulative_idle_time for server in servers]
    cum_busy_times = [server.cumulative_busy_time for server in servers]
    availability = [idle_time/(idle_time+busy_time) for (idle_time, busy_time) in zip(cum_idle_times, cum_busy_times)]
    mean_availability = sum(availability)/len(availability)

    with open(f'./experiments/{experiment_name}/{index}/data/summary-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        data_heading = ['mean job latency', 'mean task latency']
        writer.writerow(data_heading)
        data = [mean_job_latency, mean_task_latency]
        writer.writerow(data)

    print(f'experiment {index}: {mean_job_latency}, {mean_task_latency}, {mean_availability}, {total_sim_time}')

    if HISTOGRAM:
        num_bins=50
        histogram(job_latencies, num_bins, 'Distribution of Job Latencies', save_file=f'./experiments/{experiment_name}/{index}/figures/job_latencies.pdf')

        num_bins=50
        histogram(task_latencies, num_bins, 'Distribution of Task Latencies', save_file=f'./experiments/{experiment_name}/{index}/figures/task_latencies.pdf')


def generate_latin_square(n):
    latin_square = [[(i+j)%n for i in range(n)] for j in range(n)]
    with open('latin_square.txt', 'w') as f:
        f.write(f'{latin_square}')

if __name__ == '__main__':
    experiment_name = '20 servers'
    num_experiments = 12
    experiments = [i+1 for i in range(12)]
    parameters = [
        ('Sparrow', 'RandomJob', '0.06'), 
        ('Sparrow', 'RandomJob', '0.1'),
        ('Sparrow', 'RandomJob', '0.5'),
        ('LatinSquare', 'RandomJob', '0.06'),
        ('LatinSquare', 'RandomJob', '0.1'),
        ('LatinSquare', 'RandomJob', '0.5'),
        ('Sparrow', 'RandomTask', '0.06'),
        ('Sparrow', 'RandomTask', '0.1'),
        ('Sparrow', 'RandomTask', '0.5'),
        ('LatinSquare', 'RandomTask', '0.06'),
        ('LatinSquare', 'RandomTask', '0.1'),
        ('LatinSquare', 'RandomTask', '0.5')
    ]
    configs = [configparser.ConfigParser() for _ in range(len(parameters))]

    for idx,config in enumerate(configs):
        config.read('./configuration.ini')
        match parameters[idx]:
            case [scheduler_policy, completion_policy, scale]:
                print(f'{idx+1}: {scheduler_policy, completion_policy, scale}')
                config['Computer.Scheduler']['POLICY'] = scheduler_policy
                config['Processes.Completion.Task']['POLICY'] = completion_policy
                config['Processes.Arrival']['SCALE'] = scale

    processes = [multiprocessing.Process(target=experiment, args=(index, configuration)) for (index,configuration) in zip(experiments,configs)]
    for process in processes:
        process.start()

    for process in processes:
        process.join()