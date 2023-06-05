from simulation import Simulation
from matplotlib import pyplot as plt
import numpy as np
import csv

def histogram(times, num_bins, y_height=8.3, x_width=11.7):
    """y_height and x_widht are both given in inches."""
    fig,ax = plt.subplots(1,1)
    ax.hist(times, bins = num_bins, density=True, cumulative=True, histtype='step', label='cdf')
    ax.hist(times, bins = num_bins, density=True, histtype='step', label='densities')
    ax.set_title('Distribution of Job Latencies')
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')
    ax.set_yticks(np.arange(0,1.21,0.1))
    ax.set_yticks(np.arange(0,1.21,0.02),minor=True)
    ax.set_xticks(np.arange(0,20.01,1))
    ax.set_xticks(np.arange(0,20.01,0.2), minor=True)
    ax.legend()
    fig.set_figwidth(x_width) #inches
    fig.set_figheight(y_height) #inches
    plt.show()


def generate_latin_square(n):
    latin_square = [[(i+j)%n for i in range(n)] for j in range(n)]
    with open('latin_square.txt', 'w') as f:
        f.write(f'{latin_square}')

if __name__ == '__main__':
    HISTOGRAM = True

    sim = Simulation()
    sim.run()

    jobs = [job for job in sim.work]
    job_latencies = [job.finish_time - job.start_time for job in jobs]
    avg_job_latency = sum(job_latencies)/len(job_latencies)

    sim_data_headings = ['job start time', 'job finish time', 'num tasks in job', 'job latency']
    sim_data = [[job.start_time, job.finish_time, len(job.tasks), job.finish_time - job.start_time] for job in jobs]
    with open('sim-job-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(sim_data_headings)
        writer.writerows(sim_data)

    sim_data_headings = ['task start time', 'task finish time', 'task latency']
    sim_data = [[task.start_time, task.finish_time, task.finish_time - task.start_time] for job in jobs for task in job.tasks]
    with open('sim-task-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(sim_data_headings)
        writer.writerows(sim_data)

    job_finish_times = [job.finish_time for job in jobs]
    job_start_times = [job.start_time for job in jobs]
    avg_job_latency = sum([job.finish_time - job.start_time for job in jobs])/len(jobs)
    avg_time_per_job = (max(job_finish_times) - min(job_start_times))/len(jobs)

    task_finish_times = [task.finish_time for job in jobs for task in job.tasks]
    task_start_times = [task.start_time for job in jobs for task in job.tasks]
    avg_task_latency = sum([task.finish_time - task.start_time for job in jobs for task in job.tasks])/len(task_start_times)
    avg_time_per_task = (max(task_finish_times) - min(task_start_times))/len(task_start_times)

    sim_data_headings = ['avg job latency', 'avg time per job', 'avg task latency', 'avg time per task']
    with open('sim-avg-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(sim_data_headings)
        writer.writerow([avg_job_latency, avg_time_per_job, avg_task_latency, avg_time_per_task])

    print(f'{avg_job_latency}, {avg_time_per_job}, {avg_task_latency}, {avg_time_per_task}')


    if HISTOGRAM:
        num_bins=50
        histogram(job_latencies, num_bins)