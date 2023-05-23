from simulation import Simulation
from matplotlib import pyplot as plt
import numpy as np
import csv

def histogram(times, num_bins, y_height=8.3, x_width=11.7):
    """y_height and x_widht are both given in inches."""
    fig,ax = plt.subplots(1,1)
    ax.hist(job_latencies, bins = num_bins, density=True, cumulative=True, histtype='step', label='cdf')
    ax.hist(job_latencies, bins = num_bins, density=True, histtype='step', label='densities')
    ax.set_title('Distribution of Average Task Latencies')
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')
    ax.set_yticks(np.arange(0,2.51,0.1))
    ax.set_yticks(np.arange(0,2.51,0.02),minor=True)
    ax.set_xticks(np.arange(0,5.01,1))
    ax.set_xticks(np.arange(0,5.01,0.2), minor=True)
    ax.legend()
    fig.set_figwidth(x_width) #inches
    fig.set_figheight(y_height) #inches
    plt.show()

if __name__ == '__main__':
    POLICY = 'LatinSquare'
    LATIN_SQUARE_ORDER = 6

    sim = Simulation()
    sim.run()

    jobs = [job for job in sim.work]
    job_latencies = [job.finish_time - job.start_time for job in jobs]
    avg_job_latency = sum(job_latencies)/len(job_latencies)

    sim_data_headings = ['job start time', 'job finish time', 'num tasks in job', 'job latency']
    sim_data = [[job.start_time, job.finish_time, len(job.tasks), job.finish_time - job.start_time] for job in jobs]
    with open('sim-data.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(sim_data_headings)
        writer.writerows(sim_data)

    job_finish_times = [job.finish_time for job in jobs]
    job_start_times = [job.start_time for job in jobs]
    avg_time_per_job = (max(job_finish_times) - min(job_start_times))/len(jobs)


    if POLICY == 'RoundRobin' or POLICY == 'FullRepetition':
        job_batches = [jobs[6*i:6*i+5] for i in range(6000)]
        batch_times = [max(job.finish_time - job.start_time for job in batch) for batch in job_batches]
        avg_batched_latency = sum(batch_times)/len(batch_times)

        sim_data_headings = ['avg job latency', 'avg time per job', 'avg batched latency']
        sim_data = [f'{avg_job_latency}', f'{avg_time_per_job}', f'{avg_batched_latency}']
    elif POLICY == 'LatinSquare':
        sim_data_headings = ['avg job latency', 'avg time per job', 'avg task latency']
        sim_data = [f'{avg_job_latency}', f'{avg_time_per_job}', f'{avg_job_latency/LATIN_SQUARE_ORDER}']

    with open('avgs.csv', 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(sim_data_headings)
        writer.writerow(sim_data)
        
    # num_bins=50
    # histogram(job_latencies, num_bins)