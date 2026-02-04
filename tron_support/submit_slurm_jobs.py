"""
提交 Slurm 作业脚本(仅支持 DP+TP)
"""
from enum import Enum
import os
from jinja2 import Template
import subprocess
import json
from typing import List 

class Status(Enum):
    # INIT -> PENDING -> [RUNNING | FAIL | TIMEOUT | OOM] -> COMPLETED
    INIT = "init"           # Job is created
    PENDING = "pending"     # Job is waiting for resources
    RUNNING = "running"     # Job is running
    FAIL = "fail"           # Job failed
    OOM = "oom"             # Job failed due to out of memory
    TIMEOUT = "timeout"     # Job failed due to timeout
    COMPLETED = "completed" # Job is completed

class Job:
    def __init__(self, root_path: str, qos: str) -> None:
        self.root_path = root_path        
        self.name = os.path.basename(root_path)
        self.config = os.path.join(root_path, "config.json")
        self.qos = qos
        
        # Check if the status.txt file exists
        status_file_path = os.path.join(self.root_path, "status.txt")
        if not os.path.exists(status_file_path):
            # Create the status.txt file with INIT status
            with open(status_file_path, 'w') as f:
                f.write(Status.INIT.value)
        self.status = self.get_status()
        
    def get_status(self) -> Status:
        """Read the status of the job from `status.txt` and return it"""
        is_existing = lambda value_to_check: any(value.value == value_to_check for value in Status.__members__.values())

        status_file_path = os.path.join(self.root_path, "status.txt")
        with open(status_file_path, 'r') as f:
            status = f.read().strip()
            if not is_existing(status):
                raise ValueError(f"Invalid status: {status}")
            return Status(status)
        
    def set_status(self, status: Status) -> Status:
        """Update the status of the job in `status.txt` and return the new status"""
        status_file_path = os.path.join(self.root_path, "status.txt")
        with open(status_file_path, 'w') as f:
            f.write(status.value)
            return status        
    
class Scheduler:
    
    def __init__(self, inp_dir: str, qos: str) -> None:
        jobs_directory_paths = [os.path.abspath(root) for root, dirs, _ in os.walk(inp_dir) if not dirs]
        self.job_lists = [Job(job_path, qos) for job_path in jobs_directory_paths]

    def keep_only_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status == status]
    
    def filter_out_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status != status]
     
    def create_slurm_script(self, job: Job):
        """Create Slurm script for DP+TP configuration"""
        with open(job.config, 'r') as file:
            config = json.load(file)
        
        max_gpu_per_node = 8
        # Calculate world size (只有 TP 和 DP)
        world_size = config["distributed"]["tp_size"] * config["distributed"]["dp_size"]
        
        assert world_size <= max_gpu_per_node or world_size % max_gpu_per_node == 0, \
            f"World size {world_size} must be <= {max_gpu_per_node} or divisible by {max_gpu_per_node}"
        
        nodes = max(1, world_size // max_gpu_per_node)
        n_proc_per_node = min(max_gpu_per_node, world_size // nodes)
        
        assert nodes * n_proc_per_node == world_size, \
            f"nodes ({nodes}) * n_proc_per_node ({n_proc_per_node}) != world_size ({world_size})"
        
        context_bench = {
            'nodes': nodes,
            'n_proc_per_node': n_proc_per_node,
            'root_path': job.root_path,
            "config": job.config,
            "qos": job.qos,
        }
        
        base_path = os.path.join(os.getcwd(), "template/base_job.slurm")

        with open(base_path, 'r') as file:
            base_job_file = file.read()
        
        base_job_template = Template(base_job_file)
                
        # Write the rendered script to a new file
        output_file_path = os.path.join(job.root_path, "job.slurm")
        with open(output_file_path, 'w') as file:
            file.write(base_job_template.render(context_bench))

        print(f"Slurm script created at {output_file_path}")
    
    def launch_dependency(self, job_array: List[Job], env_vars):
        """Launch jobs with dependencies"""
        prev_job_id = None
        for job in job_array:
            if prev_job_id is None:         
                result = subprocess.run(
                    ["sbatch", '--parsable', os.path.join(job.root_path, "job.slurm")], 
                    env=env_vars, capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    ["sbatch", '--parsable', '--dependency=afterany:'+prev_job_id, 
                     os.path.join(job.root_path, "job.slurm")], 
                    env=env_vars, capture_output=True, text=True
                )
            job.set_status(Status.PENDING)
            prev_job_id = result.stdout.strip()
                    
    def check_status(self):
        """Check and display status of all jobs"""
        status_files = [os.path.join(job.root_path, "status.txt") for job in self.job_lists]
                
        status_counts = {
            "init": 0,
            "pending": 0,
            "running": 0,
            "fail": 0,
            "oom": 0,
            "timeout": 0,
            "completed": 0
        }
        
        for status_file in status_files:
            with open(status_file, 'r') as f:
                status = f.read().strip()
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    raise ValueError(f"Invalid status: {status}")

        total = sum(status_counts.values())
        
        # Print the status counts in a formatted table
        print(f"{'Status':<10} | {'Count':<6}")
        print(f"{'-'*10}-|-{'-'*6}")
        for status, count in status_counts.items():
            print(f"{status.capitalize():<10} | {count:<6}")
        
        print(f"{'-'*10}-|-{'-'*6}")
        print(f"{'Total':<10} | {total:<6}")

def submit_jobs(inp_dir, qos, hf_token, nb_slurm_array, only: str = None):
    """Submit jobs to Slurm cluster"""
    scheduler = Scheduler(inp_dir, qos)

    env_vars = os.environ.copy()
    env_vars["HUGGINGFACE_TOKEN"] = hf_token
    total_jobs = len(scheduler.job_lists)

    # Filter jobs based on status
    if only == "fail":
        scheduler.job_lists = scheduler.keep_only_jobs(Status.FAIL)
    elif only == "pending":
        scheduler.job_lists = scheduler.keep_only_jobs(Status.PENDING)
    elif only == "timeout":
        scheduler.job_lists = scheduler.keep_only_jobs(Status.TIMEOUT)
    elif only == "running":
        scheduler.job_lists = scheduler.keep_only_jobs(Status.RUNNING)
    
    if only is not None:
        filtered_jobs = len(scheduler.job_lists)
        if filtered_jobs == 0:
            print(f"No '{only}' jobs to resubmit")
            return
        print(f"Only {filtered_jobs}/{total_jobs} jobs with status '{only}' will be resubmitted")
    
    # Filter out completed jobs
    scheduler.job_lists = scheduler.filter_out_jobs(Status.COMPLETED)

    if nb_slurm_array > 0:
        # Use job dependencies
        base_jobs_per_array = len(scheduler.job_lists) // nb_slurm_array
        extra_jobs = len(scheduler.job_lists) % nb_slurm_array
        distribution = [base_jobs_per_array] * nb_slurm_array
        for i in range(extra_jobs):
            distribution[i] += 1
        
        start = 0
        
        for i, nb_jobs in enumerate(distribution):
            end = start + nb_jobs
            job_array = scheduler.job_lists[start:end]
            
            print(f"Launching job dependency array {i+1} with {nb_jobs} jobs")
            
            for job in job_array:
                scheduler.create_slurm_script(job)

            scheduler.launch_dependency(job_array, env_vars)
            
            start = end
    else:
        # Don't use job dependencies
        for job in scheduler.job_lists:
            scheduler.create_slurm_script(job)
            print(f"Submitting: {os.path.join(job.root_path, 'job.slurm')}")
            subprocess.run(["sbatch", os.path.join(job.root_path, "job.slurm")], env=env_vars)
            job.set_status(Status.PENDING)
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Submit jobs to the cluster')
    parser.add_argument('--inp_dir', type=str, required=True, help='Input directory containing the jobs')
    parser.add_argument('--qos', type=str, required=True, help='QOS of the jobs')
    parser.add_argument('--nb_slurm_array', type=int, default=0, help='Number of slurm arrays')
    parser.add_argument('--only', type=str, default=None, help='Filter the jobs to submit (fail/pending/timeout/running)')
    parser.add_argument('--hf_token', type=str, required=True, help='Huggingface token')

    args = parser.parse_args()
    submit_jobs(args.inp_dir, args.qos, args.hf_token, args.nb_slurm_array, only=args.only)
