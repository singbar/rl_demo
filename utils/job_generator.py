#Imports
import random
from datetime import datetime, timedelta

#This function returns node status with CPU/GPU capacity avialable
def generate_cluster_state(debug=False):
    cluster = []
    
    #Create a simulated cluster state 
    for i in range(3):
        total_cpus = random.randint(400, 800)
        total_p5 = random.randint(2000, 2500)
        total_p4 = random.randint(750, 1200)
        
        used_cpus = random.randint(0, total_cpus)
        used_p5 = random.randint(0, total_p5)
        used_p4 = random.randint(0, total_p4)

        cluster.append({
            "node_id": f"node-{i+1}",
            "total_cpus": total_cpus,
            "total_p5": total_p5,
            "total_p4": total_p4,
            "available_cpus": total_cpus - used_cpus,
            "available_p5": total_p5 - used_p5,
            "available_p4": total_p4 - used_p4,
            "status": random.choice(["OK", "BUSY", "DOWN"]),
            "hardware": {
                "p5_perf": 1.0,      # top-end performance
                "p4_perf": 0.6       # slower multiplier
            }
        })

        if debug == True:
            print(cluster)

    return cluster

#This function creates jobs with randomized resouce needs and priorities
def generate_pending_jobs(debug=False):
    jobs = []
    current_time = datetime.now()
    n = random.randint(100,500)
    arrival_offset = timedelta(seconds=random.randint(0, 300))
    deadline_offset = timedelta(seconds=random.randint(900, 4000))

    
    for i in range(n):
        base_eta = random.randint(600, 1800)  # seconds on P5 (best-case)
        p4_perf = 0.6  # 40% slower
        gpu_type = random.choice(["p5", "either"])

        job = {
            "job_id": f"job-{i+1}",
            "cpu_required": random.randint(1,150),
            "gpu_required": random.randint(5, 1500),  # abstracted perf unit
            "priority": random.randint(1, 3),  # 1 = low, 3 = high
            "arrival_time": (current_time - arrival_offset).isoformat(timespec='seconds'),
            "deadline": (current_time + deadline_offset).isoformat(timespec='seconds'),
            "model_type": random.choice(["bert", "llama", "resnet", "gpt"]),
            "preferred_gpu": gpu_type,
            "eta_p5": base_eta,
            "eta_p4": int(base_eta / p4_perf) if gpu_type == "either" else None,
        }
        jobs.append(job)
        
    if debug == True:
        print("Generated pending jobs:")
        for job in jobs:
            print(job)

    return jobs



