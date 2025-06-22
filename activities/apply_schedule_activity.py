##Imports##
from temporalio import activity

#This activity simulates a more complex scheduler activity that would schedule the job to the appropriate cluster and return a respose
@activity.defn
async def apply_schedule_activity(jobs, cluster, action):
    #Extract the job ID to schedule and the node to schedule on from the action produced by the model
    job_idx, node_idx = action 

    #Extract the text values for the job and cluster. If the job or cluster do not exist return invlaid.
    job_id = jobs[job_idx]["job_id"] if job_idx < len(jobs) else "invalid" 
    node_id = cluster[node_idx]["node_id"] if node_idx < len(cluster) else "invalid"

    #Print the scheduling action taken and return it
    print(f"Scheduling {job_id} on {node_id}")
    return f"Scheduled {job_id} on {node_id}"