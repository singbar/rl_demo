#Imports
from temporalio import activity
from utils.job_generator import generate_pending_jobs
from utils.to_builtin import to_builtin

#This activity performs simple data generation for simulated inference
@activity.defn
async def generate_jobs_activity(debug: bool = False): #Job generation logic is moved to a util so that it can be used by test scripts and test policies.
    return to_builtin(generate_pending_jobs(debug=debug)) #The to_builtin function ensures that all dicts are serializable by Temporal
