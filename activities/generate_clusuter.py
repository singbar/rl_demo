#Imports
from temporalio import activity
from utils.job_generator import generate_cluster_state
from utils.to_builtin import to_builtin

#This activity defines a starting cluster state simulated inference
@activity.defn
async def generate_cluster_activity(debug: bool = False): #Cluster generation logic is moved to a util so that it can be used by test scripts and test policies.
    return to_builtin(generate_cluster_state(debug=debug)) #The to_builtin function ensures that all dicts are serializable by Temporal