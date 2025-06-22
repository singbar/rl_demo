import asyncio
from worker.worker import main

def launch_worker():
    asyncio.run(main()) 


if __name__=="__main__":
    launch_worker()