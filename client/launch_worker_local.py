# --- Launch Temporal Worker ---
# This script is the entry point for running a Temporal worker that handles both training and inference.
# It wraps the async `main()` function from `worker/worker.py` and starts the event loop.

# --- Imports ---
import asyncio                          # Asyncio is used to manage the event loop for async operations
from worker.worker import main         # Import the main async function that initializes and runs the worker

# --- Synchronous Wrapper ---
def launch_worker():
    """
    This function bridges the gap between synchronous startup and
    asynchronous Temporal worker execution. It runs the async `main()`
    using `asyncio.run()`, which handles event loop setup and teardown.
    """
    asyncio.run(main())  # Blocking call to start the Temporal worker logic

# --- Script Entry Point Guard ---
if __name__ == "__main__":
    # Prevents this code from running if the script is imported as a module.
    launch_worker()
