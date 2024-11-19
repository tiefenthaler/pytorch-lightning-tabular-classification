import os
import time
from datetime import datetime

import numpy as np
from IPython.display import clear_output


def keep_cumpute_alive_with_cpu_activity(duration_hours=1, sleep_time=30) -> None:
    """
    Keeps the compute instance (e.g. azure ml compute instance) alive by running a periodic CPU task for a specified duration.
    Shows real-time status updates, including start time, elapsed time, and remaining time.
    Designed to work well in Jupyter Notebook.
    """
    start_time = time.time()
    end_time = start_time + duration_hours * 3600  # convert hours to seconds
    init_status = (
        f"Keeping the instance alive for {duration_hours} hours with periodic CPU activity.\n"
        f"To stop the function, create an empty file named stop_signal.txt in the same directory.\n"
        f"Current working directory: {os.getcwd()}\n"
    )

    while time.time() < end_time:
        # Check if stop signal file exists
        if os.path.isfile("stop_signal.txt"):
            print("\nStop signal received. Exiting the loop.")
            break

        # Perform a small computation to generate CPU activity
        _ = np.random.rand(1000, 1000).dot(np.random.rand(1000, 1000))

        # Calculate timing information
        elapsed_time = time.time() - start_time
        remaining_time = end_time - time.time()
        status = (
            f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Elapsed Time: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec\n"
            f"Remaining Time: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec\n"
        )

        # Update the output dynamically
        clear_output(wait=True)  # Clear the current cell output
        print(init_status + status)

        # Sleep for n seconds before the next iteration
        time.sleep(sleep_time)

    print("\nFinished keeping the instance alive.")


if __name__ == "__main__":
    # Run for the desired duration (e.g., 1 hours)
    keep_cumpute_alive_with_cpu_activity(duration_hours=10)

    # NOTE: add the following lines to a notebook cell
    # if compute == "azure":
    #     nb_utils.keep_cumpute_alive_with_cpu_activity(duration_hours=10)
