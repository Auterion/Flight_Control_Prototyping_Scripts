import sys
from pyulog import ULog

import numpy as np


def get_ulg_data(log, topic_name, variable_name, instance=0):
    variable_data = np.array([])
    for elem in log.data_list:
        if elem.name == topic_name:
            if instance == elem.multi_id:
                variable_data = elem.data[variable_name]
                break

    return variable_data


def get_data(log_dir):
    log_name = log_dir.split("/")[-1]

    if "ulg" or "ulog" in log_name:
        log = ULog(log_dir)
        timestamps = get_ulg_data(log, "vehicle_local_position", "timestamp")
        positions = np.array(
            [
                get_ulg_data(log, "vehicle_local_position", "x"),
                get_ulg_data(log, "vehicle_local_position", "y"),
                get_ulg_data(log, "vehicle_local_position", "z"),
            ]
        )

    return timestamps, positions


def main():
    timestamps, positions = get_data(sys.argv[1])

    close_calls = np.empty((1, 0))
    close_calls_t = np.empty((1, 0))

    target = np.array([10, -50, -50])  # virtual target
    distance_threshold = 5  # meters to consider a landing successful
    print("Target defined as: ", target)
    print("Distance threshold defined as: ", distance_threshold, "m")

    best_result_dist = float("inf")
    best_result_t = 0

    print("")
    for i in range(len(timestamps) - 1):
        v = positions[:, i + 1] - positions[:, i]
        w = target - positions[:, i]
        t = np.dot(v, w) / np.dot(v, v)

        if t >= 0 and t <= 1:
            distance = np.linalg.norm(target - (positions[:, i] + t * v))

            if distance < distance_threshold:
                print(
                    "Close call detected with distance: ",
                    round(distance, 2),
                    "m at timestamp: ",
                    round(timestamps[i] / 1e6, 2),
                    "s",
                )
                close_calls = np.append(close_calls, [distance])
                close_calls_t = np.append(close_calls_t, [timestamps[i]])

                best_result_dist = min(best_result_dist, distance)
                best_result_t = (
                    timestamps[i] if best_result_dist == distance else best_result_t
                )

    print("")
    print("-- Summary --")
    print("Total number of close calls: ", len(close_calls))
    print(
        "Best result: \n\r Distance to target: ",
        round(best_result_dist, 2),
        "m at timestamp: ",
        round(best_result_t / 1e6),
        "s",
    )


if __name__ == "__main__":
    main()
