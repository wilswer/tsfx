#!/usr/bin/env python
#
#  Copyright 2023 The original authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Based on https://github.com/gunnarmorling/1brc/blob/main/src/main/java/dev/morling/onebrc/CreateMeasurements.java

import os
import random
import sys
import time


def check_args(file_args):  # noqa
    """Sanity checks input and prints out usage if input is not a positive integer."""
    try:
        if len(file_args) != 2 or int(file_args[1]) <= 0:
            raise Exception()
    except:  # noqa
        print(
            "Usage: create_measurements.sh <positive integer number>",
        )
        print("        You can use underscore notation for large number of records.")
        print("        For example:  1_000_000_000 for one billion")
        exit()


def build_weather_station_name_list():  # noqa
    """Grabs the weather station names from example data provided in repo and dedups."""
    station_names = []
    with open("./test_data/weather_stations.csv") as file:
        file_contents = file.read()
    for station in file_contents.splitlines():
        if "#" in station:
            continue
        else:
            station_names.append(station.split(";")[0])
    return list(set(station_names))


def convert_bytes(num):  # noqa
    """Convert bytes to a human-readable format (e.g., KiB, MiB, GiB)."""
    for x in ["bytes", "KiB", "MiB", "GiB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)  # noqa
        num /= 1024.0


def format_elapsed_time(seconds):  # noqa
    """Format elapsed time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.3f} seconds"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)} minutes {int(seconds)} seconds"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if minutes == 0:
            return f"{int(hours)} hours {int(seconds)} seconds"
        else:
            return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"


def estimate_file_size(weather_station_names, num_rows_to_create):  # noqa
    """Estimate how large a file the test data will be."""
    total_name_bytes = sum(len(s.encode("utf-8")) for s in weather_station_names)
    avg_name_bytes = total_name_bytes / float(len(weather_station_names))

    avg_temp_bytes = 4.400200100050025

    # add 2 for separator and newline
    avg_line_length = avg_name_bytes + avg_temp_bytes + 2

    human_file_size = convert_bytes(num_rows_to_create * avg_line_length)

    return f"Estimated max file size is:  {human_file_size}."


def build_test_data(weather_station_names, num_rows_to_create):  # noqa
    """Generate and writes to file the requested length of test data."""
    start_time = time.time()
    coldest_temp = -99.9
    hottest_temp = 99.9
    station_names_10k_max = random.choices(weather_station_names, k=10_000)  # noqa
    batch_size = 10000
    chunks = num_rows_to_create // batch_size
    print("Building test data...")

    try:
        with open("./test_data/measurements.txt", "w") as file:
            progress = 0
            for chunk in range(chunks):
                batch = random.choices(station_names_10k_max, k=batch_size)  # noqa
                prepped_deviated_batch = "\n".join(
                    [
                        f"{station};{random.uniform(coldest_temp, hottest_temp):.1f}"  # noqa
                        for station in batch
                    ],
                )
                file.write(prepped_deviated_batch + "\n")

                # Update progress bar every 1%
                if (chunk + 1) * 100 // chunks != progress:
                    progress = (chunk + 1) * 100 // chunks
                    bars = "=" * (progress // 2)
                    sys.stdout.write(f"\r[{bars:<50}] {progress}%")
                    sys.stdout.flush()
        sys.stdout.write("\n")
    except Exception as e:
        print("Something went wrong. Printing error info and exiting...")
        print(e)
        exit()

    end_time = time.time()
    elapsed_time = end_time - start_time
    file_size = os.path.getsize("./test_data/measurements.txt")
    human_file_size = convert_bytes(file_size)

    print("Test data successfully written to 1brc/data/measurements.txt")
    print(f"Actual file size:  {human_file_size}")
    print(f"Elapsed time: {format_elapsed_time(elapsed_time)}")


def main() -> None:
    """Generate."""
    check_args(sys.argv)
    num_rows_to_create = int(sys.argv[1])
    weather_station_names = []
    weather_station_names = build_weather_station_name_list()
    print(estimate_file_size(weather_station_names, num_rows_to_create))
    build_test_data(weather_station_names, num_rows_to_create)
    print("Test data build complete.")


if __name__ == "__main__":
    main()
exit()
