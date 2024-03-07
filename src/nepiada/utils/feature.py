def parse_label(label):
    data_str, _ = label.split("*")
    data = eval(data_str)
    return data

def calculate_ranges(data):
    n_agents = 9  # Number of agents
    ranges = []

    sum_ranges = [0, 0]  # To store the sum of ranges for X and Y
    valid_agents_count = 0  # To count agents with valid ranges

    for i in range(n_agents):
        update1 = data[i]
        update2 = data[i + n_agents] if i + n_agents < len(data) else None

        # Filter out None values
        valid_updates = [update for update in [update1, update2] if update is not None]

        # Calculate range if we have two valid updates, else 0
        if len(valid_updates) == 2:
            range_x = abs(valid_updates[1][0] - valid_updates[0][0])
            range_y = abs(valid_updates[1][1] - valid_updates[0][1])
            agent_range = (range_x, range_y)
            sum_ranges[0] += range_x
            sum_ranges[1] += range_y
            valid_agents_count += 1
        else:
            agent_range = (0, 0)

        ranges.append(agent_range)

    return ranges, sum_ranges, valid_agents_count

label = "[(31.843550936067935, 31.650120154472056), None, None, (47.83577866512343, 43.307726158687075), None, None, None, None, (28, 17), (32.307802064531614, 31.812566480415942), None, None, (48.04570670263629, 43.379770808798014), None, None, None, None, (28, 18)]*1"
data = parse_label(label)
ranges, sum_ranges, valid_agents_count = calculate_ranges(data)

for i, agent_range in enumerate(ranges):
    print(f"Agent {i+1} Range: X = {agent_range[0]}, Y = {agent_range[1]}")

# Calculate and print the average range
if valid_agents_count > 0:
    avg_range_x = sum_ranges[0] / valid_agents_count
    avg_range_y = sum_ranges[1] / valid_agents_count
    overall_avg_range = (avg_range_x + avg_range_y) / 2
    print(f"Average Range across valid agents: X = {avg_range_x}, Y = {avg_range_y}")
    print(f"Overall Average Range: {overall_avg_range}")
else:
    print("No valid agents with range data available.")
