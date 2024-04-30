# Library imports
import copy
import itertools
import math
import networkx as nx
import numpy as np
import random
import re
import sympy as sym

from collections import defaultdict, namedtuple, Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import reduce, cmp_to_key, cache, partial
from matplotlib.path import Path
from queue import PriorityQueue

###
# Puzzle 1
###

## Part A

# Read the input data
with open("input1.txt", "r") as f:
    content = f.readlines()


# Utility function to find digits within the input string
def find_digit(content):
    first_dig = None
    last_dig = None

    # Process each character individually
    for char in content:
        # We want to know if it is a digit
        if char.isdigit():
            if first_dig is None:
                first_dig = char
            last_dig = char

    return int(first_dig + last_dig)


# Perform it on all lines
all_results = [find_digit(line) for line in content]

# Total up the results
total = sum(all_results)

## Answer
print(f"Puzzle 1 Part A: {total}")


## Part B

# Get the digits from 1 to 9
numbers_to_consider = [str(i) for i in range(1, 10)]

# Dictionary mapping numbers to words
number_to_word = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

# Creating the reverse dictionary
word_to_number = {v: k for k, v in number_to_word.items()}

# Using list comprehension to spell out the first 10 digits
spelled_out_numbers = [number_to_word[i] for i in range(1, 10)]

# Bind the two dictionaries together
full_list = numbers_to_consider + spelled_out_numbers


# Utility function to find spelled out numbers
def find_spelled_out_numbers(i, s, numbers):
    found = []

    # Process every token individually
    for i in range(len(s)):
        for number in numbers:
            # Check if the number is in the string
            if s.startswith(number, i):
                if number in word_to_number:
                    number = word_to_number[number]
                found.append((i, number))
    return found


## Now let's go through all the results
nums = [find_spelled_out_numbers(i, s, full_list) for i, s in enumerate(content)]
nums_filt = [(num[0], num[len(num) - 1]) for num in nums]
nums_add = [int(str(num[0][1]) + str(num[1][1])) for num in nums_filt]

# Produce the sum of the results
total = sum(nums_add)

## Answer
print(f"Puzzle 1 Part B: {total}")


###
# Puzzle 2
###

## Parts A and B

# Read the new input data
with open("input2.txt", "r") as f:
    content = f.readlines()

# Split on the colon and semicolon delimiters
content2 = [x.split(": ")[1] for x in content]
content3 = [x.split("; ") for x in content2]

# Set the limits of the number of cubes for each color
game_map = {"red": 12, "green": 13, "blue": 14}
pattern = "([0-9]+) (blue|green|red)"

# Loop through the games and find the impossible ones
impossible = []
all_mins = []
for i, game in enumerate(content3):
    # Tabulate every sample for this game
    res = [re.findall(pattern, x) for x in game]
    res2 = [item for sublist in res for item in sublist]

    # For each game, track the possible minimums
    mins = {"red": -1, "green": -1, "blue": -1}

    # Now loop over each and set as appropriate
    for r in res2:
        # Print games which exceed the specified limit
        if game_map[r[1]] < int(r[0]):
            impossible.append(i + 1)

        # Print games where the minimum is exceeded
        if mins[r[1]] < int(r[0]):
            mins[r[1]] = int(r[0])

    all_mins.append(mins)

# Remove any duplicates
impossible = list(set(impossible))

# These are the ones that are possible
print(f"Puzzle 2 Part A: {sum(range(1, len(content3) + 1)) - sum(impossible)}")

# Compute the powers of each
powers = [x["red"] * x["blue"] * x["green"] for x in all_mins]

# Print the answer to Part B
print(f"Puzzle 2 Part B: {sum(powers)}")

###
# Puzzle 3
###

## Part A

# Read in the input data
with open("input3.txt", "r") as f:
    content = f.readlines()

# Split on the new line, convert each token to a character
content_clean = [x.split("\n")[0] for x in content]
content_clean2 = [list(x) for x in content_clean]

# Find positions of the symbols
symbol_pos = []
num_pos = []
id = -1
for i in range(len(content_clean2)):
    first_digit = None

    # Loop row-wise over all the input
    for j in range(len(content_clean2[i])):
        char = content_clean2[i][j]

        # Case 1: We have a symbol
        if not char.isdigit() and char != ".":
            symbol_pos.append((i, j))
            first_digit = None

        # Case 2: We have a digit, and it's the start of a new number
        elif char.isdigit() and first_digit is None:
            id = id + 1
            first_digit = char
            num_pos.append((i, j, id))

        # Case 3: We have a digit, and it's not the start of a new number
        elif char.isdigit():
            num_pos.append((i, j, id))

        # Case 4: We have the "." character
        else:
            first_digit = None

valid_ids = []
gear_ids = []

# Loop over the positions of the numbers and symbols
for i, j, id in num_pos:
    for i2, j2 in symbol_pos:
        # Check if the symbol is at most one coord away in both directions
        if abs(i - i2) <= 1 and abs(j - j2) <= 1:
            valid_ids.append(id)
            gear_ids.append((str(i2) + str(j2), id))

# Remove duplicates
valid_ids = list(set(valid_ids))

# Creating a dictionary to store tuples based on the last integer
groups = {}
for tup in num_pos:
    key = tup[-1]  # Get the last element of the tuple
    if key not in groups:
        groups[key] = []
    groups[key].append(tup)

# Extracting lists from the dictionary
grouped_lists = list(groups.values())

str_list = []
full_str_list = []

# For every pair of numbers, concatenate them into a string
for lst in grouped_lists:
    my_str = ""
    for chr in lst:
        my_str = my_str + str(content_clean2[chr[0]][chr[1]])

    full_str_list.append(my_str)

    # Only add to the list if the number is valid
    if lst[0][2] not in valid_ids:
        continue

    str_list.append(int(my_str))

print(f"Puzzle 3 Part A: {sum(str_list)}")

# Creating a dictionary to store tuples based on the last integer
groups_gears = {}
for tup in gear_ids:
    key = tup[0]  # Get the last element of the tuple
    if key not in groups_gears:
        groups_gears[key] = []
    if tup[1] not in groups_gears[key]:
        groups_gears[key].append(tup[1])

# Check the groups_gears
gears_list = []
for gid, id in groups_gears.items():
    if len(id) == 2:
        gears_list.append(id)

# Now compute the power ratios
ratios = []
for glist in gears_list:
    ratios.append(int(full_str_list[glist[0]]) * int(full_str_list[glist[1]]))

print(f"Puzzle 3 Part B: {sum(ratios)}")

###
# Puzzle 4
###

## Part A

# Read the input data
with open("input4.txt", "r") as f:
    content = f.readlines()

# Split on the new line, convert each token to a character
content_clean = [x.split("\n")[0] for x in content]
content_clean2 = [x.split(": ")[1] for x in content_clean]
content_clean3 = [x.split(" | ") for x in content_clean2]
content_clean4 = [[x.split(" ") for x in y] for y in content_clean3]
content_clean5 = [
    [[item for item in inner_list if item] for inner_list in outer_list]
    for outer_list in content_clean4
]

# Store a map of all the matches by card
matches_map = {}

# Now time to score
scores = []

for i, game in enumerate(content_clean5):
    # The numbers + winning numbers form the matches
    my_numbers = game[0]
    winning_numbers = game[1]

    # Do a set match
    matches = list(set(my_numbers) & set(winning_numbers))

    if i not in matches_map:
        matches_map[i] = list(range(i + 1, i + 1 + len(matches)))

    # Append scores if greater than 0 matches
    if len(matches) > 0:
        scores.append(2 ** (len(matches) - 1))

print(f"Puzzle 4 Part A: {sum(scores)}")


## Part B


# Utility function to recursively process cards
def recursive_card_process(card, matches_map):
    # Check if the card is in the map
    if card not in matches_map or len(matches_map[card]) == 0:
        return 0

    # Get the matches for a given card
    matches = matches_map[card]

    # Count them, and recursively process every single one
    return len(matches) + sum(
        [recursive_card_process(match, matches_map) for match in matches]
    )


# Add the original count of cards to the result of the recursive processing
final_cards = len(matches_map) + sum(
    [recursive_card_process(i, matches_map) for i in matches_map.keys()]
)

print(f"Puzzle 4 Part B: {final_cards}")

###
# Puzzle 5
###

## Part A

# Read the input data
with open("input5.txt", "r") as f:
    content = f.readlines()

# Get the actual seed values
seeds = content[0].split(": ")[1].strip().split(" ")

# Find the locations of the new lines
newline_locs = [ind for ind, x in enumerate(content) if x == "\n"]

# Grab all the data in the puzzle
seed_to_soil = [
    x.strip().split(" ") for x in content[(newline_locs[0] + 2) : newline_locs[1]]
]
soil_to_fertilizer = [
    x.strip().split(" ") for x in content[(newline_locs[1] + 2) : newline_locs[2]]
]
fertilizer_to_water = [
    x.strip().split(" ") for x in content[(newline_locs[2] + 2) : newline_locs[3]]
]
water_to_light = [
    x.strip().split(" ") for x in content[(newline_locs[3] + 2) : newline_locs[4]]
]
light_to_temperature = [
    x.strip().split(" ") for x in content[(newline_locs[4] + 2) : newline_locs[5]]
]
temperature_to_humidity = [
    x.strip().split(" ") for x in content[(newline_locs[5] + 2) : newline_locs[6]]
]
humidity_to_location = [
    x.strip().split(" ") for x in content[(newline_locs[6] + 2) : len(content)]
]


# Primary utility function which takes a set of rules and builds a dictionary
def process_map(rules, flip=False):
    rel_ind = 1 - int(flip)

    class MyCustomDict(dict):
        def __init__(self, rules, *args, **kwargs):
            self.rules = rules
            super().__init__(*args, **kwargs)

        def __missing__(self, key):
            for rule in rules:
                if int(rule[rel_ind]) <= int(key) < int(rule[rel_ind]) + int(rule[2]):
                    return int(key) + int(rule[1 - rel_ind]) - int(rule[rel_ind])

            return int(key)

    # Create a defaultdict where the default value is the key itself
    my_custom_dict = MyCustomDict(rules)

    return my_custom_dict


# Build dictionaries for all mappings
sts_map = process_map(seed_to_soil)
stf_map = process_map(soil_to_fertilizer)
ftw_map = process_map(fertilizer_to_water)
wtl_map = process_map(water_to_light)
ltt_map = process_map(light_to_temperature)
tth_map = process_map(temperature_to_humidity)
htl_map = process_map(humidity_to_location)

# Apply to all seeds
all_locs = [
    htl_map[tth_map[ltt_map[wtl_map[ftw_map[stf_map[sts_map[int(s)]]]]]]] for s in seeds
]

print(f"Puzzle 5 Part A: {min(all_locs)}")


## Part B

# Now build the reverse dictionaries
sts_map_rev = process_map(seed_to_soil, flip=True)
stf_map_rev = process_map(soil_to_fertilizer, flip=True)
ftw_map_rev = process_map(fertilizer_to_water, flip=True)
wtl_map_rev = process_map(water_to_light, flip=True)
ltt_map_rev = process_map(light_to_temperature, flip=True)
tth_map_rev = process_map(temperature_to_humidity, flip=True)
htl_map_rev = process_map(humidity_to_location, flip=True)


# Utility function that, for a given seed value, checks if its in the range of any of the seed rules
def in_seed_range(seed_value, seed_rules):
    for rule in seed_rules:
        if int(rule[0]) <= int(seed_value) < int(rule[0]) + int(rule[1]):
            return True

    return False


# Get the seed rules
seed_rules = [[seeds[i], seeds[i + 1]] for i in range(0, len(seeds), 2)]

# Now we brute force:
i = 0
while True:
    # Get the seed value
    seed_val = sts_map_rev[
        stf_map_rev[ftw_map_rev[wtl_map_rev[ltt_map_rev[tth_map_rev[htl_map_rev[i]]]]]]
    ]

    # If it's in the range... we're done!
    if in_seed_range(seed_val, seed_rules):
        print(f"Puzzle 5 Part B: {i}")
        break

    # Otherwise, increment and try again
    i += 1

###
# Puzzle 6
###

## Part A

# Read the input data
with open("input6.txt", "r") as f:
    content = f.readlines()

# Get the race data
race_data = [x.split()[1:] for x in content]
times = race_data[0]
distances = race_data[1]


# Utility function which finds the winning ways for a given time
def find_winning_ways(time, distance):
    distances_traveled = []

    # Try each speed possible
    for speed in range(1, time):
        # We are driving for the remaining time
        time_driving = time - speed

        # It is a possible record if the time driving times the speed exceeds the distance
        if time_driving * speed > distance:
            distances_traveled.append(time_driving * speed)

    return distances_traveled


# Process each race
race_results = []
for i in range(len(times)):
    race_results.append(len(find_winning_ways(int(times[i]), int(distances[i]))))

# Multiply the results together
result = reduce(lambda x, y: x * y, race_results)
print(f"Puzzle 6 Part A: {result}")


## Part B

# Get the new values by merging the strings
new_time = int(reduce(lambda x, y: str(x) + str(y), times))
new_distance = int(reduce(lambda x, y: str(x) + str(y), distances))

# Call our utility function
distances_traveled = find_winning_ways(new_time, new_distance)

# Get the length
result = len(distances_traveled)
print(f"Puzzle 6 Part B: {result}")

###
# Puzzle 7
###

## Part A

# Read the input data
with open("input7.txt", "r") as f:
    content = f.readlines()

# Map possible results to scores / strength
strength_map = {
    "five_of_kind": 7,
    "four_of_kind": 6,
    "full_house": 5,
    "three_of_kind": 4,
    "two_pair": 3,
    "one_pair": 2,
    "high_card": 1,
}


# Utility function to score a hand
def score_hand(cards):
    # Five of a cind
    if len(set(cards)) == 1:
        return strength_map["five_of_kind"]
    elif len(set(cards)) == 2:
        # Four of a kind
        if cards.count(cards[0]) in [1, 4]:
            return strength_map["four_of_kind"]
        # Full house
        else:
            return strength_map["full_house"]
    elif len(set(cards)) == 3:
        # Check whether we have a three of a kind or two pair
        if cards.count(cards[0]) in [1, 3] and cards.count(cards[1]) in [1, 3]:
            return strength_map["three_of_kind"]
        else:
            return strength_map["two_pair"]
    elif len(set(cards)) == 4:
        return strength_map["one_pair"]
    else:
        return strength_map["high_card"]


# Define the card values for sorting
card_values = {
    "2": "02",
    "3": "03",
    "4": "04",
    "5": "05",
    "6": "06",
    "7": "07",
    "8": "08",
    "9": "09",
    "T": "10",
    "J": "11",
    "Q": "12",
    "K": "13",
    "A": "14",
}


# Utility function that compares two hands
def compare_hands(data, reverse=False):
    def compare(a, b):
        # Get each character of the string
        a_list = list(a)
        b_list = list(b)

        # Get the numerical values of each card
        a_num = [card_values[x] for x in a_list]
        b_num = [card_values[x] for x in b_list]

        # Combine them into one big number
        a_result = int(reduce(lambda x, y: x + y, a_num))
        b_result = int(reduce(lambda x, y: x + y, b_num))

        # Sort the number to determine which wins
        if a_result > b_result:
            return 1
        elif a_result == b_result:
            return 0
        else:
            return -1

    # Get a lambda key from this
    custom_sort = cmp_to_key(compare)

    return sorted(data, key=custom_sort, reverse=reverse)


# Get the rank order of the strength of hand scores
def rank_order(data):
    # Sort the data in ascending order
    sorted_data = compare_hands(data)

    # Create a dictionary to map each string to its rank
    rank_map = {}
    for i, item in enumerate(sorted_data):
        rank_map[item] = i + 1

    # Return the rank order of the original data
    return [rank_map[item] for item in data]


# Break any ties among groups of same strength
def break_ties(camel_scores):
    # Create a dictionary to group tuples by their second value
    grouped_data = defaultdict(list)
    for key, value in camel_scores:
        grouped_data[value].append((key, value))

    # Convert the dictionary values to lists of lists
    result = list(grouped_data.values())

    # Loop over each list set
    final_scores = []
    for res in result:
        if len(res) == 1:
            final_scores.append((res[0][0], res[0][1], 1))
        else:
            # Get the actual strings associated with each
            hands = [camel[x[0]][0] for x in res]
            ranks = rank_order(hands)

            # Append the rank order to the final scores
            for i, r in enumerate(ranks):
                final_scores.append((res[i][0], res[i][1], r))

    return final_scores


# Utility function to process scores into ranks
def process_scores(camel_scores):
    # Sort the scores in ascending order
    camel_scores_sorted = sorted(camel_scores, key=lambda x: x[1])
    final_results = []
    prev_value = -1
    cur_rank = 0

    # For every sorted score, we check if we have ties
    for css in camel_scores_sorted:
        if css[1] == prev_value:
            cur_rank = cur_rank - 1

        # Rank it
        prev_value = css[1]
        cur_rank += 1

        # Append the rank
        final_results.append((css[0], cur_rank))

    return final_results


# Call the scoring procedure
camel = [x.strip().split() for x in content]
camel_scores = [(i, score_hand(list(x[0]))) for i, x in enumerate(camel)]
camel_scoreranks = process_scores(camel_scores)
camel_fullscores = break_ties(camel_scoreranks)

# Now, let's sort the final list
camel_sorted = sorted(camel_fullscores, key=lambda x: (x[1], x[2]))

# Final winnings
winnings = []
for i, result in enumerate(camel_sorted):
    winnings.append((i + 1) * int(camel[result[0]][1]))

print(f"Puzzle 7 Part A: {sum(winnings)}")


## Part B

# Set the new value of Joker to be 1
card_values["J"] = "01"


# Score hands by trying joker values
def score_hand_joker(cards):
    cards_set = list(set(cards))

    # If there's no joker, same score as before
    if "J" not in cards_set:
        return score_hand(cards)

    # If the hand is ALL jokers, return best score possible
    if "J" in cards_set and len(cards_set) == 1:
        return strength_map["five_of_kind"]

    # Build a list of all possible hands with jokers
    possible_cards = [cards]
    for card in cards_set:
        if card != "J":
            # Get a new possible set
            new_cards = [card if x == "J" else x for x in cards]
            possible_cards.append(new_cards)

    # Score them all and take the highest score
    all_scores = [score_hand(cards) for cards in possible_cards]

    return max(all_scores)


camel_scores = [(i, score_hand_joker(list(x[0]))) for i, x in enumerate(camel)]
camel_scoreranks = process_scores(camel_scores)
camel_fullscores = break_ties(camel_scoreranks)

# Now, let's sort the final list
camel_sorted = sorted(camel_fullscores, key=lambda x: (x[1], x[2]))

# Final winnings
winnings = []
for i, result in enumerate(camel_sorted):
    winnings.append((i + 1) * int(camel[result[0]][1]))

print(f"Puzzle 7 Part B: {sum(winnings)}")

###
# Puzzle 8
###

## Part A

# Read the input data
with open("input8.txt", "r") as f:
    content = f.readlines()


def get_node_mapping(line):
    # Splitting the line at '=' and stripping any whitespace
    key, value = [x.strip() for x in line.split("=")]

    # Removing parentheses and splitting the value part at ','
    values = value.strip("()").split(", ")

    # Creating the desired dictionary structure
    result = {key: {"L": values[0], "R": values[1]}}

    return result


# Read the directions to take
directions = list(content[0].strip())

# Get the node mappings
node_mappings_raw = [get_node_mapping(line) for line in content[2:]]
node_mappings = reduce(lambda a, b: {**a, **b}, node_mappings_raw)

# Begin the traversal
steps = 0

cur_node = "AAA"
while True:
    if cur_node == "ZZZ":
        print(f"Puzzle 8 Part A: {steps}")
        break

    # Get the direction that we are taking
    direction = directions[steps % len(directions)]

    # Get the node we are going to
    cur_node = node_mappings[cur_node][direction]

    # Increment the number of steps
    steps += 1

## Part B

# Track the paths simultaneously
cur_nodes = [x for x in node_mappings.keys() if x.endswith("A")]
cur_result = [{"cur_node": node, "steps": 0, "done": False} for node in cur_nodes]

# Flag for whether we are still lost
lost = True
dir_string = ""
while lost:
    # Process each path
    for i, node_data in enumerate(cur_result):
        # No need to process anything if we've found the shortest path
        if cur_result[i]["done"]:
            continue

        # Get the direction that we are taking
        direction = directions[node_data["steps"] % len(directions)]

        # Get where we're going next
        next_node = node_mappings[node_data["cur_node"]][direction]

        # Step there!
        cur_result[i] = {
            "cur_node": next_node,
            "steps": node_data["steps"] + 1,
            "done": next_node.endswith("Z"),
        }

        # End condition: all nodes took a step, all are on an end node
        if all([x["done"] for x in cur_result]):
            lost = False
            break


# Find the LCM of two numbers
def lcm_of_two_numbers(a, b):
    return abs(a * b) // math.gcd(a, b)


# Generalize to many numbers
def lcm(numbers):
    return reduce(lcm_of_two_numbers, numbers)


# Print the result
print(f"Puzzle 8 Part B: {lcm([x['steps'] for x in cur_result])}")

###
# Puzzle 9
###

## Part A

# Read the input data
with open("input9.txt", "r") as f:
    content = f.readlines()

# Clean it up
content = [x.strip() for x in content]
content_split = [[int(y) for y in x.split(" ")] for x in content]

# Process each report one by one
final_scores = []
final_scores_front = []
for report in content_split:
    report_differences = [report]
    while True:
        # Get the new line of interest
        cur_report = report_differences[len(report_differences) - 1]

        report_difference = []
        # Loop to the end, computing the differences between each number
        for i in range(len(cur_report) - 1):
            report_difference.append(cur_report[i + 1] - cur_report[i])

        # Append to our list
        report_differences.append(report_difference)

        # Check if all report differences are zero
        if all([x == 0 for x in report_difference]):
            break

    # Now we process what the next value is:
    final_value = 0
    last_value = 0

    # And do the same for the front of the sequence
    final_value_front = 0
    last_value_front = 0
    for i in range(len(report_differences) - 2, 0, -1):
        # Part A
        last_value = last_value + report_differences[i][len(report_differences[i]) - 1]
        final_value = (
            last_value + report_differences[i - 1][len(report_differences[i - 1]) - 1]
        )

        # Part B
        last_value_front = report_differences[i][0] - last_value_front
        final_value_front = report_differences[i - 1][0] - last_value_front

    final_scores.append(final_value)
    final_scores_front.append(final_value_front)

print(f"Puzzle 9 Part A: {sum(final_scores)}")
print(f"Puzzle 9 Part B: {sum(final_scores_front)}")

###
# Puzzle 10
###

## Part A

# Read the input data
with open("input10.txt", "r") as f:
    content = f.readlines()

# Split into a list of lists
content_split = [list(x.strip()) for x in content]

# Find the starting position
starting_coord = (-1, -1)
for j, row in enumerate(content_split):
    for i, char in enumerate(row):
        # We found the start
        if char == "S":
            starting_coord = (i, j)
            break

# Build a map between directions and their corresponding coordinates
direction_map = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

pipe_map = {
    "N": ["|", "7", "F"],
    "S": ["|", "J", "L"],
    "E": ["-", "7", "J"],
    "W": ["-", "F", "L"],
}

pipe_map_rev = {
    "|": ["N", "S"],
    "-": ["E", "W"],
    "7": ["S", "W"],
    "F": ["S", "E"],
    "J": ["N", "W"],
    "L": ["N", "E"],
}

avoid_backwards_map = {"N": "S", "S": "N", "E": "W", "W": "E"}


# Utility function to take a single step
def take_step(current_pos, last_step=None):
    for direction, offset in direction_map.items():
        if (
            last_step in avoid_backwards_map
            and direction == avoid_backwards_map[last_step]
        ):
            continue

        # Get the character we are on, skip if it's not a valid character
        cur_char = content_split[current_pos[1]][current_pos[0]]
        if cur_char != "S" and direction not in pipe_map_rev[cur_char]:
            continue

        # Compute the new position
        new_pos = (current_pos[0] + offset[0], current_pos[1] + offset[1])

        # Check if the new position is valid
        if (
            new_pos[0] >= 0
            and new_pos[0] < len(content_split[0])
            and new_pos[1] >= 0
            and new_pos[1] < len(content_split)
        ):
            # Now we must check if the new position is a valid character
            if content_split[new_pos[1]][new_pos[0]] in pipe_map[direction]:
                return new_pos, direction

    return new_pos, None


my_polygon = [starting_coord]  # For part B later


# Perform the full game
def take_loop():
    # Initialize the current position
    current_pos = starting_coord
    last_step = None

    # Loop until we find the end
    steps = 0
    while True:
        # Take a step
        current_pos, last_step = take_step(current_pos, last_step)

        # Increment the number of steps
        steps += 1

        # Append to the polygon
        my_polygon.append(current_pos)

        # Check if we are at the end
        if last_step is None:
            break

    return current_pos, steps


# Print the puzzle result
print(f"Puzzle 10 Part A: {int(take_loop()[1] / 2)}")


## Part B

# Use matplotlib to create a polygon
p = Path(my_polygon)

# Track the points within the polygon
points_within = 0

# Process each point in the grid
for j in range(len(content_split)):
    for i in range(len(content_split[0])):
        # But now check if it contains the point
        if (i, j) not in my_polygon and p.contains_point((i, j)):
            points_within += 1

# Print the puzzle result
print(f"Puzzle 10 Part B: {points_within}")


###
# Puzzle 11
###

## Part A

# Read the input data
with open("input11.txt", "r") as f:
    content = f.readlines()

# Split into a list of lists
content_split = [list(x.strip()) for x in content]

# Get the rows in content_split that have no "#" characters
row_expand = [i for i, row in enumerate(content_split) if "#" not in row]

# Get the columns in content_split that have no "#" characters
col_expand = [i for i, col in enumerate(zip(*content_split)) if "#" not in col]

# Cosmic Expansion
new_universe = []

# Next, we go through every column in every row
for i in range(len(content_split)):
    # Get the row of relevance
    row = copy.deepcopy(content_split[i])

    # Loop over the elements of the row
    for j in range(len(row) - 1, -1, -1):
        # We are going to add a new column anywhere col_expand has an index
        if j in col_expand:
            row.insert(j, ".")

    # See if we need to expand the row
    if i in row_expand:
        new_universe.append(["."] * len(row))

    # Add to the new universe
    new_universe.append(row)

# Get a list of coordinates of all "#" characters in (i, j) format
coords = [
    (i, j)
    for i in range(len(new_universe))
    for j in range(len(new_universe[0]))
    if new_universe[i][j] == "#"
]

# Get the pairwise distance between all coordinates
distances = []
for i in range(len(coords)):
    for j in range(i + 1, len(coords)):
        distances.append(
            (
                coords[i],
                coords[j],
                abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]),
            )
        )

# Sum all the distances
print(f"Puzzle 11 Part A: {sum([x[2] for x in distances])}")


## Part B


# Utility distance function that takes into account the expansion
def universe_distance(coords, expansion=1000000):
    coord_dist = [
        sum((expansion, 1)[coord in coords] for coord in range(coord))
        for coord in coords
    ]

    return sum(abs(i - j) for i in coord_dist for j in coord_dist) // 2


# Get a list of coordinates in the original non-expanded universe
orig_coords = [
    (i, j)
    for i in range(len(content_split))
    for j in range(len(content_split[0]))
    if content_split[i][j] == "#"
]

# Split into lists of X and Y
all_xcoords = [coord[0] for coord in orig_coords]
all_ycoords = [coord[1] for coord in orig_coords]

# Compute the overall sum of distances given the 1,000,000 expansion
print(f"Puzzle 11 Part B: {sum(map(universe_distance, [all_xcoords, all_ycoords]))}")

###
# Puzzle 12
###

## Parts A and B

# Read the input data
with open("input12.txt", "r") as f:
    content = f.readlines()


# Split the input data with an optional expandion factor
def split_content(content, expansion_factor=1):
    content_split = []
    for line in content:
        # Split the line, and clean up newline characters
        line = line.strip()
        row, counts = line.split()

        # Part B: Optional expansion of the underlying row
        row = "?".join([row] * expansion_factor)
        counts = tuple(int(x) for x in counts.split(",")) * expansion_factor

        # Append the data
        content_split.append((row, counts))

    return content_split


@cache
def get_possible_configurations(row, counts):
    # First, we see if there are now more rows to handle
    if not row:
        return len(counts) == 0

    # Next, we see if there are no more counts to handle
    if not counts:
        return "#" not in row

    # Recursively processs the list
    final_value = 0
    if row[0] in ".?":
        possibilities = get_possible_configurations(row[1:], counts)
        final_value += possibilities

    # Now
    if (
        row[0] in "#?"
        and "." not in row[: counts[0]]
        and not counts[0] > len(row)
        and (counts[0] == len(row) or row[counts[0]] != "#")
    ):
        final_value += get_possible_configurations(row[counts[0] + 1 :], counts[1:])

    return final_value


# Get two sets of input data
content_split_a = split_content(content)
content_split_b = split_content(content, expansion_factor=5)

# Fancy mapping of function
final_score_a = list(map(lambda x: get_possible_configurations(*x), content_split_a))
final_score_b = list(map(lambda x: get_possible_configurations(*x), content_split_b))

print(f"Puzzle 12 Part A: {sum(final_score_a)}")
print(f"Puzzle 12 Part B: {sum(final_score_b)}")

###
# Puzzle 13
###

## Part A

# Read the input data
with open("input13.txt", "r") as f:
    content = f.readlines()


# Function to split the list
def split_on_newline(lst):
    result = []
    current = []

    for item in lst:
        if item == "\n":
            if current:  # Only add non-empty lists
                result.append(current)
            current = []
        else:
            current.append(item)

    if current:  # Add the last sublist if it's not empty
        result.append(current)

    return result


# Splitting the original list
split_lists = split_on_newline(content)

# Split on the empty list into a new set of lists
content_split = [[list(x.strip()) for x in y] for y in split_lists]

# Produce an inverted version of the puzzle
content_inverted = []
for puzzle in content_split:
    content_inverted.append([list(column) for column in zip(*puzzle)])

# Smudge map
smudge_map = {"#": ".", ".": "#"}


def find_split(my_content, smudge=False):
    # Start with index 1, compare 0 to 2:
    success_ind = None
    for i in range(1, len(my_content)):
        check_factor = 1
        while True:
            # If we are checking off the edge, break
            if i - check_factor < 0 or i + check_factor > len(my_content):
                break

            # Otherwise, compare the two
            side1 = copy.deepcopy(my_content[i - check_factor])
            side2 = copy.deepcopy(my_content[i + check_factor - 1])

            # Smudge or no smudge, if they match we can continue
            if side1 == side2:
                success_ind = i

            # Here is where we try to smudge... success_ind must be set.
            reset = side1 != side2
            if reset and smudge:
                for j in range(len(side1)):
                    side1_smudged = copy.deepcopy(my_content[i - check_factor])
                    side1_smudged[j] = smudge_map[side1[j]]

                    # Check if they are equal
                    if side1_smudged == side2:
                        reset = False
                        success_ind = i
                        break

            # If after all the smudging, no luck, we need to try a new value
            if reset:
                success_ind = None

            # If we didn't have a success, break
            if success_ind is None:
                break
            else:
                check_factor += 1

        # Did we find a successful index? We're done!
        if success_ind is not None:
            break

    return success_ind


# Process each puzzle
final_scores = []
for i, puzzle in enumerate(content_split):
    # Find the split
    fac = 100
    split_ind = find_split(puzzle)
    if not split_ind:
        fac = 1
        split_ind = find_split(content_inverted[i])

    # Get the score
    final_scores.append(fac * split_ind)

print(f"Puzzle 13 Part A: {sum(final_scores)}")


## Part B


def read_puzzle_input(file_path: str):
    def parse_grid(grid_str: str) -> np.ndarray:
        return np.array(
            [[cell == "#" for cell in line] for line in grid_str.split("\n")]
        )

    with open(file_path, "r") as file:
        return [parse_grid(grid) for grid in file.read().strip().split("\n\n")]


def find_symmetry_axis_score(grid: np.ndarray) -> int:
    height, width = grid.shape

    # Check for symmetry along vertical axis
    for i in range(1, width):
        margin = min(i, width - i)
        if (
            np.sum(np.fliplr(grid[:, i - margin : i]) ^ grid[:, i : i + margin]).sum()
            == 1
        ):
            return i

    # Check for symmetry along horizontal axis
    for i in range(1, height):
        margin = min(i, height - i)
        if (np.flipud(grid[i - margin : i, :]) ^ grid[i : i + margin, :]).sum() == 1:
            return i * 100

    raise ValueError("Symmetry axis not found")


content_split_b = read_puzzle_input("input13.txt")
print(
    f"Puzzle 13 Part B: {sum(map(partial(find_symmetry_axis_score), content_split_b))}"
)


###
# Puzzle 14
###

## Part A

# Read the input data
with open("input14.txt", "r") as f:
    content = f.readlines()

# Split on the empty list into a new set of lists
content_split = [list(x.strip()) for x in content]

# Produce an inverted version of the puzzle
content_inverted = [list(column) for column in zip(*content_split)]

# Now that we've inverted it, we're still trying to push rocks north
# In the new inverted framework, that would mean RIGHT TO LEFT.


def find_first_empty(sub_row):
    move = 0
    for i in range(len(sub_row)):
        if sub_row[i] == ".":
            move += 1
        else:
            return move

    return move


# Loop over each "row" (column)
new_content = []
for i, row in enumerate(content_inverted):
    new_row = copy.deepcopy(row)
    for j, rock in enumerate(row):
        # Only rock we can actually move
        if rock == "O":
            # Traverse backwards to find the spot for the rock
            sub_row = new_row[:j]
            sub_row.reverse()

            # Call our utility function
            first_empty = find_first_empty(sub_row)

            # If there is a
            if first_empty > 0:
                new_row[j - first_empty] = rock
                new_row[j] = "."

    new_content.append(new_row)

new_content_original = [list(column) for column in zip(*new_content)]


def calculate_load(puzzle):
    scores = []
    for i, row in enumerate(puzzle):
        scores.append((len(puzzle) - i) * row.count("O"))

    return scores


print(f"Puzzle 14 Part A: {sum(calculate_load(new_content_original))}")


## Part B


# Load the number of 'O's in the northern part of the grid.
def load_north(grid):
    return sum(
        len(grid) - y
        for x in range(len(grid[0]))
        for y in range(len(grid))
        if grid[y][x] == "O"
    )


# Initial grid setup.
grid = content_split


# Slide elements in the grid in a given direction (dx, dy).
def slide(grid, iterators, dx=None, dy=None):
    for x, y in iterators:
        # Continue sliding while within bounds and conditions are met.
        while (
            0 <= y + dy < len(grid)
            and 0 <= x + dx < len(grid[0])
            and grid[y][x] == "O"
            and grid[y + dy][x + dx] == "."
        ):
            grid[y][x], grid[y + dy][x + dx] = ".", "O"
            x, y = x + dx, y + dy
    return grid


# Slide functions for each direction.
def slide_north(grid):
    return slide(
        grid,
        ((x, y) for x in range(len(grid[0])) for y in range(len(grid))),
        dx=0,
        dy=-1,
    )


def slide_south(grid):
    return slide(
        grid,
        ((x, y) for x in range(len(grid[0])) for y in reversed(range(len(grid)))),
        dx=0,
        dy=1,
    )


def slide_west(grid):
    return slide(
        grid,
        ((x, y) for y in range(len(grid)) for x in range(len(grid[0]))),
        dx=-1,
        dy=0,
    )


def slide_east(grid):
    return slide(
        grid,
        ((x, y) for y in range(len(grid)) for x in reversed(range(len(grid[0])))),
        dx=1,
        dy=0,
    )


# Convert grid to a string representation.
def grid_to_string(grid):
    return "\n".join("".join(line) for line in grid)


# Cycle through each slide direction.
def cycle(grid):
    for func in [slide_north, slide_west, slide_south, slide_east]:
        func(grid)


# Track seen grid states and iteration counter.
seen_states = {}
iteration = 0
goal = 1000000000

# Loop to find repeat state or reach goal.
while iteration < goal:
    grid_state = grid_to_string(grid)

    if grid_state in seen_states:
        offset = seen_states[grid_state]
        break
    else:
        seen_states[grid_state] = iteration

    cycle(grid)

    iteration += 1

# Reverse mapping from iteration number to grid state.
iteration_to_state = {value: key for key, value in seen_states.items()}


# Lookup grid state for a given number of iterations.
def lookup_grid(iterations):
    if iterations in iteration_to_state:
        return iteration_to_state[iterations]

    cycle_length = iteration - offset
    index = ((iterations - offset) % cycle_length) + offset

    return iteration_to_state[index]


# Print the result for the second part.
print(f"Puzzle 14 Part B: {load_north(lookup_grid(goal).splitlines())}")


###
# Puzzle 15
###

## Part A

# Read the input data
with open("input15.txt", "r") as f:
    content = f.readlines()


# Utility function to determine the HASH of a string
def hash_string(s):
    # Start at zero, and split the string
    cur_value = 0
    s_list = list(s)

    # Get the value of the first character
    for char in s_list:
        cur_value += ord(char)
        cur_value *= 17
        cur_value = cur_value % 256

    return cur_value


# Split the content string on the comma
content_split = [x.strip() for x in content[0].split(",")]

# Apply the hash_string function to each string in content_split
hashes = [hash_string(x) for x in content_split]

# Print the answer of the puzzle
print(f"Puzzle 15 Part A: {sum(hashes)}")


## Part B
# Initialize an empty list to store boxes
boxes = []
for i in range(256):
    boxes.append([])

# Process each segment in the content split
for segment in content_split:
    # Handle segments ending with '-'
    if "-" in segment:
        new_string = segment[:-1]
        box_index = hash_string(new_string)

        # Skip if the box is empty
        if len(boxes[box_index]) == 0:
            continue

        # Remove matching string from the box
        for idx in reversed(range(len(boxes[box_index]))):
            original_string, lens_num = boxes[box_index][idx]

            if original_string == new_string:
                del boxes[box_index][idx]
                break

    # Handle segments with '='
    elif "=" in segment:
        new_string, new_lens = segment.split("=")
        new_lens = int(new_lens)

        # Hash the string to get the box index
        box_index = hash_string(new_string)

        should_add = True
        # Update existing string or mark to add new
        for idx, (original_string, lens_num) in enumerate(boxes[box_index]):
            if original_string == new_string:
                boxes[box_index][idx] = (new_string, new_lens)
                should_add = False

                break
        # Add new string if not found
        if should_add:
            boxes[box_index].append((new_string, new_lens))

# Calculate the final answer
final_result = 0
for box_idx, box in enumerate(boxes):
    for item_idx, item in enumerate(box):
        # Get the focus power
        _, lens_num = item
        focus_power = (box_idx + 1) * (item_idx + 1) * lens_num

        # Add to the score
        final_result += focus_power

# Print the Part B answer
print(f"Puzzle 15 Part B: {final_result}")


###
# Puzzle 16
###

## Part A

# Read and process input data
with open("input16.txt", "r") as file:
    puzzle_input = [list(line.strip()) for line in file]

# Directions constants
RIGHT, LEFT, UP, DOWN = (1, 0), (-1, 0), (0, -1), (0, 1)

# Mapping of tile types to possible moves
MOVE_OPTIONS = {
    (".", RIGHT): [RIGHT],
    (".", LEFT): [LEFT],
    (".", UP): [UP],
    (".", DOWN): [DOWN],
    ("-", RIGHT): [RIGHT],
    ("-", LEFT): [LEFT],
    ("-", UP): [LEFT, RIGHT],
    ("-", DOWN): [LEFT, RIGHT],
    ("|", RIGHT): [UP, DOWN],
    ("|", LEFT): [UP, DOWN],
    ("|", UP): [UP],
    ("|", DOWN): [DOWN],
    ("\\", RIGHT): [DOWN],
    ("\\", LEFT): [UP],
    ("\\", UP): [LEFT],
    ("\\", DOWN): [RIGHT],
    ("/", RIGHT): [UP],
    ("/", LEFT): [DOWN],
    ("/", UP): [RIGHT],
    ("/", DOWN): [LEFT],
}

# Initialize beam tracking
energized_tiles = defaultdict(set)
beams = [((-1, 0), RIGHT)]

# Process beams
while beams:
    beam_position, direction = beams.pop()
    x, y = beam_position[0] + direction[0], beam_position[1] + direction[1]

    # Check bounds
    if not (0 <= x < len(puzzle_input[0]) and 0 <= y < len(puzzle_input)):
        continue

    # Check if beam already passed
    if direction in energized_tiles[(x, y)]:
        continue

    energized_tiles[(x, y)].add(direction)

    # Add new directions for the beam to follow
    for new_direction in MOVE_OPTIONS[(puzzle_input[y][x], direction)]:
        beams.append(((x, y), new_direction))

# Print the Part A answer
print(f"Puzzle 16 Part A: {len(energized_tiles)}")


## Part B

# Initialize variables for Part B
max_energized_tiles = -1
beam_starts = [((-1, y), RIGHT) for y in range(len(puzzle_input))]
beam_starts += [((len(puzzle_input[0]), y), LEFT) for y in range(len(puzzle_input))]
beam_starts += [((x, -1), DOWN) for x in range(len(puzzle_input[0]))]
beam_starts += [((x, len(puzzle_input)), DOWN) for x in range(len(puzzle_input[0]))]

# Process each beam start
for beam_start in beam_starts:
    energized_tiles = defaultdict(set)
    beams = [beam_start]

    while beams:
        beam_position, direction = beams.pop()
        x, y = beam_position[0] + direction[0], beam_position[1] + direction[1]

        # Check bounds
        if not (0 <= x < len(puzzle_input[0]) and 0 <= y < len(puzzle_input)):
            continue

        # Check if beam already passed
        if direction in energized_tiles[(x, y)]:
            continue

        energized_tiles[(x, y)].add(direction)

        # Add new directions for the beam to follow
        for new_direction in MOVE_OPTIONS[(puzzle_input[y][x], direction)]:
            beams.append(((x, y), new_direction))

    max_energized_tiles = max(max_energized_tiles, len(energized_tiles))

# Print the Part B answer
print(f"Puzzle 16 Part B: {max_energized_tiles}")


###
# Puzzle 17
###

## Part A

# Read and process input data
with open("input17.txt", "r") as file:
    content_lines = file.readlines()

content_grid = [[int(char) for char in line.strip()] for line in content_lines]


class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)


class WalkerState:
    def __init__(
        self, cost: int, x: int, y: int, direction: Direction, step_counter: int
    ):
        self.cost = cost
        self.x = x
        self.y = y
        self.direction = direction
        self.step_counter = step_counter

    def __lt__(self, other):
        return (self.cost, self.x, self.y, self.direction.value, self.step_counter) < (
            other.cost,
            other.x,
            other.y,
            other.direction.value,
            other.step_counter,
        )

    def __eq__(self, other):
        return (self.cost, self.x, self.y, self.direction, self.step_counter) == (
            other.cost,
            other.x,
            other.y,
            other.direction,
            other.step_counter,
        )

    def __hash__(self):
        return hash((self.cost, self.x, self.y, self.direction, self.step_counter))

    def get_visited_info(self):
        return (self.x, self.y, self.direction, self.step_counter)


def dijkstra(graph: list[list[int]], max_steps: int) -> int:
    queue = PriorityQueue()

    queue.put(WalkerState(0, 0, 0, Direction.RIGHT, 1))
    queue.put(WalkerState(0, 0, 0, Direction.DOWN, 1))

    visited = set()

    while not queue.empty():
        current_state = queue.get()
        if current_state.get_visited_info() in visited:
            continue
        visited.add(current_state.get_visited_info())

        new_x, new_y = (
            current_state.x + current_state.direction.value[1],
            current_state.y + current_state.direction.value[0],
        )
        if not (0 <= new_x < len(graph[0]) and 0 <= new_y < len(graph)):
            continue

        new_cost = current_state.cost + graph[new_y][new_x]
        if (
            current_state.step_counter <= max_steps
            and new_x == len(graph[0]) - 1
            and new_y == len(graph) - 1
        ):
            return new_cost

        for new_direction in Direction:
            if (
                new_direction.value[0] + current_state.direction.value[0] == 0
                and new_direction.value[1] + current_state.direction.value[1] == 0
            ):
                continue

            new_step_counter = (
                current_state.step_counter + 1
                if new_direction == current_state.direction
                else 1
            )
            if new_step_counter > max_steps:
                continue

            queue.put(
                WalkerState(new_cost, new_x, new_y, new_direction, new_step_counter)
            )

    return -1


# Print the Part A answer
print(f"Puzzle 17 Part A: {dijkstra(content_grid, 3)}")


# Part B
def dijkstra_with_min_max_steps(
    graph: list[list[int]], min_steps: int, max_steps: int
) -> int:
    queue = PriorityQueue()

    queue.put(WalkerState(0, 0, 0, Direction.RIGHT, 1))
    queue.put(WalkerState(0, 0, 0, Direction.DOWN, 1))

    visited = set()

    while not queue.empty():
        current_state = queue.get()
        if current_state.get_visited_info() in visited:
            continue
        visited.add(current_state.get_visited_info())

        new_x, new_y = (
            current_state.x + current_state.direction.value[1],
            current_state.y + current_state.direction.value[0],
        )
        if not (0 <= new_x < len(graph[0]) and 0 <= new_y < len(graph)):
            continue

        new_cost = current_state.cost + graph[new_y][new_x]
        if (
            min_steps <= current_state.step_counter <= max_steps
            and new_x == len(graph[0]) - 1
            and new_y == len(graph) - 1
        ):
            return new_cost

        for new_direction in Direction:
            if (
                new_direction.value[0] + current_state.direction.value[0] == 0
                and new_direction.value[1] + current_state.direction.value[1] == 0
            ):
                continue

            new_step_counter = (
                current_state.step_counter + 1
                if new_direction == current_state.direction
                else 1
            )
            if new_step_counter > max_steps or (
                current_state.step_counter < min_steps
                and new_direction != current_state.direction
            ):
                continue

            queue.put(
                WalkerState(new_cost, new_x, new_y, new_direction, new_step_counter)
            )

    return -1


# Print the Part B answer
print(f"Puzzle 17 Part B: {dijkstra_with_min_max_steps(content_grid, 4, 10)}")


###
# Puzzle 18
###

## Part A

# Read and process input data
with open("input18.txt", "r") as file:
    content_lines = file.readlines()


@dataclass(frozen=True)
class Position:
    row: int
    col: int

    def __add__(self, other):
        return Position(row=self.row + other.row, col=self.col + other.col)

    def __mul__(self, other):
        return Position(row=self.row * other, col=self.col * other)


# Dictionary to map directions to their corresponding Position changes
directions = {
    "U": Position(-1, 0),
    "L": Position(0, -1),
    "D": Position(1, 0),
    "R": Position(0, 1),
}

# Initialize positions and boundaries for both parts
pos1, pos2 = Position(0, 0), Position(0, 0)
boundary1, boundary2 = [pos1], [pos2]
perimeter1, perimeter2 = 0, 0

# Process each line of the content
for line in content_lines:
    udlr, num, colour = line.strip().split()

    # Part 1
    direction = directions[udlr]
    pos1 += direction * int(num)
    boundary1.append(pos1)
    perimeter1 += int(num)

    # Part 2
    hex_code = colour[2:-1]
    num2 = int(hex_code[:5], base=16)
    udlr2 = {"0": "R", "1": "D", "2": "L", "3": "U"}[hex_code[-1]]
    direction2 = directions[udlr2]
    pos2 += direction2 * num2
    boundary2.append(pos2)
    perimeter2 += num2


# Function to calculate area using the shoelace formula
def shoelace(boundary):
    determinant = 0
    for i in range(len(boundary) - 1):
        p1, p2 = boundary[i], boundary[i + 1]
        determinant += p1.row * p2.col - p2.row * p1.col
    return abs(determinant // 2)


# Calculate areas for both parts
area1 = shoelace(boundary1) + perimeter1 // 2 + 1
print(f"Puzzle 18 Part A: {area1}")

area2 = shoelace(boundary2) + perimeter2 // 2 + 1
print(f"Puzzle 18 Part B: {area2}")


###
# Puzzle 19
###

## Part A

# Read and process input data
with open("input19.txt") as file:
    workflows_data, ratings_data = file.read().split("\n\n")

# Parsing workflows
workflows = {
    name: rules[:-1].split(",")
    for name, rules in (line.split("{") for line in workflows_data.splitlines())
}

# Parsing ratings
ratings = [line[1:-1].split(",") for line in ratings_data.splitlines()]

# Process and evaluate each rating
accepted_ratings = []
for rating in ratings:
    name = "in"
    x, m, a, s = 0, 0, 0, 0  # Initialize variables

    # Executing rating rules
    for command in rating:
        exec(command)

    # Workflow processing
    while name not in "AR":
        for rule in workflows[name][:-1]:
            condition, next_name = rule.split(":")
            if eval(condition):
                name = next_name
                break
        else:
            name = workflows[name][-1]

    # Append the accepted rating
    accepted_ratings.append(x + m + a + s if name == "A" else 0)

# Print the Part A Answer
print(f"Puzzle 19 Part A: {sum(accepted_ratings)}")


# Utility function to find all the rules
def find_rules(current, rules):
    if current == "A":
        all_rules.append(rules)
    elif current != "R":
        temp_rules = []

        for rule in workflows[current][:-1]:
            next_workflow = rule.split(":")[1]
            condition = rule.split(":")[0]

            # Process rule
            find_rules(next_workflow, rules + temp_rules + [condition])

            # Inverting conditions
            if condition[1] == "<":
                temp_rules.append(
                    condition.split("<")[0] + ">=" + condition.split("<")[1]
                )
            elif condition[1] == ">":
                temp_rules.append(
                    condition.split(">")[0] + "<=" + condition.split(">")[1]
                )

        # Default workflow
        next_workflow = workflows[current][-1]
        find_rules(next_workflow, rules + temp_rules)


all_rules = []
find_rules("in", [])

total_combinations = 0
for rule_set in all_rules:
    max_values = {char: 4001 for char in "xmas"}
    min_values = {char: 0 for char in "xmas"}

    # Evaluating rule set
    for rule in rule_set:
        # Handling for rules with '>=', '<=', '>', and '<'
        char = rule[0]
        operator = rule[1]
        if rule[2] == "=":  # Adjust for '>=', '<=' operators
            operator += "="
            value = int(rule[3:])
        else:
            value = int(rule[2:])

        if char in "xmas":
            if operator == "<":
                max_values[char] = min(max_values[char], value)
            elif operator == "<=":
                max_values[char] = min(max_values[char], value + 1)
            elif operator == ">":
                min_values[char] = max(min_values[char], value)
            elif operator == ">=":
                min_values[char] = max(min_values[char], value - 1)

    # Calculating possible combinations
    product = 1
    for key in max_values:
        product *= max_values[key] - min_values[key] - 1

    total_combinations += product

# Print the Part B Answer
print(f"Puzzle 19 Part B: {total_combinations}")


###
# Puzzle 20
###

## Part A


class Module:
    def __init__(self, name, kind, destinations):
        self.name = name
        self.kind = kind
        self.destinations = destinations
        self.flipflop = False
        self.conjunction = {}


def read_input():
    modules = {}
    with open("input20.txt") as file:
        for line in file:
            module_info, destinations_info = line.split("->")
            kind, name = re.match(r"(%|&)?(\w+)", module_info).groups()
            destinations = re.findall(r"\w+", destinations_info)
            modules[name] = Module(name, kind, destinations)

    for module in modules.values():
        module.destinations = [modules.get(dest, dest) for dest in module.destinations]

    return modules


def solve_one(modules):
    Signal = namedtuple("Signal", ("sender", "receiver", "pulse"))

    def reset_conjunctions():
        for module in modules.values():
            if module.kind == "&":
                for mod_in in modules.values():
                    if module in mod_in.destinations:
                        module.conjunction[mod_in] = False

    def press_button():
        signals = [Signal("button", modules["broadcaster"], False)]
        while signals:
            count_high = sum(signal.pulse for signal in signals)
            counts[True] += count_high
            counts[False] += len(signals) - count_high
            signals = handle_signals(signals)
        return counts

    def handle_signals(signals_in):
        signals_out = []
        for signal in signals_in:
            if isinstance((receiver := signal.receiver), str):
                continue

            match receiver.kind, signal.pulse:
                case None, _:
                    pulse = signal.pulse
                case "%", False:
                    pulse = receiver.flipflop = not receiver.flipflop
                case "&", _:
                    receiver.conjunction[signal.sender] = signal.pulse
                    pulse = not all(receiver.conjunction.values())
                case _:
                    continue

            signals_out.extend(
                Signal(receiver, dest, pulse) for dest in receiver.destinations
            )
        return signals_out

    reset_conjunctions()
    counts = Counter()
    for _ in range(1000):
        press_button()

    return math.prod(counts.values())


# Print the Part A Answer
print(f"Puzzle 20 Part A: {solve_one(read_input())}")


# Part B
def solve_two(modules):
    Signal = namedtuple("Signal", ("sender", "receiver", "pulse"))

    def reset_conjunctions():
        for module in modules.values():
            if module.kind == "&":
                for mod_in in modules.values():
                    if module in mod_in.destinations:
                        module.conjunction[mod_in] = False

    def press_button():
        signals = [Signal("button", modules["broadcaster"], False)]
        while signals:
            signals = handle_signals(signals)

    def handle_signals(signals_in):
        signals_out = []
        for signal in signals_in:
            if isinstance((receiver := signal.receiver), str):
                continue

            match receiver.kind, signal.pulse:
                case None, _:
                    pulse = signal.pulse
                case "%", False:
                    pulse = receiver.flipflop = not receiver.flipflop
                case "&", _:
                    receiver.conjunction[signal.sender] = signal.pulse
                    pulse = not all(receiver.conjunction.values())
                    if pulse and len(cycle_times[receiver]) < 2:
                        cycle_times[receiver].append(iteration)
                case _:
                    continue

            signals_out.extend(
                Signal(receiver, dest, pulse) for dest in receiver.destinations
            )
        return signals_out

    ultimate_module = "rx"
    penultimates = [
        module for module in modules.values() if ultimate_module in module.destinations
    ]
    assert len(penultimates) == 1, "Only a single penultimate module exists"
    penultimate = penultimates[0]
    assert penultimate.kind == "&", "The penultimate module is a conjunction"

    antepenultimates = [
        module for module in modules.values() if penultimate in module.destinations
    ]
    assert all(
        module.kind == "&" for module in antepenultimates
    ), "All antepenultimate modules are conjunctions"

    reset_conjunctions()
    cycle_times = defaultdict(list)
    for iteration in itertools.count(1):
        press_button()
        if all(len(cycle_times[module]) >= 2 for module in antepenultimates):
            break

    assert all(
        cycle_times[module][1] == 2 * cycle_times[module][0]
        for module in antepenultimates
    ), (
        "A high pulse is emitted for the second time by all antepenultimate modules "
        "after exactly twice the number of iterations it took for the first time"
    )

    return math.lcm(*(cycle_times[module][0] for module in antepenultimates))


# Print the Part B Answer
print(f"Puzzle 20 Part B: {solve_two(read_input())}")


###
# Puzzle 21
###

## Part A


# Read and process input data
def read_lines_to_start():
    lines = []
    start_position = None
    with open("input21.txt", "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            stripped_line = line.strip()
            lines.append(list(stripped_line))
            if "S" in stripped_line:
                start_position = (i, stripped_line.index("S"))

    return lines, start_position


def part_one(steps=64):
    lines, start = read_lines_to_start()
    height, width = len(lines), len(lines[0])

    next_positions = [start]
    for _ in range(steps):
        current_positions = deepcopy(next_positions)
        visited = set(deepcopy(next_positions))
        next_positions = []

        while current_positions:
            current_position = current_positions.pop(0)
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_y, new_x = current_position[0] + dy, current_position[1] + dx
                if (
                    0 <= new_y < height
                    and 0 <= new_x < width
                    and (new_y, new_x) not in visited
                    and lines[new_y][new_x] != "#"
                ):
                    visited.add((new_y, new_x))
                    next_positions.append((new_y, new_x))

    return len(next_positions)


# Print the Part A Answer
print(f"Puzzle 21 Part A: {part_one()}")


## Part B


def part_two():
    # Read the input data
    lines, start = read_lines_to_start()
    height, width = len(lines), len(lines[0])
    modulus = 26501365 % height

    seen_states = []

    # Directions for movement
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Run the simulation for 3 cycles
    for run in [modulus, modulus + height, modulus + height * 2]:
        next_positions = [start]
        for _ in range(run):
            current_positions = deepcopy(next_positions)
            visited = set(deepcopy(next_positions))
            next_positions = []

            while current_positions:
                current_position = current_positions.pop(0)

                for dy, dx in directions:
                    new_y, new_x = (current_position[0] + dy), (
                        current_position[1] + dx
                    )
                    if (
                        lines[new_y % height][new_x % width] != "#"
                        and (new_y, new_x) not in visited
                    ):
                        visited.add((new_y, new_x))
                        next_positions.append((new_y, new_x))

        seen_states.append(len(next_positions))

    # Solve the system of equations
    m, n = seen_states[1] - seen_states[0], seen_states[2] - seen_states[1]
    a = (n - m) // 2
    b = m - 3 * a
    c = seen_states[0] - b - a

    # Calculate the answer
    ceiling = math.ceil(26501365 / height)
    answer = a * ceiling**2 + b * ceiling + c

    return answer


# Print the Part B Answer
print(f"Puzzle 21 Part B: {part_two()}")


###
# Puzzle 22
###

## Parts A and B

with open("input22.txt", "r") as file:
    # Initialize variables
    grid = {}
    bricks_down = {}
    bricks_up = {}
    falling_bricks = {}

    # Brick list
    bricks = []
    p1 = p2 = 0
    data = [
        [[*map(int, y.split(","))] for y in x.split("~")]
        for x in file.read().splitlines()
    ]

    # Build the grid
    for brick_data in sorted(data, key=lambda x: min(x[0][2], x[1][2])):
        x_range, y_range, z_range = [
            list(range(*sorted([brick_data[0][i], brick_data[1][i] + 1])))
            for i in range(3)
        ]
        brick = set()

        # Find the lowest z value
        while (
            z_range
            and z_range[0] > 1
            and not any(
                (a, b, c - 1) in grid for a in x_range for b in y_range for c in z_range
            )
        ):
            z_range = [z_range[0] - 1] + z_range[:-1]

        # Add the brick to the grid
        current_brick = tuple(
            sorted({(a, b, c) for a in x_range for b in y_range for c in z_range})
        )
        bricks.append(current_brick)
        bricks_down[current_brick] = set()

        # Process bricks for falling check
        min_z = min(z for _, _, z in current_brick)
        for x, y, z in current_brick:
            grid[x, y, z] = current_brick
            if z == min_z and (x, y, z - 1) in grid:
                down_brick = grid[x, y, z - 1]
                bricks_down[current_brick].add(down_brick)
                bricks_up.setdefault(down_brick, set()).add(current_brick)

    # Solve the puzzle
    for brick in bricks:
        # Check if the brick is a falling brick
        upper_bricks = bricks_up.get(brick, [])
        if not upper_bricks or all(
            len(bricks_down.get(upper_brick, [])) > 1 for upper_brick in upper_bricks
        ):
            p1 += 1

        # Find the falling bricks
        queue = [brick]
        falling_bricks = {brick}
        while queue:
            current_brick = queue.pop()

            # Check if the brick is a falling brick
            for next_brick in bricks_up.get(current_brick, set()):
                if not bricks_down[next_brick] - falling_bricks:
                    falling_bricks.add(next_brick)
                    queue.append(next_brick)

        # Update count for Part B
        p2 += len(falling_bricks - {brick})

# Print the Part A Answer
print(f"Puzzle 22 Part A: {p1}")

# Print the Part B Answer
print(f"Puzzle 22 Part B: {p2}")


###
# Puzzle 22
###

## Part A

with open("input23.txt", "r") as file:
    content = file.readlines()

content_lines = [line.strip() for line in content]

# Initialize grid dimensions
num_rows, num_cols = len(content_lines), len(content_lines[0])
start, target = (0, 1), (num_rows - 1, num_cols - 2)

# Initialize directed and undirected graphs
directed_graph = nx.grid_2d_graph(num_rows, num_cols, create_using=nx.DiGraph)
undirected_graph = nx.grid_2d_graph(num_rows, num_cols)

# Directions represented by symbols
directions = {">": (0, -1), "<": (0, 1), "^": (1, 0), "v": (-1, 0)}

# Process grid to update graphs
for row_index, line in enumerate(content_lines):
    for col_index, char in enumerate(line):
        position = (row_index, col_index)

        if char == "#":
            # Remove walls from both graphs
            directed_graph.remove_node(position)
            undirected_graph.remove_node(position)
        elif delta := directions.get(char):
            delta_row, delta_col = delta
            directed_graph.remove_edge(
                position, (row_index + delta_row, col_index + delta_col)
            )

# Part A: Calculate longest simple path in the directed graph
part_a_answer = max(map(len, nx.all_simple_edge_paths(directed_graph, start, target)))

print(f"Puzzle 23 Part A: {part_a_answer}")

# Part B: Process undirected graph to simplify paths
nodes_with_two_neighbors = [
    node for node in undirected_graph.nodes if len(undirected_graph.edges(node)) == 2
]

for node in nodes_with_two_neighbors:
    neighbors = list(undirected_graph.neighbors(node))
    new_weight = sum(
        undirected_graph.edges[node, neighbor].get("d", 1) for neighbor in neighbors
    )

    # Add new edge with combined weight and remove the intermediate node
    undirected_graph.add_edge(neighbors[0], neighbors[1], d=new_weight)
    undirected_graph.remove_node(node)

# Part B: Calculate the maximum path weight in the undirected graph
part_b_answer = max(
    nx.classes.function.path_weight(undirected_graph, path, "d")
    for path in nx.all_simple_paths(undirected_graph, start, target)
)

print(f"Puzzle 23 Part B: {part_b_answer}")


###
# Puzzle 24
###

## Part A

# Reading and processing input data
with open("input24.txt", "r") as file:
    content = [
        [int(digit) for digit in re.findall("(-?\d+)", line)]
        for line in file.readlines()
    ]


def solve_part1(data):
    # Initialize variables
    valid_intersections = 0
    min_limit, max_limit = 200_000_000_000_000, 400_000_000_000_000

    # Process each pair of lines
    for line1, line2 in itertools.combinations(data, 2):
        xp1, yp1, _, xv1, yv1, _ = line1
        xp2, yp2, _, xv2, yv2, _ = line2

        # Skip parallel lines
        if yv1 * xv2 == yv2 * xv1:
            continue

        # Calculate intersection time
        t1 = (yv2 * (xp1 - xp2) - xv2 * (yp1 - yp2)) / (yv1 * xv2 - xv1 * yv2)
        t2 = (yv1 * (xp2 - xp1) - xv1 * (yp2 - yp1)) / (yv2 * xv1 - xv2 * yv1)

        # Skip invalid intersections
        if t1 < 0 or t2 < 0:
            continue

        # Calculate intersection coordinates
        ix = xp1 + t1 * xv1
        iy = yp1 + t1 * yv1

        # Check if intersection is within limits
        if min_limit <= ix <= max_limit and min_limit <= iy <= max_limit:
            valid_intersections += 1

    return valid_intersections


# Print the Part A Answer
print(f"Puzzle 24 Part A: {solve_part1(content)}")

## Part B


def solve_part2(pos_vel):
    # Randomly sample three items
    random_sample = random.sample(pos_vel, 3)

    # Define symbols for positions and velocities
    P0x, P0y, P0z = sym.symbols("P0x P0y P0z", real=True)
    V0x, V0y, V0z = sym.symbols("V0x V0y V0z", real=True)
    t1, t2, t3 = sym.symbols("t1 t2 t3", real=True)

    # Extract positions and velocities
    positions, velocities = zip(*random_sample)

    # Formulate equations
    equations = []
    for i, (pos, vel) in enumerate(zip(positions, velocities)):
        for coord, value, symbol in zip("xyz", pos, (P0x, P0y, P0z)):
            eq = sym.Eq(
                symbol
                + sym.symbols(f"V0{coord}", real=True)
                * sym.symbols(f"t{i+1}", real=True),
                value + vel["xyz".index(coord)] * sym.symbols(f"t{i+1}", real=True),
            )
            equations.append(eq)

    # Solve the system of equations
    solution = sym.solve(equations, [P0x, P0y, P0z, V0x, V0y, V0z, t1, t2, t3])

    return sum(solution[0][:3])


# Prepare position and velocity data
pos_vel_data = [
    ((line[0], line[1], line[2]), (line[3], line[4], line[5])) for line in content
]

# Print the Part B Answer
print(f"Puzzle 24 Part B: {solve_part2(pos_vel_data)}")


###
# Puzzle 25
###

## Part A

# Reading and processing input data
with open("input25.txt", "r") as file:
    content_lines = [line.strip() for line in file.readlines()]

# Initialize the network graph
network_graph = nx.Graph()

# Process each line of the content
for line in content_lines:
    node_name, connections_str = line.split(":")
    node_name = node_name.strip()
    connections = connections_str.strip().split()

    # Add the node and its connections to the graph
    for connection in connections:
        network_graph.add_edge(node_name, connection)

# Remove the minimum edge cut to separate the graph into two components
min_edge_cut = nx.minimum_edge_cut(network_graph)
network_graph.remove_edges_from(min_edge_cut)

# Find the connected components
connected_components = list(nx.connected_components(network_graph))

# Ensure there are exactly two components
if len(connected_components) == 2:
    first_component, second_component = connected_components
else:
    raise ValueError("The graph does not split into exactly two components.")

# Calculate the product of the sizes of the two components
part_a_answer = len(first_component) * len(second_component)

# Print the Part A Answer
print(f"Puzzle 25 Part A: {part_a_answer}")
