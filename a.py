import random
from collections import defaultdict

class Lecture:
    def __init__(self, subject, instructor):
        self.subject = subject
        self.instructor = instructor

    def __str__(self):
        return f"{self.subject} ({self.instructor})"

def create_empty_timetable(days, slot_mapping, halls):
  timetable = {}
  for day in days:
    timetable[day] = {}
    
    for slot_name in slots[day]:
       slot = slot_mapping[slot_name]
       timetable[day][slot] = {hall: None for hall in halls}

  return timetable

def generate_random_timetable(days, slot_mapping, halls, subjects, instructors):
    timetable = create_empty_timetable(days, slot_mapping, halls)

    for day in days:
        for slot_name in slots[day]:
            slot = slot_mapping[slot_name]
            for hall in halls:
                subject = random.choice(subjects)
                instructor = random.choice(instructors[subject])
                timetable[day][slot][hall] = Lecture(subject, instructor)

    return timetable

def calculate_fitness(timetable, seating_capacity, instructors):
    clash_penalty = 10  # A penalty for each clash
    preference_bonus = 1  # A bonus for each instructor's preferred slot

    fitness = 0
    total_clashes = 0
    total_utilization = 0
    instructor_preferences = defaultdict(int)

    for day in timetable:
        for slot in timetable[day]:
            for hall in timetable[day][slot]:
                lecture = timetable[day][slot][hall]

                if lecture is not None:
                    total_utilization += 1
                    instructor_preferences[lecture.instructor] += 1

                    # Check for clashes within the same hall
                    clashes_in_hall = sum(1 for h in timetable[day][slot] if timetable[day][slot][h] is not None)
                    if clashes_in_hall > 1:
                        total_clashes += 1

                    # Check for clashes with other halls
                    for other_hall in halls:
                        if other_hall != hall and timetable[day][slot][other_hall] is not None:
                            if timetable[day][slot][other_hall].subject == lecture.subject:
                                total_clashes += 1

    # Calculate bonus for instructor preferences
    for instructor in instructors:
        if instructor in instructor_preferences:
            preferred_slots = instructors[instructor]
            for slot_name, slot_time in slot_mapping.items():
                if slot_time in preferred_slots:
                    # Check how many lectures of this instructor are scheduled in the preferred slots
                    instructor_count = instructor_preferences[instructor]
                    if instructor_count > 0:
                        fitness += preference_bonus * instructor_count

    # Calculate final fitness score
    fitness -= clash_penalty * total_clashes
    fitness += total_utilization  # This ensures we prioritize utilizing all available slots

    return fitness

# Assuming the `instructors` dictionary contains the preferred slots for each instructor
# Update the calculate_fitness function accordingly if needed

def tournament_selection(population, fitness_scores, tournament_size):
    """
    Perform tournament selection to select individuals for mating.

    Parameters:
        population (list): List of candidate solutions (chromosomes).
        fitness_scores (list): List of fitness scores corresponding to each individual in the population.
        tournament_size (int): The size of the tournament.

    Returns:
        list: List of selected individuals (chromosomes).
    """
    selected_individuals = []

    # Perform tournament selection until we have selected enough individuals
    while len(selected_individuals) < len(population):
        tournament_candidates = random.sample(range(len(population)), tournament_size)
        best_candidate = tournament_candidates[0]

        # Find the best candidate in the tournament
        for candidate in tournament_candidates:
            if fitness_scores[candidate] > fitness_scores[best_candidate]:
                best_candidate = candidate

        selected_individuals.append(population[best_candidate])

    return selected_individuals

import random

def one_point_crossover(parent1, parent2):
    """
    Perform one-point crossover between two parent chromosomes.

    Parameters:
        parent1 (dict): The first parent chromosome (timetable).
        parent2 (dict): The second parent chromosome (timetable).

    Returns:
        tuple: A tuple containing two offspring chromosomes created after crossover.
    """
    # Ensure that both parents have the same days and slots
    assert parent1.keys() == parent2.keys() # "Parent timetables must have the same days."

    # Choose a random day for crossover
    crossover_day = random.choice(list(parent1.keys()))

    # Get the common slots for the crossover day
    common_slots = set(slots[crossover_day]).intersection(parent1[crossover_day].keys())

    # Create new dictionaries for the offspring timetables
    offspring1 = {day: {} for day in days}
    offspring2 = {day: {} for day in days}

    # Perform crossover for the chosen day
    for day in days:
        if day == crossover_day:
            # For the crossover day, copy the lectures of the common slots from both parents
            for slot_name in common_slots:
                slot = slot_mapping[slot_name]
                offspring1[day][slot] = parent2[day][slot].copy()
                offspring2[day][slot] = parent1[day][slot].copy()
        else:
            # For other days, copy the lectures from the respective parents
            for slot_name in common_slots:
                slot = slot_mapping[slot_name]
                offspring1[day][slot] = parent1[day][slot].copy()
                offspring2[day][slot] = parent2[day][slot].copy()

    return offspring1, offspring2



def mutate(individual, mutation_rate):
    """
    Mutate the given individual with a certain mutation rate.

    Parameters:
        individual (list): The individual (chromosome) to be mutated.
        mutation_rate (float): The probability of mutation (usually a small value).

    Returns:
        list: The mutated individual.
    """
    mutated_individual = individual.copy()

    # Iterate through the individual and apply mutation with the given mutation rate
    for day in mutated_individual:
        for slot in mutated_individual[day]:
            for hall in mutated_individual[day][slot]:
                # Randomly decide whether to mutate or not
                if random.random() < mutation_rate:
                    # Find a different day, slot, and hall to swap with
                    other_day = random.choice(days)
                    other_slot = random.choice(slots[other_day])
                    other_hall = random.choice(halls)

                    # Swap lectures between the two slots
                    temp_lecture = mutated_individual[day][slot][hall]
                    mutated_individual[day][slot][hall] = mutated_individual[other_day][other_slot][other_hall]
                    mutated_individual[other_day][other_slot][other_hall] = temp_lecture

    return mutated_individual

def elitism(population, fitness_scores, num_elites):
    """
    Select the top N elite individuals from the population.

    Parameters:
        population (list): List of candidate solutions (chromosomes).
        fitness_scores (list): List of fitness scores corresponding to each individual in the population.
        num_elites (int): The number of elite individuals to select.

    Returns:
        list: List of selected elite individuals (chromosomes).
    """
    elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:num_elites]
    elite_individuals = [population[i] for i in elite_indices]

    return elite_individuals

def should_terminate(fitness_scores, generations, max_generations, target_fitness):
    """
    Check if the genetic algorithm should terminate.

    Parameters:
        fitness_scores (list): List of fitness scores corresponding to each individual in the population.
        generations (int): Current generation number.
        max_generations (int): Maximum number of generations to run the algorithm.
        target_fitness (float): The target fitness score (the algorithm will stop if this score is reached).

    Returns:
        bool: True if the termination condition is met, False otherwise.
    """
    if generations >= max_generations:
        return True

    # Check if the target fitness score is reached
    if max(fitness_scores) >= target_fitness:
        return True

    return False

def genetic_algorithm(days, slots, halls, subjects, instructors, students, seating_capacity, slot_mapping,
                      population_size, num_elites, mutation_rate, max_generations, target_fitness):
    """
    Run the genetic algorithm for timetable scheduling.

    Parameters:
        days (list): List of days in the week.
        slots (dict): Dictionary containing slots for each day.
        halls (list): List of available halls.
        subjects (list): List of subjects to schedule.
        instructors (dict): Dictionary containing preferred slots for each instructor.
        students (dict): Dictionary containing the number of students for each subject.
        seating_capacity (int): Capacity of each hall (maximum number of students per hall).
        slot_mapping (dict): Slot name mapping to slot times.
        population_size (int): Size of the population in each generation.
        num_elites (int): Number of elite individuals to preserve in each generation.
        mutation_rate (float): Probability of mutation.
        max_generations (int): Maximum number of generations to run the algorithm.
        target_fitness (float): The target fitness score (the algorithm will stop if this score is reached).

    Returns:
        dict: The final timetable as the best individual (chromosome).
    """
    # Generate the initial random population
    population = [generate_random_timetable(days, slot_mapping, halls, subjects, instructors)
                  for _ in range(population_size)]

    generations = 0

    while not should_terminate([calculate_fitness(individual, seating_capacity, instructors) for individual in population],
                               generations, max_generations, target_fitness):

        # Calculate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, seating_capacity, instructors) for individual in population]

        # Perform elitism to select the best individuals
        elites = elitism(population, fitness_scores, num_elites)

        # Fill the rest of the population with offspring from crossover and mutation
        num_offspring = population_size - num_elites

        offspring = []
        while len(offspring) < num_offspring:
            # Perform tournament selection to select parents
            parents = tournament_selection(population, fitness_scores, 2)

            # Perform crossover to create offspring
            offspring1, offspring2 = one_point_crossover(parents[0], parents[1])

            # Perform mutation on offspring
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)

            offspring.append(offspring1)
            offspring.append(offspring2)

        # Combine elites and offspring to create the next generation
        population = elites + offspring
        for individual in population:
            print(calculate_fitness(individual, seating_capacity, instructors))
        generations += 1

    # Find the best individual (chromosome) as the final timetable
    fitness_scores = [calculate_fitness(individual, seating_capacity, instructors) for individual in population]
    best_index = fitness_scores.index(max(fitness_scores))
    best_timetable = population[best_index]

    return best_timetable


# Sample input data
subjects = ['DSA','OS','DPIoT','CodingTh','RTS','DIP','CRYALO','SNA','ISM','DIA','MLPR','DAA','MIICT(S)','ISP','DL','Compiler','WSN','TOC','CN','IGT','ML','BFSC','FCC','AICD','GameTheory','OPTIENG','FNC','DICD(S)','TSSN','DSDFPGA','ARFE','CCNGS','CSE','MAS','IVLSI','MWE','POC','IMEMS','IoT','ESD','Antenna','IBD(S)','TWRM','Ethnic','IMTC','EHB','PTS','MACROECO','EFE','RPE','MOBROB','MOBROS(S)','KInDynamics','MII','SMI','MD1','FMM','SDC','IW','MT2','MPOS','MIC','ROBOTICS','TQM','MDR','ICE']
instructors = {
  'CN': ['Subrat K Dash','Saurabh Kumar','Varun Kumar Sharma','Mohit Gupta','Ashish Kumar Dwivedi'],
  'MII': ['Dr. Mohan K Kadalbajoo','Pratibha Garg','Ratan Kumar Giri','Vikas Gupta'],
  'OS': ['Vikas Bajpai','Poulami Dalapati','Shweta Singh'],
  'DMS': ['Sudheer Kumar Sharma','Indra Deep Mastan','Shweta Singh'],
  'VEE': ['Narendra Kumar','TBD 2','Payel Pal','Karni Pal Bhati'],
  'TWRM': ['TBD 1'],
  'KInDynamics': ['Servesh Kumar Agnihotri'],
  'DAA': ['Bharavi Mishra','Shweta Bhandari','Shweta Saharan'],
  'POC': ['Deepak Nair','Vinay Bankey','Chirag Kumar','S Debnath'],
  'CSE': ['Bharat Verma','JOYEETA SINGHA','Kanjalochan Jena','Vinay Kumar Tiwari'],
  'TOC': ['Aloke Datta'],
  'FMM': ['Praveen Kumar Sharma'],
  'EFE': ['Sagnik Bagchi'],
  'DSA': ['Anukriti','Mukesh Jadon','Nirmal S','Poonam Gera'],
  'PTS': ['Rajbala Singh'],
  'MD1': ['A K Dargar'],
  'MIC': ['Rahul Singhal','Vikram Sharma'],
  'TQM': ['Manoj Kumar','Vikram Sharma'],
  'MT2': ['Deepak R Unune'],
  'IVLSI': ['Harshvardhan Kumar','Kusum Lata'],
  'ICE': ['K K Khatri'],
  'IoT': ['Abhishek Sharma','Vinay Kumar Tiwari'],
  'MLPR': ['Pawan Kumar'],
  'RPE': ['G C Tikkiwal'],
  'MDR': ['Atul Mishra'],
  'ARFE': ['Gopinath Samanta','R Tomar'],
  'DPIoT': ['Prof. Y'],
  'CodingTh': ['Prof. Z'],
  'RTS': ['Prof. W'],
  'DIP': ['Prof. V'],
  'CRYALO': ['Prof. U'],
  'SNA': ['Prof. T'],
  'ISM': ['Prof. S'],
  'DIA': ['Prof. R'],
  'MIICT(S)': ['Prof. O'],
  'ISP': ['Prof. N'],
  'DL': ['Prof. M'],
  'Compiler': ['Prof. L'],
  'WSN': ['Prof. K'], 
  'IGT': ['Prof. H'],
  'ML': ['Prof. G'],
  'BFSC': ['Prof. F'],
  'FCC': ['Prof. E'],
  'AICD': ['Prof. D'],
  'GameTheory': ['Prof. C'],
  'OPTIENG': ['Prof. B'],
  'FNC': ['Prof. A'],
  'DICD(S)': ['Prof. Z'],
  'TSSN': ['Prof. Y'],
  'DSDFPGA': ['Prof. X'],
  'CCNGS': ['Prof. V'],
  'MAS': ['Prof. T'],
  'MWE': ['Prof. R'],
  'IMEMS': ['Prof. P'],
  'ESD': ['Prof. N'],
  'Antenna': ['Prof. M'],
  'IBD(S)': ['Prof. L'],
  'Ethnic': ['Prof. J'],
  'IMTC': ['Prof. I'],
  'EHB': ['Prof. H'],
  'MACROECO': ['Prof. F'],
  'MOBROB': ['Prof. C'],
  'MOBROS(S)': ['Prof. B'],
  'SMI': ['Prof. Y'],
  'SDC': ['Prof. V'],
  'IW': ['Prof. U'],
  'MPOS': ['Prof. S'],
  'ROBOTICS': ['Prof. Q']
}
students = data = {
  'Y19 B.Tech CSE': {'DSA': 1, 'OS': 2, 'RTS': 21, 'DIA': 17, 'MLPR': 16, 'DAA': 2, 'MIICT(S)': 76, 'ISP': 63, 'Compiler': 4, 'TOC': 2, 'CN': 5, 'GameTheory': 33, 'FNC': 5, 'IBD(S)': 44, 'IMTC': 29, 'EHB': 34, 'MACROECO': 41, 'MPOS': 3},
  'Y19 B.Tech ECE': {'DSA': 1, 'CN': 6, 'AICD': 9, 'GameTheory': 37, 'FNC': 38, 'DICD(S)': 9, 'CSE': 1, 'MAS': 1, 'IVLSI': 1, 'MWE': 1, 'POC': 1, 'IMTC': 13, 'EHB': 28, 'MACROECO': 20, 'MII': 60, 'MPOS': 1},
  'Y19 B.Tech ME': {'DSA': 1, 'GameTheory': 4, 'IMTC': 9, 'EHB': 6, 'MACROECO': 9, 'EFE': 2, 'MOBROB': 22, 'MOBROS(S)': 12, 'IW': 6, 'MPOS': 12, 'ICE': 1},
  'Y20 B.Tech CCE': {'DSA': 5, 'OS': 4, 'CRYALO': 2, 'SNA': 20, 'ISM': 1, 'DAA': 11, 'DL': 16, 'WSN': 4, 'CN': 110, 'IGT': 20, 'ML': 14, 'BFSC': 1, 'FCC': 5, 'OPTIENG': 5, 'TSSN': 1, 'CCNGS': 7, 'CSE': 3, 'POC': 23, 'IMEMS': 34, 'IoT': 110, 'Antenna': 1, 'Ethnic': 36, 'EFE': 110, 'SMI': 7, 'SDC': 3},
  'Y20 B.Tech CSE': {'DSA': 4, 'OS': 6, 'DIP': 1, 'CRYALO': 92, 'SNA': 77, 'ISM': 57, 'DAA': 24, 'DL': 80, 'Compiler': 230, 'WSN': 66, 'TOC': 18, 'CN': 29, 'IGT': 79, 'ML': 54, 'BFSC': 27, 'FCC': 89, 'PTS': 2, 'EFE': 7},
  'Y20 B.Tech ECE': {'DSA': 6, 'CN': 136, 'OPTIENG': 6, 'TSSN': 58, 'DSDFPGA': 49, 'CCNGS': 76, 'CSE': 1, 'IVLSI': 20, 'MWE': 3, 'POC': 15, 'IMEMS': 50, 'ESD': 49, 'Antenna': 3, 'Ethnic': 48, 'PTS': 136, 'SMI': 13, 'SDC': 3},
  'Y20 B.Tech ME': {'DSA': 2, 'Ethnic': 11, 'EFE': 36, 'KInDynamics': 18, 'SMI': 12, 'MD1': 14, 'FMM': 16, 'SDC': 5, 'MT2': 5, 'MIC': 3, 'ROBOTICS': 35, 'TQM': 2, 'ICE': 36},
  'Y21 B.Tech CCE': {'DSA': 12, 'OS': 110, 'DAA': 110, 'CSE': 110, 'POC': 110},
  'Y21 B.Tech CSE': {'DSA': 3, 'OS': 235, 'DPIoT': 44, 'CodingTh': 96, 'DIP': 95, 'DAA': 235, 'TOC': 235, 'CN': 235, 'PTS': 113, 'EFE': 122, 'MDR': 1}, 
  'Y21 B.Tech ECE': {'DSA': 10, 'CSE': 127, 'IVLSI': 127, 'MWE': 127, 'POC': 127, 'MDR': 1},
  'Y21 B.Tech ME': {'DSA': 2, 'KInDynamics': 39, 'MD1': 39, 'FMM': 39, 'MT2': 39, 'MIC': 39, 'TQM': 39, 'MDR': 4},
  'Y16 B.Tech CSE': {'OS': 1, 'RTS': 1, 'DIA': 2, 'MLPR': 2, 'ISP': 1, 'Compiler': 1, 'TOC': 3, 'CN': 1, 'IMTC': 2, 'MPOS': 1},
  'Y18 B.Tech CCE': {'OS': 1, 'RTS': 1, 'DIA': 1, 'DAA': 1, 'CN': 1, 'CSE': 1, 'MAS': 1, 'POC': 1, 'IoT': 2, 'EFE': 1},
  'Y21 B.Tech Dual Degree CSE': {'OS': 8, 'DPIoT': 2, 'CodingTh': 4, 'DIP': 2, 'DAA': 8, 'TOC': 8, 'CN': 8, 'PTS': 8},
  'Y17 B.Tech CSE': {'RTS': 1, 'DIA': 1, 'MLPR': 1, 'DAA': 1, 'IMTC': 1},
  'Y18 B.Tech CSE': {'RTS': 1, 'ISM': 1, 'DIA': 2, 'MLPR': 1, 'DAA': 1, 'ISP': 1, 'Compiler': 2, 'WSN': 1, 'ML': 1, 'IMTC': 1, 'MACROECO': 1},
  'Y19 B.Tech Dual Degree CSE': {'RTS': 7, 'DIA': 2, 'MLPR': 3, 'ISP': 8, 'IMTC': 6, 'EHB': 1, 'MACROECO': 3, 'MPOS': 5},
  'Y19 B.Tech CCE': {'RTS': 7, 'DIA': 2, 'DAA': 1, 'MIICT(S)': 29, 'ISP': 17, 'CN': 1, 'AICD': 1, 'GameTheory': 26, 'FNC': 23, 'DICD(S)': 2, 'CSE': 1, 'MAS': 3, 'IoT': 2, 'IBD(S)': 4, 'IMTC': 19, 'EHB': 16, 'MACROECO': 12, 'EFE': 1, 'MII': 29, 'MPOS': 1},
  'Y22 M.Sc CSE': {'RTS': 2, 'DIA': 2, 'MLPR': 4, 'ISP': 4, 'TWRM': 4},
  'Y20 B.Tech Dual Degree CSE': {'CRYALO': 6, 'SNA': 5, 'ISM': 2, 'DL': 1, 'Compiler': 11, 'WSN': 1, 'TOC': 1, 'CN': 1, 'IGT': 3, 'ML': 2, 'FCC': 6, 'Ethnic': 5},
  'Y18 B.Tech Dual Degree CSE': {'MLPR': 4, 'IBD(S)': 3},
  'Y20 B.Tech Dual Degree ECE': {'CN': 7, 'OPTIENG': 2, 'TSSN': 3, 'DSDFPGA': 1, 'CCNGS': 4, 'CSE': 1, 'POC': 1, 'IMEMS': 2, 'ESD': 2, 'Antenna': 1, 'Ethnic': 1, 'PTS': 7, 'SMI': 2},
  'Y19 B.Tech Dual Degree ECE': {'AICD': 1, 'ARFE': 1, 'IMTC': 3, 'EHB': 1, 'MACROECO': 2, 'MII': 6, 'MPOS': 1},
  'Y17 B.Tech Dual Degree ECE': {'CSE': 1, 'EFE': 1},
  'Y21 B.Tech Dual Degree ECE': {'CSE': 8, 'IVLSI': 8, 'MWE': 8, 'POC': 8},
  'Y21 M.Sc PHY': {'IMTC': 3, 'EHB': 6},
  'Y22 Ph.D ECE': {'RPE': 5},
  'Y22 Ph.D HSS': {'RPE': 2},
  'Y22 Ph.D MTH': {'RPE': 2},
  'Y22 Ph.D PHY': {'RPE': 1}
}
seating_capacity = 30
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Slot name mapping
slot_mapping = {
    'slot1': '8-9',
    'slot2': '9-10',
    'slot3': '10-11',
    'slot4': '11-12',
    'slot5': '12-1',
    'slot6': '1-2',
    'slot7': '2-3',
    'slot8': '3-4',
    'slot9': '4-5',
    'slot10': '5-6',
    'slot11': '8-9.30',
    'slot12': '9.30-11',
    'slot13': '11-12.30',
    'slot14': '1-2.30',
    'slot15': '2.30-4',
    'slot16': '4-5.30',
}

# Define slots for each day separately
slots = {
    'Monday': ['slot1', 'slot2', 'slot3', 'slot4', 'slot5', 'slot6', 'slot7', 'slot8', 'slot9', 'slot10'],
    'Tuesday': ['slot11', 'slot12', 'slot13', 'slot14', 'slot15', 'slot16'],
    'Wednesday': ['slot1', 'slot2', 'slot3', 'slot4', 'slot5', 'slot6', 'slot7', 'slot8', 'slot9', 'slot10'],
    'Thursday': ['slot11', 'slot12', 'slot13', 'slot14', 'slot15', 'slot16'],
    'Friday': ['slot1', 'slot2', 'slot3', 'slot4', 'slot5', 'slot6', 'slot7', 'slot8', 'slot9', 'slot10'],
}


halls = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19']

timetable = generate_random_timetable(days, slot_mapping, halls, subjects, instructors)

# Calculate fitness for the generated timetable
fitness_score = calculate_fitness(timetable, seating_capacity, instructors)
print("Fitness Score:", fitness_score)

# # Print the timetable
for day in timetable:
    print(day)
    for slot in timetable[day]:
        print(f"- {slot}")
        for hall in timetable[day][slot]:
            lecture = timetable[day][slot][hall]
            print(f"-- {hall}: {lecture}")
    print()

best_timetable = genetic_algorithm(days, slots, halls, subjects, instructors, students, seating_capacity,
                                   slot_mapping, population_size=500, num_elites=5, mutation_rate=0.08,
                                   max_generations=2000, target_fitness=50)

# Print the final timetable
for day in best_timetable:
    print(day)
    for slot in best_timetable[day]:
        print(f"- {slot}")
        for hall in best_timetable[day][slot]:
            lecture = best_timetable[day][slot][hall]
            print(f"-- {hall}: {lecture}")
    print()
