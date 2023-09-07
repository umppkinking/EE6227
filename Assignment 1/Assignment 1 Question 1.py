import random
import math

random_table = [181, 637, 962, 726, 680, 199, 478, 541, 540, 266, 121, 963, 294, 798, 673, 185, 440, 91, 497, 892, 97, 200, 978,
                894, 373, 74, 119, 419, 939, 214, 252, 190, 333, 27, 932, 200, 661, 4, 7, 406,
                586, 314, 26, 882, 405, 426, 673, 583, 697, 116, 716, 284, 89, 817, 504, 779, 512, 606, 939, 36, 498, 247, 532,
                731, 537, 503, 859, 331, 967, 891, 154, 335, 725, 834, 639, 657, 880, 922, 82, 516,
                430, 954, 463, 721, 488, 533, 135, 143, 173, 964, 569, 10, 576, 800, 46, 211, 871, 619, 187, 570, 171, 947, 852,
                523, 808, 12, 789, 875, 852, 845, 674, 874, 237, 935, 728, 899, 649, 791, 136, 516,
                151, 243, 884, 276, 11, 594, 360, 998, 240, 565, 699, 471, 17, 78, 493, 268, 602, 476, 261, 997, 613, 197, 377,
                654, 135, 759, 284, 110, 488, 696, 653, 480, 671, 285, 402, 743, 973, 132, 529, 172,
                940, 159, 749, 135, 359, 407, 961, 210, 864, 684, 67, 778, 911, 943, 709, 793, 874, 957, 1, 881, 432, 796, 311,
                127, 726, 849, 108, 938, 345, 329, 333, 678, 608, 834, 154, 532, 921, 42, 159, 169]
reminder_mat_num = 79
random.seed(reminder_mat_num)


# convert the binary code into real value in [x y]
def bitstring_to_real(bin_code, x, y):
    decimal = 0
    n = len(bin_code)

    for i in range(n):
        decimal += bin_code[9 - i] * (2 ** i)

    return x + (y - x) / ((2 ** n) - 1) * decimal


# calculate the fitness value
def function_Ackley(dec_code, n):
    dec_count = 0
    cos_count = 0

    for i in range(n):
        dec_count += dec_code[i] ** 2
        cos_count += math.cos(2 * math.pi * dec_code[i])

    return -20 * math.exp(-0.2 * math.sqrt(1 / n) * dec_count) - math.exp(1 / n * cos_count) + 20 + math.exp(1)


# get the random number from the random table
def num_from_table(table, index):
    n = len(table)
    while index > n:
        index -= n
    return table[index - 1]


# get the mutation gene
def mutation(rand_num):
    if rand_num >= 500:
        return 1
    else:
        return 0


# get the split point mapping from [0, 999] to [0, 9], reserved integer
def recombination_split_point(rand_num):
    return int(rand_num / 1000 * 10)


# whether the probability of recombination happened mapping from [0, 999] to [0, 1]
def recombination_probability(rand_num):
    return rand_num / 1000


# get the recombination parent index mapping from [0, 999] to [0, 4]
def recombination_parent_index(rand_num):
    return int(rand_num / 2000 * 10)


# get the offspring chromosomes
def mate(father_chromosomes, mother_chromosomes, recombination_prob, split_point, recombination_num):
    # recombination_prob <= 0.7 means the crossover happens
    if recombination_prob <= 0.7:
        # recombination_num <= 500 means the genes before split_point come from father and the rest come from mother
        if recombination_num <= 500:
            return father_chromosomes[:split_point] + mother_chromosomes[split_point:]
        else:
            return mother_chromosomes[:split_point] + father_chromosomes[split_point:]
    else:
        # recombination_num <= 500 means the whole genes come from father
        if recombination_num <= 500:
            return father_chromosomes
        else:
            return mother_chromosomes


# initialization the binary list
initial_bin_list = []
for _ in range(10):
    person_list = []
    for _ in range(3):
        bin_code = []
        for _ in range(10):
            bin_code.append(random.randint(0, 1))
        person_list.append(bin_code)
    initial_bin_list.append(person_list)

# initialization binary code and fitness list
initial_bin_fitness_list = []
for chromosomes in initial_bin_list:
    bin_fitness_code = []
    dec_code = []
    for i in chromosomes:
        j = bitstring_to_real(i, -20, 20)
        dec_code.append(j)
    fitness = function_Ackley(dec_code, len(chromosomes))
    bin_fitness_code.append(chromosomes)
    bin_fitness_code.append(fitness)
    initial_bin_fitness_list.append(bin_fitness_code)

print(initial_bin_fitness_list)
# generation
generation = 1

# start index in the random table is  79
index = reminder_mat_num

current_bin_fitness_list = initial_bin_fitness_list

while generation <= 3:
    # elitism 10% parents to 10% offspring
    current_bin_fitness_list.append(current_bin_fitness_list[0])

    # top 50% parents mate to 90% offspring
    for _ in range(9):
        # select parents
        random_num = num_from_table(random_table, index)
        father_index = recombination_parent_index(random_num)
        father = current_bin_fitness_list[father_index]
        index += 1

        random_num = num_from_table(random_table, index)
        mother_index = recombination_parent_index(index)
        mother = current_bin_fitness_list[mother_index]
        index += 1

        offspring = []
        for i in range(3):
            # crossover in each chromosomes
            random_num = num_from_table(random_table, index)
            prob = recombination_probability(random_num)
            index += 1

            random_num = num_from_table(random_table, index)
            split = recombination_split_point(random_num)
            index += 1

            random_num = num_from_table(random_table, index)
            rank = random_num
            index += 1

            offspring_chromosomes = mate(father[0][i], mother[0][i], prob, split, rank)

            # mutation
            random_num = num_from_table(random_table, index)
            mut_prob = random_num
            index += 1

            # mutation happen
            if mut_prob >= 900:
                random_num = num_from_table(random_table, index)
                mut_gene = mutation(random_num)
                index += 1

                random_num = num_from_table(random_table, index)
                mut_index = int(random_num / 1000 * 10)
                index += 1

                offspring_chromosomes[mut_index] = mut_gene

            offspring.append(offspring_chromosomes)

        offspring_dec_code = []
        offspring_bin_fitness_code = []
        for i in offspring:
            j = bitstring_to_real(i, -20, 20)
            offspring_dec_code.append(j)

        # calculate the fitness for each offspring
        offspring_fitness = function_Ackley(offspring_dec_code, len(offspring))
        offspring_bin_fitness_code.append(offspring)
        offspring_bin_fitness_code.append(offspring_fitness)

        # combine the offspring and parents
        current_bin_fitness_list.append(offspring_bin_fitness_code)

    # select the top 50% population to the next generation which maintains 10 population size
    current_bin_fitness_list.sort(key=lambda x: x[1])
    current_bin_fitness_list = current_bin_fitness_list[:10]

    print(current_bin_fitness_list)

    generation += 1

