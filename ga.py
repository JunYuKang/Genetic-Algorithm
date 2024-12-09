import random
from math import sin, pi
from tabulate import tabulate
import matplotlib.pyplot as plt

class init_x:
    def __init__(self, lower_bound, upper_bound, bits):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bits = bits
    
    def gene(self):
        random_bits = random.getrandbits(self.bits)
        x_bits = bin(random_bits)[2:].zfill(self.bits)
        x_int = int(x_bits, 2)
        x_float = self.lower_bound + x_int * (self.upper_bound - self.lower_bound) / (2 ** self.bits - 1)
        return x_bits, x_float

def genetic_algorithm(x20):
    x1 = init_x(-3.0, 12.1, 18)
    x2 = init_x(4.1, 5.8, 15)

    # 如果 x20 為空，生成初始種群
    if not x20:
        for i in range(20):
            x1_bits, x1_float = x1.gene()
            x2_bits, x2_float = x2.gene()
            new_x_bits = x1_bits + x2_bits
            new_x_float = (x1_float, x2_float)

            # 計算 f(x1, x2)
            f_value = 21.5 + x1_float * sin(4 * pi * x1_float) + x2_float * sin(20 * pi * x2_float)
            x20.append([i + 1, new_x_bits, new_x_float, f_value])

    print(tabulate(x20, tablefmt='grid'))

    # 計算總和
    x20_total = sum(x[3] for x in x20)
    print('總和 :', x20_total)

    # 計算累積機率
    c20 = []
    cumulative_sum = 0
    for i in range(20):
        c = x20[i][3] / x20_total
        cumulative_sum += c
        c20.append([x20[i][0], x20[i][3], cumulative_sum])

    print(tabulate(c20, tablefmt='grid'))

    # 生成隨機數據
    random_20 = []
    for i in range(20):
        random_float1 = random.uniform(0, 1)
        random_float2 = random.uniform(0, 1)
        random_20.append([i + 1, random_float1, random_float2])

    print(tabulate(random_20, tablefmt='grid'))

    # 輪盤選擇
    new_population = []
    for i in range(20):
        for j in range(20):
            if random_20[i][1] < c20[j][2]:
                new_population.append(x20[j])
                break

    # 保留最佳個體（精英策略）
    best_individual = max(x20, key=lambda x: x[3])
    new_population[0] = best_individual

    print(tabulate(new_population, tablefmt='grid'))

    # 交叉操作
    crossed_population = []
    for i in range(20):
        if random_20[i][2] < 0.25:
            crossed_population.append(new_population[i])

    print('交配後的族群')
    print(tabulate(crossed_population, tablefmt='grid'))

    # 交叉操作示例（單點交叉）
    def single_point_crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1[1]) - 1)
        child1_bits = parent1[1][:crossover_point] + parent2[1][crossover_point:]
        child2_bits = parent2[1][:crossover_point] + parent1[1][crossover_point:]
        return child1_bits, child2_bits

    # 執行交叉操作
    if len(crossed_population) >= 2:
        new_generation = [best_individual]  # 保留最佳個體
        for i in range(0, len(crossed_population) - 1, 2):
            parent1 = crossed_population[i]
            parent2 = crossed_population[i + 1]
            child1_bits, child2_bits = single_point_crossover(parent1, parent2)
            
            # 計算子代的浮點數值和適應值
            child1_x1_bits = child1_bits[:18]
            child1_x2_bits = child1_bits[18:]
            child1_x1_float = x1.lower_bound + int(child1_x1_bits, 2) * (x1.upper_bound - x1.lower_bound) / (2 ** x1.bits - 1)
            child1_x2_float = x2.lower_bound + int(child1_x2_bits, 2) * (x2.upper_bound - x2.lower_bound) / (2 ** x2.bits - 1)
            child1_f_value = 21.5 + child1_x1_float * sin(4 * pi * child1_x1_float) + child1_x2_float * sin(20 * pi * child1_x2_float)
            
            child2_x1_bits = child2_bits[:18]
            child2_x2_bits = child2_bits[18:]
            child2_x1_float = x1.lower_bound + int(child2_x1_bits, 2) * (x1.upper_bound - x1.lower_bound) / (2 ** x1.bits - 1)
            child2_x2_float = x2.lower_bound + int(child2_x2_bits, 2) * (x2.upper_bound - x2.lower_bound) / (2 ** x2.bits - 1)
            child2_f_value = 21.5 + child2_x1_float * sin(4 * pi * child2_x1_float) + child2_x2_float * sin(20 * pi * child2_x2_float)
            
            new_generation.append([len(new_generation) + 1, child1_bits, (child1_x1_float, child1_x2_float), child1_f_value])
            new_generation.append([len(new_generation) + 1, child2_bits, (child2_x1_float, child2_x2_float), child2_f_value])

        for i in range(len(new_generation)):
            if i < len(new_population):
                new_population[i] = new_generation[i]
            else:
                new_population.append(new_generation[i])

        print('更新後的族群')
        print(tabulate(new_population, tablefmt='grid'))

    # 突變操作
    mutation_rate = 0.1  # 突變率
    for individual in new_population:
        if random.uniform(0, 1) < mutation_rate:
            mutation_point = random.randint(0, len(individual[1]) - 1)
            mutated_bits = list(individual[1])
            mutated_bits[mutation_point] = '1' if mutated_bits[mutation_point] == '0' else '0'
            individual[1] = ''.join(mutated_bits)
            # 更新浮點數值和適應值
            x1_bits = individual[1][:18]
            x2_bits = individual[1][18:]
            x1_float = x1.lower_bound + int(x1_bits, 2) * (x1.upper_bound - x1.lower_bound) / (2 ** x1.bits - 1)
            x2_float = x2.lower_bound + int(x2_bits, 2) * (x2.upper_bound - x2.lower_bound) / (2 ** x2.bits - 1)
            f_value = 21.5 + x1_float * sin(4 * pi * x1_float) + x2_float * sin(20 * pi * x2_float)
            individual[2] = (x1_float, x2_float)
            individual[3] = f_value

    return new_population

x20 = []
fitness_history = []
for generation in range(10000):
    x20 = genetic_algorithm(x20)
    # 記錄每次迭代的最佳適應值
    best_fitness = max(individual[3] for individual in x20)
    fitness_history.append(best_fitness)

# 繪製折線圖
plt.plot(range(1, 10001), fitness_history, marker='o')
plt.title('Best Fitness Value Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.grid(True)
plt.show()