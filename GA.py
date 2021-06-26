# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:09:05 2021

@author: ruj72813
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    return 100 - (x[0] - 5) ** 2 - (x[1] - 5) ** 2 - (x[2] - 5) ** 2


class GA(object):

    def __init__(self, pop_size, cho_length, var_num, bound, p_c, p_m, function, generation=100):
        self.population_size = pop_size
        self.cho_length = cho_length
        self.num = var_num
        self.up_bd = np.array(bound[0])
        self.down_bd = np.array(bound[1])
        self.bound = bound
        self.pc = p_c
        self.pm = p_m
        self.population = self.species_origin()
        self.f = function
        self.generation = generation

    def species_origin(self):
        population = [[]]
        for i in range(self.population_size):
            temp = []
            for j in range(self.num):
                temporary = []
                for k in range(self.cho_length):
                    temporary.append(random.randint(0, 1))
                    # 随机产生一个染色体,由二进制数组成
                temp.append(temporary)
            population.append(temp)
        return population[1:]
        # 将种群返回，种群是个3维数组，个体、特性和染色体3维

    def translation(self):
        temporary = []
        for people in self.population:
            tmp = []
            for var in people:
                total = 0
                for k in range(self.cho_length):
                    total += var[k] * (math.pow(2, k))  # 求和
                tmp.append(total)
                # 一个染色体编码完成，由一个二进制数编码为一个十进制数
            temporary.append(tmp)
        return temporary  # 此时应为2维

    # 目标函数相当于环境 对染色体进行筛选，这里是f(x)
    def function(self):
        func = []
        temp = np.array(self.translation())
        for i in range(self.population_size):
            x = temp[i] / (math.pow(2, self.cho_length) - 1) * (self.up_bd - self.down_bd) + self.down_bd
            func.append(self.f(x))
        return func  # 此时返回一维数组

    # 定义适应度
    def fitness(self):
        fitness_value = []
        func = self.function()
        for i in range(len(func)):
            if func[i] > 0:
                temporary = func[i]
            else:
                temporary = 0.0
            fitness_value.append(temporary)
        return fitness_value

    # 3.选择种群中个体适应度最大的个体
    def selection(self, fitness_value):
        # 射杀一半
        fit_new = [[fitness_value[i], i] for i in range(len(fitness_value))]
        fit_new.sort()
        new_pop = self.population.copy()
        for i in range(self.population_size):
            if i < self.population_size / 2:
                for j in range(self.num):
                    for k in range(self.cho_length):
                        new_pop[i][j][k] = self.population[int(fit_new[i + int(self.population_size / 2)][1])][j][k]
            else:
                for j in range(self.num):
                    for k in range(self.cho_length):
                        new_pop[i][j][k] = self.population[int(fit_new[i][1])][j][k]
        self.population = new_pop.copy()

    # 4.交叉操作
    def crossover(self):
        population = self.population
        pop_len = len(population)
        for i in range(pop_len - 1):
            if random.random() < self.pc:
                point = random.randint(0, len(population[0]))
                # 在种群个数内随机生成单点交叉点
                temporary1 = []
                temporary2 = []
                temporary1.extend(population[i][0:point])
                temporary1.extend(population[i + 1][point:len(population[i])])
                temporary2.extend(population[i + 1][0:point])
                temporary2.extend(population[i][point:len(population[i])])
                population[i] = temporary1
                population[i + 1] = temporary2
                # 第i个染色体和第i+1个染色体基因重组/交叉完成

    def mutation(self):
        for people in self.population:
            for var in people:
                if random.random() < self.pm:
                    point = random.randint(0, self.cho_length - 1)
                    if var[point] == 1:
                        var[point] = 0
                    else:
                        var[point] = 1

    def evolution(self, fitness_value):
        self.selection(fitness_value)
        self.crossover()
        self.mutation()

    def b2d(self, best_individual):
        best_x = []
        for i in range(self.num):
            total = 0
            for j in range(self.cho_length):
                total = total + best_individual[i][j] * math.pow(2, j)
            total = total / (math.pow(2, self.cho_length) - 1) * (self.up_bd[i] - self.down_bd[i]) + self.down_bd[i]
            best_x.append(total)
        return best_x

    # 寻找最好的适应度和个体
    def best(self, fitness_value):
        population = self.population
        best_individual = population[0]
        best_fitness = fitness_value[0]
        for i in range(1, len(population)):
            # 循环找出最大的适应度，适应度最大的也就是最好的个体
            if fitness_value[i] > best_fitness:
                best_fitness = fitness_value[i]
                best_individual = population[i]
        return best_individual, best_fitness

    @staticmethod
    def result_plot(results):
        x = []
        y = []
        for i in range(len(results)):
            x.append(i)
            y.append(results[i][0])
        plt.plot(x, y)
        plt.show()

    def show(self):
        for people in self.population:
            print(self.b2d(people), f(self.b2d(people)))

    def solver(self):

        results = [[]]

        for i in range(self.generation):
            fitness_value = self.fitness()
            best_individual, best_fitness = self.best(fitness_value)
            results.append([best_fitness, self.b2d(best_individual)])
            # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制,适应度函数
            self.evolution(fitness_value)

        results = results[1:]
        # self.result_plot(results)
        # print(results)
        results.sort(key=lambda x: x[0])
        # self.result_plot(results)
        return results[-1]


if __name__ == '__main__':
    population_size = 100
    num = 3
    bound = [[0, 0, 0], [10, 10, 10]]
    chromosome_length = 10
    pc = 0.6
    pm = 0.5
    ga = GA(population_size, chromosome_length, num, bound, pc, pm, f)
    print(ga.solver())
