"""
Source code attached for the manuscript:

"Development of a health care system simulator for Japan toward efficient allocation of medical resources" by
Yusuke Kawada, Yusuke Hayashi, and Hirokazu Sugiyama

This code corresponds to the compartment model representing overall health transitions.
Details of the model are described in "Methods / Country adaptation module / 2. Analyze disease trends".
The output figure corresponds to "Supplementary Information S1.

Created by Yusuke Kawada/Sugiyama-Badr Lab/The University of Tokyo
Last saved on Feb 04 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

group_num = 15
group_width = 5

t_spans = [(2001, 2004), (2004, 2007), (2007, 2010), (2010, 2013), (2013, 2016), (2016, 2019)]

input_dir = os.path.join('input', 'For_overall_health_transition')
output_dir = os.path.join('output', 'For_overall_health_transition')
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = ['#FF4B00', '#03AF7A', '#005AFF'] # red(healthy), green(in-care), blue(deceased)
font_size = 18
m_size = 11

def state_change(t, X, beta_all, mu_all):
    population_now, healthy_now, care_now = np.split(X, [group_num, 2*group_num])
    
    # get aging_flow
    now_row = int(start_row + np.floor(t - start_year)) # row having # of birth in the year t
    aging_flow = np.zeros(group_num + 1)

    # aging_flow = [v_0, v_1, ....., v_n]
    for i in range(group_num + 1):
        aging_flow[i] = df_birth[now_row - group_width * i][1]

    # add x_H,0 = 1, x_C,0 = 0(assumption) to healthy_now, care_now
    healthy_now = np.insert(healthy_now, 0, healthy_0)
    care_now = np.insert(care_now, 0, care_0)

    # model equation
    d_population = np.zeros(group_num)
    d_healthy = np.zeros(group_num+1)
    d_care = np.zeros(group_num+1)

    for i in range(group_num):
        d_population[i] = aging_flow[i] - aging_flow[i+1]

    for i in range(1, group_num+1):
        d_healthy[i] = (healthy_now[i-1]-healthy_now[i]) * aging_flow[i-1] / population_now[i-1] - beta_all[i] * healthy_now[i] # population[i-1] = N_i
        d_care[i] = (care_now[i-1]-care_now[i]) * aging_flow[i-1] / population_now[i-1] -mu_all[i] * care_now[i] + beta_all[i] * healthy_now[i]

    # change healthy, care, death
    d_healthy = np.delete(d_healthy, 0)
    d_care = np.delete(d_care, 0)
    
    dX = np.append(d_population, d_healthy)
    dX = np.append(dX, d_care)

    return dX

def k_make(ks):
    # beta_all corresponds to beta, mu_all corresponds to mu
    # define beta_all, mu_all here !Note that beta_all0, mu_all0 is included in beta_all, mu_all
    k1, k2, k3, k4, k5, k6= ks
    beta_all = np.zeros(group_num+1)
    mu_all = np.zeros(group_num+1)
    for i in range(5):
        beta_all[i] = k1
        mu_all[i] = k4
    for i in range(5, 12):
        beta_all[i] = k2
        mu_all[i] = k5
    for i in range(12, group_num + 1):
        beta_all[i] = k3
        mu_all[i] = k6
    
    return beta_all, mu_all

if __name__ == '__main__':
    R2_record = np.zeros((len(t_spans), 11))
    for i_t in range(len(t_spans)):
        t_span = t_spans[i_t]
        start_year = t_span[0]
        end_year = t_span[1]

        # Specify the file path
        file_path_birth = os.path.join(input_dir, 'birth_num.csv') #file path of birth number data
        file_path_start = os.path.join(input_dir, '99_All',f'X_all_{str(start_year)}.csv') #file path of input data
        file_path_end = os.path.join(input_dir, '99_All',f'X_all_{str(end_year)}.csv') #file path of measured data
        file_path_result = os.path.join(input_dir, 'fitted_transition_rates_all.csv') #file path of fitted transition rate

        # Read the CSV file
        # df_birth->0:year, 1:birth_num
        # df_start, df_end->0:start_age, 1:end_age, 2:healthy, 3:care, 4:death
        df_birth = pd.read_csv(file_path_birth)
        df_start = pd.read_csv(file_path_start)
        df_end = pd.read_csv(file_path_end)
        df_result = pd.read_csv(file_path_result)
        df_birth = df_birth.values
        df_result = df_result.values

        # covert data to １D array
        healthy_start = df_start.iloc[0:group_num, [2]].values.flatten()
        care_start = df_start.iloc[0:group_num, [3]].values.flatten()
        healthy_end = df_end.iloc[0:group_num, [2]].values.flatten()
        care_end = df_end.iloc[0:group_num, [3]].values.flatten()
        death_end = df_end.iloc[0:group_num, [4]].values.flatten()

        measured = np.append(healthy_end, care_end)
        # made asuumption that x_healthy, x_care at age 0 is same as the start year
        healthy_0 = healthy_start[0]
        care_0 = care_start[0]

        # make fist population
        population_start = np.zeros(group_num)
        start_row = 0 # row having # of birth in start_year
        while df_birth[start_row][0] < start_year:
            start_row += 1

        for i_g in range(group_num):
            for j_g in range(group_width):
                population_start[i_g] += df_birth[start_row - group_width * i_g - j_g][1]

        start_row += 1

        # make population, x_healthy, x_care into one array
        # [N_1, N_2, ....., N_n, x_h1, x_h2, ....., x_hn, x_c1, x_c2, ....., x_cn]
        start = np.append(population_start, healthy_start)
        start = np.append(start, care_start)

        fitting_result = df_result[i_t][2:]
        beta_all, mu_all = k_make(fitting_result)
        sol = solve_ivp(state_change, t_span, start, args=(beta_all, mu_all), t_eval=[end_year], method='LSODA')

        population_calc, healthy_calc, care_calc = np.split(sol.y.squeeze(), [group_num, 2*group_num])
        death_calc = 1 - healthy_calc - care_calc

        # make graph
        X = []
        for i_g in range(group_num):
            X.append(str(i_g*5) + '\u2013' + str(i_g*5+4))
        X = np.array(X)
        plt.figure(figsize=(6, 4.5))
        plt.plot(X, healthy_end, color=colors[0], marker='.', ls='', markersize=m_size, label='healthy')
        plt.plot(X, healthy_calc, color=colors[0], ls='-')
        plt.plot(X, care_end, color=colors[1], marker='.', ls='', markersize=m_size, label='in-care')
        plt.plot(X, care_calc, color=colors[1], ls='-')
        plt.plot(X, death_end, color=colors[2], marker='.', ls='', markersize=m_size, label='deceased')
        plt.plot(X, death_calc, color=colors[2], ls='-')
        plt.xlabel('Age', fontsize=font_size)
        plt.ylabel('Composition of overall health state ${x}^{\mathrm{all}}_{i}$ [\u2013]', fontsize=font_size)
        plt.xticks(np.arange(len(X)), rotation=90)
        plt.tick_params(labelsize=font_size)
        plt.ylim([0, 1])
        plt.title(f'{start_year}\u2013{end_year}', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.savefig(os.path.join(output_dir, f'Overall_health_transition_{start_year}_{end_year}.tif'), bbox_inches='tight')
        plt.show()
        plt.close()

        # output fitting result to csv
        df_calc = df_end.copy()
        df_calc.iloc[0:group_num, [2]] = healthy_calc
        df_calc.iloc[0:group_num, [3]] = care_calc
        df_calc.iloc[0:group_num, [4]] = death_calc
        df_calc.drop(index=df_calc.index[group_num:], inplace=True)
        df_calc.to_csv(os.path.join(output_dir, f'X_all_calc_{end_year}_from_{start_year}.csv'), index=False)
        
        # record R2
        R2_record[i_t][0] = start_year
        R2_record[i_t][1] = end_year
        for i in range(6):
            R2_record[i_t][i+2] = fitting_result[i]
        
        R2_record[i_t][8] = r2_score(healthy_end, healthy_calc)
        R2_record[i_t][9] = r2_score(care_end, care_calc)
        R2_record[i_t][10] = r2_score(death_end, death_calc)
        
    # make CSv for R2
    df_record = pd.DataFrame(R2_record, columns=['start_year', 'end_year', 'beta_all_young', 'beta_all_middle', 'beta_all_elderly', 
                                                'mu_all_young', 'mu_all_middle', 'mu_all_elderly',
                                                'R2(healthy)', 'R2(care)', 'R2(death)'])
    df_record.to_csv(os.path.join(output_dir, 'Fitting_evaluations.csv'), index=False)
