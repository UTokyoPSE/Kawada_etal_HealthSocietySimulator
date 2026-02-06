# Main processing of the simulator
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Class to manage common variables
class Context:
    """
    Class to manage common variables
    """
    # Load data
    def __init__(self):
        # Constant part
        input_dir = os.path.join('input', 'For_simulator')
        self.repeat_num = 3  # Number of repetitions
        self.time_span = 12  # Years to observe
        self.record_span = 3  # Record every N years
        self.start_year = 2019  # Starting year
        self.num_agent_each_age = 10000
        self.covered_age = 75
        self.group_width = 5
        self.group_num = self.covered_age // self.group_width
        # Read initial prevalence and population
        df_prev = pd.read_csv(os.path.join(input_dir, 'prevalence', f"{self.start_year}.csv"), index_col=0)
        self.df_population = df_prev.iloc[[-2, -1, 0], :]  # Birth count, survivors, care count
        self.df_prev = df_prev.iloc[1:-2, :]  # Each disease prevalence
        self.disease_list = self.df_prev.index  # List of diseases
        self.df_beta = pd.read_csv(os.path.join(input_dir, 'transition_rate', "beta_2019.csv"), index_col=0)
        self.df_mu = pd.read_csv(os.path.join(input_dir, 'transition_rate', "mu_2019.csv"), index_col=0)
        self.df_birth_num = pd.read_csv(os.path.join(input_dir, 'birth_num', "birth_num_middle.csv"))  # Birth count data, read without index

# Class to set initial condition + newborn state
class InitialCondition:
    def __init__(self, context, which_overlap):
        self.context = context  # Load class managing common variables
        self.which_overlap = which_overlap  # Specify overlap state
        self.update_initial_condition()  # Set initial condition
        
    def update_initial_condition(self):
        # Create array to store agent state
        # Overall health state
        num_agent_each_age = self.context.num_agent_each_age
        self.agent_whole_health = np.zeros((num_agent_each_age, self.context.covered_age))  # Agent health state (0:healthy, 1:care, 2:death)
        self.agent_diseases = []
        
        # State of each disease
        for disease in self.context.disease_list:
            self.agent_diseases.append(np.zeros((num_agent_each_age, self.context.covered_age)))  # Agent disease state (0:healthy, 1:care, 2:death)

        # Set initial state for each age
        for age in range(self.context.covered_age):
            age_group = age // self.context.group_width  # Convert to 5-year age groups
            alive_num = self.context.df_population.iloc[1, age_group]  # Number of survivors
            cumulative_birth = self.context.df_population.iloc[0, age_group]  # Cumulative births

            # Judge if agents are dead overall
            alive_ratio = alive_num / cumulative_birth
            self.agent_whole_health[:, age] = 2 * (np.random.rand(num_agent_each_age) > alive_ratio).astype(int)

            # Set initial disease state based on overlap type
            # Initial disease format: row->agent, column->disease
            if self.which_overlap == 1:
                initial_disease = self.overlape_type1(age)
            elif self.which_overlap == 2:
                initial_disease = self.overlape_type2(age)

            # Update overall health state
            self.agent_whole_health[:, age] += np.any(initial_disease == 1, axis=1).astype(int)            
            
            # Set initial state of each disease
            for i_disease in range(len(self.context.disease_list)):
                self.agent_diseases[i_disease][: , age] = initial_disease[:, i_disease]

        # Save created initial state
        # self.record_each_age_group()
    
    def update_newborn_state(self):
        # Set newborn state
        num_agent_each_age = self.context.num_agent_each_age
        self.newborn_state = np.zeros(num_agent_each_age)
        self.newborn_diseases = np.zeros((num_agent_each_age, len(self.context.disease_list)))
        
        age_group =  0  # Convert to 5-year age groups
        alive_num = self.context.df_population.iloc[1, age_group]  # Number of survivors
        cumulative_birth = self.context.df_population.iloc[0, age_group]  # Cumulative births

        # Judge if agents are dead overall
        alive_ratio = alive_num / cumulative_birth
        self.newborn_state = 2 * (np.random.rand(num_agent_each_age) > alive_ratio).astype(int)

        # Set initial disease state based on overlap type
        if self.which_overlap == 1:
            initial_disease = self.overlape_type1(0)
        elif self.which_overlap == 2:
            initial_disease = self.overlape_type2(0)
        
        # Update overall health state
        self.newborn_state += np.any(initial_disease == 1, axis=1).astype(int)            
        
        # Set initial state of each disease
        for i_disease in range(len(self.context.disease_list)):
            self.newborn_diseases[:, i_disease] = initial_disease[:, i_disease]

    # Overlap type 1: Completely random
    def overlape_type1(self, age):
        num_agent_each_age = self.context.num_agent_each_age
        age_group = age // self.context.group_width  # Convert to 5-year age groups
        alive_num = self.context.df_population.iloc[1, age_group]  # Number of survivors
        df_age_group = self.context.df_prev.iloc[:, age_group]  # Prevalence of target age group

        # Set initial disease state (row->agent, column->disease)
        initial_disease = np.zeros((num_agent_each_age, len(df_age_group)))
        for i_disease in range(len(df_age_group)):
            prev_ratio = df_age_group.iloc[i_disease] / alive_num
            initial_disease[:,i_disease] = (np.random.rand(num_agent_each_age) < prev_ratio).astype(int)
            initial_disease[:,i_disease] = np.where(self.agent_whole_health[:,age] == 2, 2, initial_disease[:, i_disease])  # Dead agents remain dead
        return initial_disease

    # Allocate diseases among agents with diseases
    # & : numpy logical AND, | : numpy logical OR
    def overlape_type2(self, age):
        num_agent_each_age = self.context.num_agent_each_age
        age_group = age // self.context.group_width  # Convert to 5-year age groups
        alive_num = self.context.df_population.iloc[1, age_group]  # Number of survivors
        care_num = self.context.df_population.iloc[2, age_group]  # Number in care
        df_age_group = self.context.df_prev.iloc[:, age_group]  # Prevalence of target age group

        care_ratio = care_num / alive_num
        agent_whole_health_temp = np.where(self.agent_whole_health[:,age] == 2, 2, (np.random.rand(num_agent_each_age) < care_ratio).astype(int))  # Agents with disease
        initial_disease = np.zeros((num_agent_each_age, len(df_age_group)))
        for i_disease in range(len(df_age_group)):
            prev_ratio_in_care = df_age_group.iloc[i_disease] / care_num
            initial_disease[:,i_disease] = ((agent_whole_health_temp == 1) & (np.random.rand(num_agent_each_age) < prev_ratio_in_care)).astype(int)
            initial_disease[:,i_disease] = np.where(self.agent_whole_health[:,age] == 2, 2, initial_disease[:, i_disease])  # Dead agents remain dead
        return initial_disease
    
    # Record proportion at initial state for each 5-year age group in csv file
    def record_each_age_group(self, dir_name):
        group_num = self.context.group_num
        group_width = self.context.group_width
        num_agent_each_age = self.context.num_agent_each_age
        alive_num = self.context.df_population.iloc[1, :]  # Number of survivors
        cumulative_birth = self.context.df_population.iloc[0, :]  # Cumulative births
        care_num = self.context.df_population.iloc[2, :]  # Number in care
        col_name = []
        for i_group in range(group_num):
            col_name.append(f"{i_group*group_width}~{(i_group+1)*group_width-1}")
        df_record = pd.DataFrame(columns=col_name)

        # Proportion of overall health state
        # Proportion of health state for each age
        x_c = np.sum(self.agent_whole_health == 1, axis=0) / num_agent_each_age
        x_d = np.sum(self.agent_whole_health == 2, axis=0) / num_agent_each_age
        
        # Convert to 5-year age groups
        x_c_group = np.zeros(group_num)
        x_d_group = np.zeros(group_num)
        for i_group in range(group_num):
            x_c_group[i_group] = np.average(x_c[i_group*group_width:(i_group+1)*group_width])
            x_d_group[i_group] = np.average(x_d[i_group*group_width:(i_group+1)*group_width])
        x_h_group = 1 - x_c_group - x_d_group

        # Actual proportions
        x_c_group_actual = care_num / cumulative_birth
        x_d_group_actual = (cumulative_birth - alive_num) / cumulative_birth
        x_d_group_actual = np.where(x_d_group_actual < 0, 0, x_d_group_actual)
        x_c_group_actual = x_c_group_actual[:len(x_c_group)]
        x_d_group_actual = x_d_group_actual[:len(x_d_group)]
        x_h_group_actual = 1 - x_c_group_actual - x_d_group_actual

        df_record.loc["healthy_model"] = x_h_group
        df_record.loc["healthy_actual"] = x_h_group_actual
        df_record.loc["care_model"] = x_c_group
        df_record.loc["care_actual"] = x_c_group_actual
        df_record.loc["death_model"] = x_d_group
        df_record.loc["death_actual"] = x_d_group_actual

        # Proportion by disease
        # Proportion of disease for each age
        for i_disease, disease in enumerate(self.context.disease_list):
            x_c_disease = np.sum(self.agent_diseases[i_disease] == 1, axis=0) / num_agent_each_age

            # Convert to 5-year age groups
            x_c_disease_group = np.zeros(group_num)
            for i_group in range(group_num):
                x_c_disease_group[i_group] = np.average(x_c_disease[i_group*group_width:(i_group+1)*group_width])
            
            # Actual proportion
            prev_patient_num = self.context.df_prev.iloc[i_disease, :]
            x_c_disease_group_actual = prev_patient_num / cumulative_birth
            x_c_disease_group_actual = x_c_disease_group_actual[:len(x_c_disease_group)]

            df_record.loc[f"{disease}_model"] = x_c_disease_group
            df_record.loc[f"{disease}_actual"] = x_c_disease_group_actual
        
        df_record.to_csv(f"{dir_name}/{self.context.start_year}(initial_condition_type{self.which_overlap}).csv")

# Class representing each disease
class DiseaseClass:
    # Initialization (specify disease type)
    def __init__(self, i_disease, context, initial_condition, simulation_whole):
        self.i_disease = i_disease  # Index of disease among all diseases (ex) 0:Diabetes
        self.context = context
        self.disease_name = self.context.disease_list[i_disease]  # Disease name
        self.initial_condition = initial_condition  # Load class setting initial condition
        self.simulation_whole = simulation_whole  # Load class managing overall simulation
        self.load_initial_condition()

    # Set initial condition (health state due to disease, 0, 1, 2 three states (not infected, infected, dead))
    def load_initial_condition(self):
        self.agent_state = self.initial_condition.agent_diseases[self.i_disease].copy()
        # Structure of self.agent_state: row->agent, column->age
    
    def update_transition_rate(self, array_of_beta, array_of_mu):
        self.array_of_beta = array_of_beta.values
        self.array_of_mu = array_of_mu.values
    
    # Operation to age (shift array by 1)
    def aging(self):
        self.agent_state = np.roll(self.agent_state, 1, axis=1)
        self.agent_state[:, 0] = self.initial_condition.newborn_diseases[:, self.i_disease]

    # Update state
    def disease_progression(self):
        self.num_agent_for_each_age = self.context.num_agent_each_age
        for age in range(self.context.covered_age):
            age_group = age // self.context.group_width  # Convert to 5-year age groups
            beta_this_age = self.array_of_beta[age_group]
            mu_this_age = self.array_of_mu[age_group]

            # Death from disease, (probability mu_this_age)*(number of people with disease)
            self.agent_state[:, age] += (np.random.rand(self.num_agent_for_each_age) < mu_this_age).astype(int) * (self.agent_state[:, age] == 1).astype(int)

            # Disease infection
            # When net infection rate is positive
            if beta_this_age > 0:
                self.agent_state[:, age] += (np.random.rand(self.num_agent_for_each_age) < beta_this_age).astype(int) * (self.agent_state[:, age] == 0).astype(int)
            # When net infection rate is negative (recovery) *Modify here if infection rate negative places are rewritten with care as base
            else:
                # Number of recovered = (probability (-beta_this_age))*(number of people without disease)
                num_cured_people = np.sum((np.random.rand(self.num_agent_for_each_age) < -beta_this_age).astype(int) * (self.agent_state[:, age] == 0).astype(int))
                # Get index of people with disease
                indices = np.where(self.agent_state[:, age] == 1)[0]
                indices = np.random.permutation(indices)  # Randomly shuffle index of people with disease
                random_indices = np.random.permutation(indices)
                # Cure as many people as num_cured_people from those with disease (considering not to become negative)
                for i_index in range(min(len(random_indices), num_cured_people)):
                    self.agent_state[random_indices[i_index], age] = 0
    
    # Make dead people in overall health state also dead in each disease state
    def death_by_others(self):
        self.agent_state = np.where(self.simulation_whole.agent_whole_health == 2, 2, self.agent_state)
    
    # Return prevalence for each age
    def return_prev(self):
        return np.sum(self.agent_state == 1, axis=0) / self.context.num_agent_each_age

# Class managing overall simulation
class SimulationWhole:
    def __init__(self, context, initial_condition, dir_name_mother):
        self.context = context
        self.initial_condition = initial_condition
        self.dir_name_mother = dir_name_mother
        self.disease_classes_make()
        self.new_simulation()

    # Perform initial setup when conducting new simulation at once
    def new_simulation(self):
        self.initial_condition.update_initial_condition()  # Update initial condition
        self.load_initial_condition()
        self.make_first_demographics()
        self.simu_year = self.context.start_year  # Year to start simulation
        
        # Create directory to save this simulation
        now = datetime.now()
        # Output in yymmddhhmm format
        formatted_now = now.strftime('%y%m%d%H%M%S')
        self.dir_name = os.path.join(self.dir_name_mother, formatted_now)
        os.makedirs(self.dir_name, exist_ok=True)  # Create output directory
        
        # Save initial state
        # self.initial_condition.record_each_age_group(dir_name=self.dir_name)

    # Create class for simulation of each disease
    def disease_classes_make(self):
        ctx = self.context
        self.disease_classes = []
        for i_disease in range(len(ctx.disease_list)):
            self.disease_classes.append(DiseaseClass(i_disease, ctx, self.initial_condition, self))
            self.disease_classes[i_disease].update_transition_rate(ctx.df_beta.iloc[i_disease, :], ctx.df_mu.iloc[i_disease, :])

    # Initialize demographics
    def make_first_demographics(self):
        # Set initial population
        self.demographics = np.zeros(self.context.covered_age)
        self.simu_now_row = 0  # Row number in birth count database
        while self.context.df_birth_num.iloc[self.simu_now_row, 0] < self.context.start_year:
            self.simu_now_row += 1
        
        for i in range(self.context.covered_age):
            self.demographics[i] = self.context.df_birth_num.iloc[self.simu_now_row - i, 1]
    
    # Initialize health state
    def load_initial_condition(self):
        # Overall
        self.agent_whole_health = self.initial_condition.agent_whole_health.copy()
        
        # Each disease
        for diesease in self.disease_classes:
            diesease.load_initial_condition()
    
    # Conduct simulation
    def simulation_conduct(self):
        self.record_each_age_group()  # Save initial state
        for t_simu in range(1, self.context.time_span+1):
            self.health_transition()
            # print(f"Year {t_simu} simulation completed")
            if t_simu % self.context.record_span == 0:
                self.record_each_age_group()
    
    # Update demographics
    def update_demographics(self):
        self.simu_now_row += 1
        self.demographics = np.roll(self.demographics, 1)
        self.demographics[0] = self.context.df_birth_num.iloc[self.simu_now_row, 1]

    # Operation to age (shift array by 1)
    def aging_whole(self):
        self.agent_whole_health = np.roll(self.agent_whole_health, 1, axis=1)
        self.agent_whole_health[:, 0] = self.initial_condition.newborn_state

    # Change health state
    def health_transition(self):
        self.simu_year = self.simu_year + 1
        # Update state of 0-year-old agents (randomly generate)
        self.initial_condition.update_newborn_state()

        # Update progression state for each disease
        for disease in self.disease_classes:
            disease.disease_progression()

        # Update overall health state
        for age in range(self.context.covered_age):
            # Combine disease progression states for target age into one
            state_each_disease = np.zeros((self.context.num_agent_each_age, len(self.disease_classes)))
            for disease in self.disease_classes:
                state_each_disease[:, disease.i_disease] = disease.agent_state[:, age] 

            # Categorize state
            self.agent_whole_health[:, age] = np.zeros(self.context.num_agent_each_age)  # First reset everyone to healthy state
            self.agent_whole_health[:, age] += np.any(state_each_disease == 1, axis=1).astype(int)  # Become care state
            self.agent_whole_health[:, age] = np.where(np.any(state_each_disease == 2, axis=1), 2, self.agent_whole_health[:, age])  # Become death state

        # Make dead people in overall health also dead in each disease
        self.initial_condition.update_newborn_state()  # Update newborn state (added 12/24)
        for disease in self.disease_classes:
            disease.death_by_others()
            disease.aging()
        
        self.aging_whole()
        self.update_demographics()
    
    # Save records
    def record_each_age_group(self):
        group_num = self.context.group_num
        group_width = self.context.group_width
        num_agent_each_age = self.context.num_agent_each_age
        col_name = []
        for i_group in range(group_num):
            col_name.append(f"{i_group*group_width}~{(i_group+1)*group_width-1}")
        df_record = pd.DataFrame(columns=col_name)

        # Proportion of overall health state
        # Proportion of health state for each age
        x_c = np.sum(self.agent_whole_health == 1, axis=0) / num_agent_each_age
        x_d = np.sum(self.agent_whole_health == 2, axis=0) / num_agent_each_age
        
        # Convert to 5-year age groups
        demographic_each_group = np.zeros(group_num)  # Population for each 5-year age group
        x_c_each_group = np.zeros(group_num)  # Proportion of care for each 5-year age group
        x_d_each_group = np.zeros(group_num)  # Proportion of death for each 5-year age group
        for i_group in range(group_num):
            # Get data for ages in target group
            demographic_group = self.demographics[i_group*group_width:(i_group+1)*group_width] 
            x_c_group = x_c[i_group*group_width:(i_group+1)*group_width]
            x_d_group = x_d[i_group*group_width:(i_group+1)*group_width]
            
            # Calculate weighted proportions based on population for each age
            demographic_each_group[i_group] = np.sum(demographic_group)
            x_c_each_group[i_group] = np.dot(demographic_group, x_c_group) / np.sum(demographic_group)
            x_d_each_group[i_group] = np.dot(demographic_group, x_d_group) / np.sum(demographic_group)

        x_h_each_group = 1 - x_c_each_group - x_d_each_group

        # Save overall proportions
        df_record.loc["demography"] = demographic_each_group
        df_record.loc["healthy"] = x_h_each_group
        df_record.loc["care"] = x_c_each_group
        df_record.loc["death"] = x_d_each_group
        
        # Proportion by disease
        # Proportion of disease for each age
        for disease in self.disease_classes:
            x_c_disease = disease.return_prev()

            # Convert to 5-year age groups
            x_c_disease_each_group = np.zeros(group_num)
            for i_group in range(group_num):
                demographic_group = self.demographics[i_group*group_width:(i_group+1)*group_width] 
                x_c_disease_group = x_c_disease[i_group*group_width:(i_group+1)*group_width]
                x_c_disease_each_group[i_group] = np.dot(demographic_group, x_c_disease_group) / np.sum(demographic_group)
            df_record.loc[f"{disease.disease_name}"] = x_c_disease_each_group
            
        df_record.to_csv(f"{self.dir_name}/{self.simu_year}.csv")