from arg_parser import parse_args, load_params
import sys
import sweet_tools
import numpy as np
import pandas as pd

# Create a fake list of command line arguments
sys.argv = ['', '--config', 'C:/Users/hughr/OneDrive/Documents/RMI/What_a_Waste/city_level_data_0_0.csv']

# call parse_args()
args = parse_args()

# And use args to load parameters
params = load_params(args.config)

runs = {}

for idx in params.keys():
    
    # Load parameters
    run_params = params[idx]
    
    # if run_params['city'] != 'Liege':
    #     continue
    
    # Get diversion volumes
    run_params, compost_vol = sweet_tools.calc_compost_vol(run_params)
    run_params, anaerobic_vol = sweet_tools.calc_anaerobic_vol(run_params)
    run_params, combustion_vol = sweet_tools.calc_combustion_vol(run_params)
    run_params, recycling_vol = sweet_tools.calc_recycling_vol(run_params)
    
    for c in run_params['waste_fractions'].keys():
        if c not in compost_vol.keys():
            compost_vol[c] = 0
        if c not in anaerobic_vol.keys():
            anaerobic_vol[c] = 0
        if c not in combustion_vol.keys():
            combustion_vol[c] = 0
        if c not in recycling_vol.keys():
            recycling_vol[c] = 0
            
    if run_params['compost_total'] != run_params['compost_total']:
        break
    
    divs = {'compost': compost_vol, 'anaerobic': anaerobic_vol, 'combustion': combustion_vol, 'recycling': recycling_vol}
    
    # if run_params['city'] == 'Liege':
    #     break
    
    divs = check_masses(run_params, divs)
            
    qs = {}
    ms = {}
    masses_compost = {}
    masses_anaerobic = {}
    q_dfs = {}
    m_dfs = {}
    organic_df = {}
    
    # One iteration each for landfill and dump site
    for site in run_params['sites']:
        # Set up for next step
        mcf = run_params['mcfs'][site]
        qs[site] = {}
        ms[site] = {}
        q_dfs[site] = {}
        m_dfs[site] = {}
        
        # Loop through waste components
        for waste in run_params['components']:
            # Continue setup...
            ms[site][waste] = []
            qs[site][waste] = {}
            if site == 'landfill':
                masses_compost[waste] = []
                masses_anaerobic[waste] = []
                organic_df[waste] = {}
            
            # Loop through years
            for t in range(50):
                ms[site][waste].append((run_params['waste_mass'] * 
                                        run_params['waste_fractions'][waste] - 
                                        divs['compost'][waste] - 
                                        divs['anaerobic'][waste] - 
                                        divs['combustion'][waste] - 
                                        divs['recycling'][waste]) 
                                        * run_params['split_fractions'][site] * (1.03 ** t))
                
                # Only have collect compost and anaerobic masses once as they don't go to either site
                if site == 'landfill':
                    masses_compost[waste].append(divs['compost'][waste] * (1.03 ** t))
                    masses_anaerobic[waste].append(divs['anaerobic'][waste] * (1.03 ** t))
                
                # Loop through previous years to get methane after decay
                ch4 = []
                for y in range(t):
                    val = run_params['ks'][waste] * \
                          sweet_tools.L_0[waste] * \
                          ms[site][waste][y] * \
                          np.exp(-run_params['ks'][waste] * \
                          (t - y - 0.5)) * \
                          mcf * \
                          (1 - sweet_tools.oxidation_factor['without_lfg'][site])
                    ch4.append(val)
                    
                # Sum CH4 for waste from all previous years
                qs[site][waste][t] = sum(ch4)
                if site == 'landfill':
                    organic_df[waste][t] = masses_compost[waste][t] * run_params['mef_compost'] + \
                                           masses_anaerobic[waste][t] * sweet_tools.mef_anaerobic * sweet_tools.ch4_to_co2e
                
        
        q_dfs[site] = pd.DataFrame(qs[site])
        q_dfs[site]['total'] = q_dfs[site].sum(axis=1)
        m_dfs[site] = pd.DataFrame(ms[site])
        compost_df = pd.DataFrame(masses_compost)
        anaerobic_df = pd.DataFrame(masses_anaerobic)
        organic_df = pd.DataFrame(organic_df) #.applymap(sweet_tools.convert_methane_m3_to_ton_co2e)
        organic_df['total'] = organic_df.sum(axis=1)
        
        
    runs[idx] = {}
    runs[idx]['q'] = q_dfs
    runs[idx]['m'] = m_dfs
    runs[idx]['organic'] = pd.DataFrame(organic_df)
    
    params[idx] = run_params
    
    if run_params['city'] == 'Liege':
        break

#%%

import copy

#divs = {'compost': compost_vol, 'anaerobic': anaerobic_vol, 'combustion': combustion_vol, 'recycling': recycling_vol}
#new_divs = {'compost': compost_vol.copy(), 'anaerobic': anaerobic_vol.copy(), 'combustion': combustion_vol.copy(), 'recycling': recycling_vol.copy()}

def check_masses(run_params, divs):
    # ok...how do I do this. Fudge.
    #old_divs = copy.deepcopy(divs)
    #new_divs = copy.deepcopy(divs)
    masses = {x: run_params['waste_fractions'][x] * run_params['waste_mass'] for x in run_params['waste_fractions'].keys()}
    
    fractions_before = {}
    for div in divs.keys():
        fractions_before[div] = sum([x for x in divs[div].values()])/run_params['waste_mass']
    
    problems = [set()]
    net_masses = {}
    for waste in masses.keys():
        net_mass = masses[waste] - (divs['compost'][waste] + divs['anaerobic'][waste] + divs['combustion'][waste] + divs['recycling'][waste])
        net_masses[waste] = net_mass
        if net_mass < 0:
            problems[0].add(waste)
    dont_add_to = problems[0].copy()
    #old_net_masses = copy.deepcopy(net_masses)
    
    while problems:
        probs = problems.pop(0)
        for waste in probs:
            deficit = -net_masses[waste]
            total_subtracted = divs['compost'][waste] + divs['anaerobic'][waste] + divs['recycling'][waste]
            # fractions = [compost_vol[waste] / total_subtracted, 
            #              anaerobic_vol[waste] / total_subtracted, 
            #              combustion_vol[waste] / total_subtracted, 
            #              recycling_vol[waste] / total_subtracted]
            
            fraction_to_fix = deficit / total_subtracted
            # add_back_amounts = {'compost': compost_vol[waste] * fraction_to_fix, 
            #                     'anaerobic': anaerobic_vol[waste] * fraction_to_fix, 
            #                     'combustion': combustion_vol[waste] * fraction_to_fix, 
            #                     'recycling': recycling_vol[waste] * fraction_to_fix}
            
            add_back_amounts = {}
            for div in divs.keys():
                if div == 'compost':
                    if waste in run_params['unprocessable']:
                        add_back_amounts[div] = divs[div][waste] * fraction_to_fix / \
                                                (1 - run_params['non_compostable_not_targeted_total']) / \
                                                (1 - run_params['unprocessable'][waste])
                    else:
                        assert divs[div][waste] == 0, 'Hope this doesnt happen'
                        add_back_amounts[div] = 0
                elif div == 'combustion':
                    continue
                    # add_back_amounts[div] = divs[div][waste] * fraction_to_fix / \
                    #                         (1 - run_params['combustion_reject_rate'])
                elif div == 'recycling':
                    if waste in run_params['recycling_reject_rates']:
                        add_back_amounts[div] = divs[div][waste] * fraction_to_fix / \
                                                (run_params['recycling_reject_rates'][waste])
                    else:
                        assert divs[div][waste] == 0, 'Hope this doesnt happen'
                        add_back_amounts[div] = 0
                else:
                    add_back_amounts[div] = divs[div][waste] * fraction_to_fix
                    
                # Don't adjust the amount subtracted by the efficiency losses, this is the important part
                divs[div][waste] -= divs[div][waste] * fraction_to_fix
                
            # compost_vol[waste] -= compost_vol[waste] * fraction_to_fix
            # anaerobic_vol[waste] -= anaerobic_vol[waste] * fraction_to_fix
            # combustion_vol[waste] -= combustion_vol[waste] * fraction_to_fix
            # recycling_vol[waste] -= recycling_vol[waste] * fraction_to_fix
            
            for div in divs.keys():
                if div == 'combustion':
                    assert div not in add_back_amounts.keys(), 'This should be zero'
                    continue
                amount = add_back_amounts[div]
                if amount == 0:
                    continue
                types_to_add_to = [x for x in run_params[str(div) + '_waste_fractions'].keys() if x not in dont_add_to]
                fraction_of_types_adding_to = sum([run_params[str(div) + '_waste_fractions'][x] for x in types_to_add_to])
                for w in types_to_add_to:
                    if div == 'compost':
                        divs[div][w] += amount * run_params[str(div) + '_waste_fractions'][w] / fraction_of_types_adding_to * \
                                        (1 - run_params['non_compostable_not_targeted_total']) * \
                                        (1 - run_params['unprocessable'][w])                                
                    # elif div == 'combustion':
                    #     #continue
                    #     divs[div][w] += amount * run_params[str(div) + '_waste_fractions'][w] / fraction_of_types_adding_to * \
                    #                     (1 - run_params['combustion_reject_rate'])
                    elif div == 'recycling':
                        divs[div][w] += amount * run_params[str(div) + '_waste_fractions'][w] / fraction_of_types_adding_to * \
                                        (run_params['recycling_reject_rates'][w])
                    else:
                        divs[div][w] += amount * run_params[str(div) + '_waste_fractions'][w] / fraction_of_types_adding_to
                    
        net_masses = {}
        new_probs = set()
        for waste in masses.keys():
            net_mass = masses[waste] - (divs['compost'][waste] + divs['anaerobic'][waste] + divs['combustion'][waste] + divs['recycling'][waste])
            net_masses[waste] = net_mass
            if (net_mass < -0.1):
                new_probs.add(waste)
                dont_add_to.add(waste)
                
        if len(new_probs) > 0:
            problems.append(new_probs)
    
    fractions_after = {}
    for div in divs.keys():
        fractions_after[div] = sum([x for x in divs[div].values()])/run_params['waste_mass']
        
    # for div in divs.keys():
    #     assert (fractions_before[div] - fractions_after[div]) < .01, 'total diversion fractions should not change'
        
    original_fractions = {'compost': {}}
    for waste in run_params['compost_components']:
        final_mass_after_adjustment = divs['compost'][waste]
        input_mass_after_adjustment = final_mass_after_adjustment / \
                                      (1 - run_params['non_compostable_not_targeted_total']) / \
                                      (1 - run_params['unprocessable'][waste])
        try:
            original_fractions['compost'][waste] = input_mass_after_adjustment / run_params['compost_total']
            print(run_params['city'], run_params['compost_total'])
        except:
            original_fractions['compost'][waste] = np.nan
        
    original_fractions['anaerobic'] = {}
    for waste in run_params['anaerobic_components']:
        final_mass_after_adjustment = divs['anaerobic'][waste]
        try:
            original_fractions['anaerobic'][waste] = final_mass_after_adjustment / run_params['anaerobic_total']
        except:
            original_fractions['anaerobic'][waste] = np.nan
        
    original_fractions['combustion'] = {}
    for waste in run_params['combustion_components']:
        final_mass_after_adjustment = divs['combustion'][waste]
        input_mass_after_adjustment = final_mass_after_adjustment / \
                                      (1 - run_params['combustion_reject_rate'])
        try:
            original_fractions['combustion'][waste] = input_mass_after_adjustment / run_params['combustion_total']
        except:
            original_fractions['combustion'][waste] = np.nan
        
    original_fractions['recycling'] = {}
    for waste in run_params['recycling_components']:
        final_mass_after_adjustment = divs['recycling'][waste]
        input_mass_after_adjustment = final_mass_after_adjustment / \
                                      (run_params['recycling_reject_rates'][waste])
        try:
            original_fractions['recycling'][waste] = input_mass_after_adjustment / run_params['recycling_total']
        except:
            original_fractions['recycling'][waste]
        
    return divs

#%%

mo = pd.DataFrame(masses_compost)
ma = pd.DataFrame(masses_anaerobic)

sum([x for x in divs['combustion'].values()])

q_dfs['dump_site'].applymap(sweet_tools.convert_methane_m3_to_ton_co2e)

divs['combustion']['food']
old_divs['combustion']['food']
run_params['waste_mass'] * run_params['waste_fractions']['food'] * run_params['combustion_fraction']

run_params['waste_mass'] * run_params['waste_fractions']['food'] - divs['compost']['food'] - divs['anaerobic']['food'] - divs['combustion']['food']
            
sum([x for x in add_back_amounts.values()])

sum([x for x in run_params['waste_fractions'].values()])
sum([x for x in masses.values()])
sum([x for x in net_masses.values()])
sum([x for x in new_net_masses.values()])

run_params['waste_fractions']
run_params['waste_mass'] / run_params['population'] * 1000 / 365

sum([x for x in compost_vol.values()])
sum([x for x in new_compost_vol.values()])

sum([x for x in anaerobic_vol.values()])
sum([x for x in new_anaerobic_vol.values()])

run_params['compost_waste_fractions']
run_params['split_fractions']

sum([x for x in combustion_vol.values()])
sum([x for x in new_combustion_vol.values()])

sum([x for x in recycling_vol.values()])
sum([x for x in new_recycling_vol.values()])

run_params['combustion_waste_fractions']
sum([x for x in run_params['combustion_waste_fractions'].values()])
run_params['recycling_waste_fractions']

for c in run_params['components']:
    print(c, run_params['waste_fractions'][c] * run_params['waste_mass'],
          compost_vol[c],
          anaerobic_vol[c],
          combustion_vol[c],
          recycling_vol[c])

run_params['compost_fraction'] * run_params['waste_mass']
run_params['anaerobic_fraction'] * run_params['waste_mass']
run_params['combustion_fraction'] * run_params['waste_mass']
run_params['recycling_fraction'] * run_params['waste_mass']

sum([x for x in original_fractions['compost'].values()])
sum([x for x in original_fractions['anaerobic'].values()])
sum([x for x in original_fractions['combustion'].values()])
sum([x for x in original_fractions['recycling'].values()])
