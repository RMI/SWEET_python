# Initiate parameter dictionary
params = {
    'city': 'Providence',
    'country': 'United States',
    'population': 1100000,
    'growth_rate': 1.03, # Need a lookup for this
    'waste_per_capita': None, # Unit is kg/person/day. Only use this or total, not both
    'waste_total': 500000, # Unit is kt/year
    'waste_fractions': {'food': 0.3, # These are fractions, should equal 1
                         'green': 0.1, 
                         'wood': 0.1,
                         'paper_cardboard': 0.1,
                         'plastic': 0.1,
                         'metal': 0.1,
                         'glass': 0.1,
                         'rubber': 0.05,
                         'other': 0.05,
                         },
    
    # Precipitation
    'precip': 1277.8, # mm

    # Depth
    'depth': 10, # m
    
    # Model components. These shouldn't change
    'components': set(['food', 'green', 'wood', 'paper_cardboard', 'textiles']),
    
    # Determine split between landfill and dump site
    'sites': ['landfill', 'dump_site'],
    'split_fractions': {'landfill': 0.4, 'dump_site': 0.4}, # Doesn't have to sum to 1, some waste gets composted etc.
    
    # Compost params
    'compost_components': set(['food', 'green', 'wood', 'paper_cardboard']), # Double check we don't want to include paper
    'compost_fraction': 0.1,
    'compost_mass': None, # Set this either way not both
    
    # Anaerobic params
    'anaerobic_components': set(['food', 'green', 'wood', 'paper_cardboard']),
    'anaerobic_fraction': 0.1,
    'anaerobic_mass': None, # Set this either way not both
    
    # Combustion params
    'combustion_components': set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber']),
    'combustion_fraction': 0.1,
    'combustion_mass': None, # Set this either way not both
    
    # Recycling params
    'recycling_components': set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other']),
    'recycling_fraction': 0.1,
    'recycling_mass': None # Set this either way not both
    }