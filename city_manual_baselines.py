import copy
import numpy as np

manual_cities = set(['Milano', 'Yokohama', 'Kobe', 'Naha', 'Kitakyushu', 'Toyama', 'Oslo', 'Stockholm', 'Liege'])

def get_manual_baseline(city):
    div_multiplied_through = {}

    for div, fracts in city.div_component_fractions.items():
        div_multiplied_through[div] = {}
        for waste in fracts:
            div_multiplied_through[div][waste] = city.div_fractions[div] * city.div_component_fractions[div][waste]
    
    div_multiplied_through_adjusted = copy.deepcopy(div_multiplied_through)
    city.div_component_fractions_adjusted = copy.deepcopy(city.div_component_fractions)

    net = {}
    for waste in city.waste_fractions:
        components = []
        for div in city.divs:
            try:
                component = city.div_fractions[div] * city.div_component_fractions[div][waste]
            except:
                component = 0
            components.append(component)
        s = sum(components)
        net[waste] = city.waste_fractions[waste] - s

    if city.name == 'Milano':
        city.div_component_fractions_adjusted['combustion']['wood'] -= (-net['wood'] / city.div_fractions['combustion'])
        city.div_component_fractions_adjusted['combustion']['food'] += (-net['wood'] / city.div_fractions['combustion'])

        city.div_component_fractions_adjusted['combustion']['paper_cardboard'] -= (-net['paper_cardboard'] / city.div_fractions['combustion'])
        city.div_component_fractions_adjusted['combustion']['food'] += (-net['paper_cardboard'] / city.div_fractions['combustion'])

        city.div_component_fractions_adjusted['combustion']['plastic'] -= (-net['plastic'] / city.div_fractions['combustion'])
        city.div_component_fractions_adjusted['combustion']['food'] += (-net['plastic'] / city.div_fractions['combustion'])

    if city.name == 'Yokohama':
        # This one nearly works, but plastic is like .1% too much diversion...can fudge it somewhere. Redo this one now that I reduce div to 100% max
        div_multiplied_through_adjusted['combustion']['wood'] -= -net['wood']
        div_multiplied_through_adjusted['combustion']['green'] += -net['wood']
        remaining_green = net['green'] - -net['wood']

        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= net['other']
        div_multiplied_through_adjusted['recycling']['other'] += net['other']
        remaining_paper = -net['paper_cardboard'] - net['other']

        div_multiplied_through_adjusted['combustion']['paper_cardboard'] -= remaining_paper
        div_multiplied_through_adjusted['combustion']['food'] += remaining_paper
        remaining_food = net['food'] - remaining_paper

        div_multiplied_through_adjusted['combustion']['plastic'] -= remaining_food
        div_multiplied_through_adjusted['combustion']['food'] += remaining_food

        div_multiplied_through_adjusted['combustion']['plastic'] -= remaining_green
        div_multiplied_through_adjusted['combustion']['green'] += remaining_green

        div_multiplied_through_adjusted['recycling']['plastic'] -= net['glass']
        div_multiplied_through_adjusted['recycling']['glass'] += net['glass']

        div_multiplied_through_adjusted['recycling']['plastic'] -= net['metal']
        div_multiplied_through_adjusted['recycling']['metal'] += net['metal']

    if city.name == 'Kobe':
        div_multiplied_through_adjusted['recycling']['rubber'] -= -net['rubber']
        div_multiplied_through_adjusted['recycling']['glass'] += -net['rubber']
        remaining_glass = net['glass'] - -net['rubber']

        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= -net['paper_cardboard']
        div_multiplied_through_adjusted['recycling']['other'] += -net['paper_cardboard']
        remaining_other = net['other'] - -net['paper_cardboard']

        div_multiplied_through_adjusted['recycling']['wood'] -= -net['wood']
        div_multiplied_through_adjusted['recycling']['metal'] += -net['rubber']
        remaining_metal = net['metal'] - -net['wood']

        div_multiplied_through_adjusted['recycling']['plastic'] -= remaining_other
        div_multiplied_through_adjusted['recycling']['other'] += remaining_other
        remaining_plastic = -net['plastic'] - remaining_other

        div_multiplied_through_adjusted['combustion']['plastic'] -= remaining_plastic
        div_multiplied_through_adjusted['combustion']['food'] += remaining_plastic

    if city.name == 'Naha':
        div_multiplied_through_adjusted = copy.deepcopy(div_multiplied_through)

        div_multiplied_through_adjusted['recycling']['plastic'] -= -net['plastic']
        div_multiplied_through_adjusted['recycling']['metal'] += -net['plastic']
        remaining_metal = net['metal'] - -net['plastic']

        div_multiplied_through_adjusted['combustion']['paper_cardboard'] -= (net['food'] + net['green'])
        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= (remaining_metal + net['other'])

        div_multiplied_through_adjusted['combustion']['food'] += net['food']
        div_multiplied_through_adjusted['combustion']['green'] += net['green']
        div_multiplied_through_adjusted['recycling']['metal'] += remaining_metal
        div_multiplied_through_adjusted['recycling']['other'] += net['other']

    if city.name == 'Kitakyushu':
        # This one leaves net paper_cardboard at -.001, but nowhere to put it...its fine? I think? Can reduce total max diversion from 100 to 99...
        # But then cities with full combustion will still get a landfill
        div_multiplied_through_adjusted['combustion']['plastic'] -= -net['plastic']
        div_multiplied_through_adjusted['combustion']['food'] += -net['plastic']
        remaining_food = net['food'] - -net['plastic']

        div_multiplied_through_adjusted['combustion']['paper_cardboard'] -= remaining_food
        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= (net['glass'] + net['metal'] + net['other'])

        div_multiplied_through_adjusted['combustion']['food'] += remaining_food
        div_multiplied_through_adjusted['recycling']['glass'] += net['glass']
        div_multiplied_through_adjusted['recycling']['metal'] += net['metal']
        div_multiplied_through_adjusted['recycling']['other'] += net['other']

    if city.name == 'Toyama':
        div_multiplied_through_adjusted['combustion']['rubber'] -= -net['rubber']
        div_multiplied_through_adjusted['combustion']['food'] += -net['rubber']
        remaining_food = net['food'] - -net['rubber']

        div_multiplied_through_adjusted['combustion']['wood'] -= -net['wood']
        div_multiplied_through_adjusted['combustion']['food'] += -net['wood']
        remaining_food = remaining_food - -net['wood']

        div_multiplied_through_adjusted['combustion']['plastic'] -= -net['plastic']
        div_multiplied_through_adjusted['combustion']['food'] += -net['plastic']
        remaining_food = remaining_food - -net['plastic']

        div_multiplied_through_adjusted['combustion']['paper_cardboard'] -= (remaining_food + net['green'])
        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= (net['glass'] + net['metal'] + net['other'])

        div_multiplied_through_adjusted['combustion']['food'] += remaining_food
        div_multiplied_through_adjusted['combustion']['green'] += net['green']
        div_multiplied_through_adjusted['recycling']['glass'] += net['glass']
        div_multiplied_through_adjusted['recycling']['metal'] += net['metal']
        div_multiplied_through_adjusted['recycling']['other'] += net['other']

    if city.name == 'Oslo':
        div_multiplied_through_adjusted['recycling']['wood'] -= -net['wood']
        div_multiplied_through_adjusted['recycling']['other'] += -net['wood']
        remaining_other = net['other'] - -net['wood']

        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= -net['paper_cardboard']
        div_multiplied_through_adjusted['recycling']['other'] += -net['paper_cardboard']
        remaining_other = remaining_other - -net['paper_cardboard']

        div_multiplied_through_adjusted['recycling']['textiles'] -= -net['textiles']
        div_multiplied_through_adjusted['recycling']['other'] += -net['textiles']
        remaining_other = remaining_other - -net['textiles']

        div_multiplied_through_adjusted['recycling']['plastic'] -= -net['plastic']
        div_multiplied_through_adjusted['recycling']['other'] += -net['plastic']
        remaining_other = remaining_other - -net['plastic']

    if city.name == 'Stockholm':
        div_multiplied_through_adjusted['recycling']['wood'] -= -net['wood']
        div_multiplied_through_adjusted['recycling']['other'] += -net['wood']
        remaining_other = net['other'] - -net['wood']

        div_multiplied_through_adjusted['recycling']['paper_cardboard'] -= -net['paper_cardboard']
        div_multiplied_through_adjusted['recycling']['other'] += -net['paper_cardboard']
        remaining_other = remaining_other - -net['paper_cardboard']

        div_multiplied_through_adjusted['recycling']['plastic'] -= -net['plastic']
        div_multiplied_through_adjusted['recycling']['other'] += -net['plastic']
        remaining_other = remaining_other - -net['plastic']

    if city.name == 'Liege':
        div_multiplied_through_adjusted['combustion']['wood'] -= -net['wood']
        div_multiplied_through_adjusted['combustion']['food'] += -net['wood']
        remaining_food = net['food'] - -net['wood']

        div_multiplied_through_adjusted['combustion']['paper_cardboard'] -= -net['paper_cardboard']
        div_multiplied_through_adjusted['combustion']['food'] += -net['paper_cardboard']
        remaining_food = remaining_food - -net['paper_cardboard']

        div_multiplied_through_adjusted['recycling']['textiles'] -= -net['textiles']
        div_multiplied_through_adjusted['recycling']['other'] += -net['textiles']
        remaining_other= net['other'] - -net['textiles']

        div_multiplied_through_adjusted['recycling']['plastic'] -= remaining_other
        div_multiplied_through_adjusted['recycling']['other'] += remaining_other
        remaining_plastic = -net['plastic'] - remaining_other

        div_multiplied_through_adjusted['recycling']['plastic'] -= remaining_plastic
        div_multiplied_through_adjusted['recycling']['paper_cardboard'] += remaining_plastic
        div_multiplied_through_adjusted['combustion']['paper_cardboard'] -= remaining_plastic
        div_multiplied_through_adjusted['combustion']['food'] += remaining_plastic

    if city.name != 'Milano':
        for div, fracs in div_multiplied_through_adjusted.items():
            for waste, f in fracs.items():
                if city.div_fractions[div] > 0:
                    city.div_component_fractions_adjusted[div][waste] = f / city.div_fractions[div]
                    city.divs[div][waste] = city.waste_mass * f
                    assert np.absolute((city.waste_mass * f) - \
                        (city.waste_mass * city.div_component_fractions_adjusted[div][waste] * city.div_fractions[div])) < 1e-3
                else:
                    city.div_component_fractions_adjusted[div][waste] = 0
                    city.divs[div][waste] = 0
    else:
        for div, fracs in city.div_component_fractions_adjusted.items():
            s = sum([x for x in fracs.values()])
            # make sure the component fractions add up to 1
            if (s != 0) and (np.absolute(1 - s) > 0.01):
                print(s, 'problems')
            for waste in fracs.keys():
                city.divs[div][waste] = city.waste_mass * city.div_fractions[div] * city.div_component_fractions_adjusted[div][waste]
