import matplotlib.cm as cm

alg_naming = {
    "adahedged": "AdaHedgeD",
    "dorm": "DORM",
    "dormplus": "DORM+",
    "dub": "DUB"
}

model_alias = {
    "tuned_catboost": "LocalBoosting",
    "tuned_doy": "Climatology++", 
    "tuned_cfsv2": "CFSv2++",
    "tuned_salient_fri": "Salient++",
    "llr": "Persistence++",
    "multillr": "MultiLLR",
}

linestyle_tuple = [
     ('dashed',                (0, (5, 5))),
     ('dotted',                (0, (1, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('densely dashed',        (0, (5, 1))),
     ('dashdotted',            '-'),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('loosely dashed',        (0, (5, 10))),
     ('loosely dotted',        (0, (1, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
line_types = ['-'] + [x[1] for x in linestyle_tuple]

# Colorblind colors 
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

style_algs = {
    'DORM+' : {'linestyle': line_types[0], 'color': CB_color_cycle[0], 'linewidth': 2},
    'DUB' : {'linestyle': line_types[4], 'color': CB_color_cycle[1], 'linewidth': 2},
    'AdaHedgeD' : {'linestyle': line_types[3], 'color': CB_color_cycle[3], 'linewidth': 2},
    'DORM' : {'linestyle': line_types[2], 'color': CB_color_cycle[2], 'linewidth': 2},
    'Replicated DORM+' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'DORM+ Repl.' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'prev\_g' : {'linestyle': line_types[3], 'color': CB_color_cycle[3], 'linewidth': 2},
    'mean\_g' : {'linestyle': line_types[1], 'color': CB_color_cycle[1], 'linewidth': 2},
    'none' : {'linestyle': line_types[2], 'color': CB_color_cycle[2], 'linewidth': 2},
    'learned' : {'linestyle': line_types[0], 'color': CB_color_cycle[0], 'linewidth': 2},
    'LocalBoosting' : {'linestyle': line_types[0], 'color': CB_color_cycle[0], 'linewidth': 2},
    'Climatology++' : {'linestyle': line_types[1], 'color': CB_color_cycle[1], 'linewidth': 2},
    'CFSv2++' : {'linestyle': line_types[2], 'color': CB_color_cycle[2], 'linewidth': 2},
    'Persistence++' : {'linestyle': line_types[3], 'color': CB_color_cycle[3], 'linewidth': 2},
    'MultiLLR' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'Salient++' : {'linestyle': line_types[5], 'color': CB_color_cycle[5], 'linewidth': 2},
}