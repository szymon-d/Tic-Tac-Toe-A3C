import os


#Agents settings
settings = {
    1: {
        'epsilon': 0.1,
        'lr': 1e-4,
        'gamma': 0.5
    },
    -1: {
        'epsilon': 0.2,
        'lr': 1e-4,
        'gamma': 0.5
    },
}

#Player mapper
mapper = {-1: 'O',
          1: 'X',
          0: '_'}

CWD = os.path.dirname(__file__)
#Directory with models
MODELS_DIR = os.path.join(CWD, 'models')