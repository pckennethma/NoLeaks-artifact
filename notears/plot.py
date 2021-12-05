import numpy as np
import matplotlib.pyplot as plt

data = {
    "Earthquake":{
        "noleak-case1": {
            "epsilon": [0.00489, 0.389, 3.07],
            "f1": [0.67, 1.0, 1.0]
        },
        "noleak-case2": {
            "epsilon": [0.059, 0.447, 5.099],
            "f1": [0.75, 1.0, 1.0]
        },
        "notears":{"f1": 1.0}
    },
    "Cancer":{
        "noleak-case1": {
            "epsilon": [0.035, 0.433, 5.64],
            "f1": [0.571, 0.571, 0.571]
        },
        "noleak-case2": {
            "epsilon": [0.033, 0.423, 4.13],
            "f1": [0.571, 0.571, 0.571]
        },
        "notears":{"f1": .571}
    },
    "Asia":{
        "noleak-case1": {
            "epsilon": [0.035, 0.433, 5.64],
            "f1": [0.571, 0.571, 0.571]
        },
        "noleak-case2": {
            "epsilon": [0.033, 0.423, 4.13],
            "f1": [0.571, 0.571, 0.571]
        },
        "notears":{"f1": .706}
    }
}