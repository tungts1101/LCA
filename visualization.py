from matplotlib import pyplot as plt
import numpy as np


results = {
    "CIFAR100": {
        "MM - CIL":             {"mean": 92.75},
        "MM - CA":              {"mean": 94.00, "std": 0.39},
        "MM - LCA":             {"mean": 94.8, "std": 0.34},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 94.27, "std": 0.31},
        "MOS - LCA":            {}, # running - wan
        "SLCA - CIL":           {}, # running - wan
        "SLCA - CA":            {"mean": 93.70, "std": 0.33},
        "SLCA - LCA":           {"mean": 92.00, "std": 0.71}, # running - wan
        "EASE":                 {"mean": 91.69, "std": 0.33},
        "APER + Adapter":       {"mean": 90.78, "std": 0.45},
        "APER + Finetune":      {"mean": 81.65, "std": 0.93},
        "APER + SSF":           {"mean": 89.45, "std": 0.97},
        "APER + VPT-Deep":      {"mean": 88.96, "std": 0.91},
        "APER + VPT-Shallow":   {"mean": 88.10, "std": 0.93},
        "L2P":                  {"mean": 87.75, "std": 1.41},
        "CODA-Prompt":          {"mean": 91.01, "std": 0.21},
        "DualPrompt":           {"mean": 86.73, "std": 0.62},
    },
    "IN-R": {
        "MM - CIL":             {"mean": 85.06},
        "MM - CA":              {"mean": 85.20, "std": 0.22},
        "MM - LCA":             {"mean": 85.83, "std": 0.25},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 83.30, "std": 0.61}, # running - wan
        "MOS - LCA":            {}, # running - wan
        "SLCA - CIL":           {}, # running - wan
        "SLCA - CA":            {"mean": 81.17, "std": 0.28},
        "SLCA - LCA":           {}, # running - wan
        "EASE":                 {"mean": 82.41, "std": 0.46},
        "APER + Adapter":       {"mean": 78.80, "std": 0.58},
        "APER + Finetune":      {"mean": 72.06, "std": 0.80},
        "APER + SSF":           {"mean": 78.07, "std": 1.09},
        "APER + VPT-Deep":      {"mean": 78.77, "std": 0.73},
        "APER + VPT-Shallow":   {"mean": 68.83, "std": 0.00},
        "L2P":                  {"mean": 77.27, "std": 0.61},
        "CODA-Prompt":          {"mean": 78.22, "std": 0.36},
        "DualPrompt":           {"mean": 74.61, "std": 0.48},
    },
    "IN-A": {
        "MM - CIL":             {"mean": 66.52},
        "MM - CA":              {"mean": 72.25, "std": 1.09},
        "MM - LCA":             {"mean": 75.21},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 67.68, "std": 1.93},
        "MOS - LCA":            {"mean": 70.71, "std": 1.51},
        "SLCA - CIL":           {}, # running - wan
        "SLCA - CA":            {"mean": 45.15, "std": 19.78},
        "SLCA - LCA":           {"mean": 67.35, "std": 0.54},
        "EASE":                 {"mean": 67.77, "std": 1.84},
        "APER + Adapter":       {"mean": 58.86, "std": 1.32},
        "APER + Finetune":      {"mean": 63.17, "std": 0.5},
        "APER + SSF":           {"mean": 61.58, "std": 0.45},
        "APER + VPT-Deep":      {"mean": 56.96, "std": 0.37},
        "APER + VPT-Shallow":   {"mean": 56.91, "std": 1.39},
        "L2P":                  {"mean": 52.61, "std": 1.66},
        "CODA-Prompt":          {"mean": 48.13, "std": 0.85},
        "DualPrompt":           {"mean": 55.33, "std": 1.46},
    },
    "CUB": {
        "MM - CIL":             {"mean": 87.16},
        "MM - CA":              {"mean": 89.79, "std": 0.28},
        "MM - LCA":             {"mean": 90.8, "std": 0.31},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 92.31, "std": 0.68},
        "MOS - LCA":            {}, # running - wan
        "SLCA - CIL":           {}, # running - wan
        "SCLA - CA":            {"mean": 90.21, "std": 0.90},
        "SLCA - LCA":           {}, # running - wan
        "EASE":                 {"mean": 89.49, "std": 1.23},
        "APER + Adapter":       {"mean": 89.74, "std": 1.29},
        "APER + Finetune":      {"mean": 89.54, "std": 1.39},
        "APER + SSF":           {"mean": 89.61, "std": 1.15},
        "APER + VPT-Deep":      {"mean": 88.96, "std": 1.16},
        "APER + VPT-Shallow":   {"mean": 89.52, "std": 1.44},
        "L2P":                  {"mean": 75.82, "std": 1.77},
        "CODA-Prompt":          {"mean": 75.65, "std": 1.24},
        "DualPrompt":           {"mean": 78.86, "std": 0.97},
    },
    "OB": {
        "MM - CIL":             {"mean": 80.20},
        "MM - CA":              {"mean": 81.24, "std": 0.56},
        "MM - LCA":             {"mean": 81.53, "std": 0.46},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 86.10, "std": 0.72},
        "MOS - LCA":            {}, # running - wan
        "SLCA - CIL":           {}, # running - wan
        "SLCA - CA":            {"mean": 82.67, "std": 0.61},
        "SLCA - LCA":           {}, # running - wan
        "EASE":                 {"mean": 80.78, "std": 0.15},
        "APER + Adapter":       {"mean": 80.28, "std": 0.44},
        "APER + Finetune":      {"mean": 77.83, "std": 1.19},
        "APER + SSF":           {"mean": 80.31, "std": 0.56},
        "APER + VPT-Deep":      {"mean": 79.76, "std": 0.38},
        "APER + VPT-Shallow":   {"mean": 79.65, "std": 0.95},
        "L2P":                  {"mean": 73.83, "std": 1.23},
        "CODA-Prompt":          {"mean": 70.97, "std": 0.10},
        "DualPrompt":           {"mean": 74.41, "std": 1.17}
    },
    "VTAB": {
        "MM - CIL":             {"mean": 86.53},
        "MM - CA":              {"mean": 94.27, "std": 1.19},
        "MM - LCA":             {"mean": 94.32, "std": 1.10},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 92.56, "std": 0.61},
        "MOS - LCA":            {}, # running - wan
        "SLCA - CIL":           {}, # running - wan
        "SLCA - CA":            {"mean": 91.08, "std": 3.38},
        "SLCA - LCA":           {}, # running - wan
        "EASE":                 {"mean": 93.28, "std": 0.12},

        "APER + Adapter":       {"mean": 90.66, "std": 0.57},
        "APER + Finetune":      {"mean": 91.82, "std": 1.44},
        "APER + SSF":           {"mean": 91.84, "std": 1.51},
        "APER + VPT-Deep":      {"mean": 91.91, "std": 1.41},
        "APER + VPT-Shallow":   {"mean": 91.55, "std": 0.76},
        "L2P":                  {"mean": 82.37, "std": 2.85},
        "CODA-Prompt":          {"mean": 83.90, "std": 5.00},
        "DualPrompt":           {"mean": 83.99, "std": 5.89}
    },
    "CARS": {
        "MM - CIL":             {"mean": 69.55},
        "MM - CA":              {"mean": 75.71, "std": 1.39},
        "MM - LCA":             {"mean": 76.16, "std": 1.41},
        "MOS - CIL":            {}, # running - wan
        "MOS - CA":             {"mean": 71.43, "std": 19.54},
        "MOS - LCA":            {}, # running - wan
        "SLCA - CIL":           {}, # running - wan
        "SLCA - CA":            {"mean": 74.58, "std": 2.16},
        "SLCA - LCA":           {}, # running - wan
        "EASE":                 {"mean": 48.08, "std": 1.19},
        "APER + Adapter":       {"mean": 50.60, "std": 1.08},
        "APER + Finetune":      {"mean": 53.22, "std": 1.39},
        "APER + SSF":           {"mean": 51.30, "std": 1.09},
        "APER + VPT-Deep":      {"mean": 50.64, "std": 3.01},
        "APER + VPT-Shallow":   {"mean": 50.87, "std": 0.80},
        "L2P":                  {"mean": 53.41, "std": 1.16},
        "CODA-Prompt":          {"mean": 26.29, "std": 0.56},
        "DualPrompt":           {"mean": 49.40, "std": 2.10}
    }
}

seed1993_results = {
    "CIFAR100": {
        "MM - CIL":             [98.8, 97.82, 97.03, 96.41, 95.67, 95.01, 94.51, 93.83, 93.25, 92.75],
        "MM - CA":              [98.8, 98.1, 97.71, 97.23, 96.74, 96.28, 95.81, 95.27, 94.8, 94.4],
        "MM - LCA":             [98.8, 98.07, 97.81, 97.42, 97.0, 96.55, 96.15, 95.63, 95.19, 94.8],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [99.0, 98.2, 97.73, 97.28, 96.72, 96.25, 95.78, 95.29, 94.77, 94.27],
        "SLCA - LCA":           [],
        "EASE":                 [98.7, 97.62, 96.67, 95.96, 95.1, 94.42, 93.87, 93.18, 92.58, 92.07],
        "APER + Adapter":       [98.2, 97.0, 96.09, 95.16, 94.31, 93.5, 92.85, 92.12, 91.47, 90.9],
        "APER + Finetune":      [93.7, 90.95, 89.52, 88.15, 87.01, 85.93, 85.13, 84.18, 83.37, 82.64],
        "APER + SSF":           [98.8, 97.02, 95.73, 94.51, 93.41, 92.46, 91.73, 90.92, 90.2, 89.57],
        "APER + VPT-Deep":      [98.8, 96.82, 95.48, 94.31, 93.23, 92.23, 91.49, 90.66, 89.93, 89.27],
        "APER + VPT-Shallow":   [96.4, 94.82, 93.84, 92.85, 91.89, 91.01, 90.35, 89.55, 88.85, 88.24],
        "L2P":                  [97.4, 96.15, 95.23, 93.98, 92.86, 91.96, 91.21, 90.46, 89.82, 89.28],
        "CODA-Prompt":          [99.0, 97.1, 95.94, 94.93, 94.07, 93.13, 92.38, 91.8, 91.34, 90.88],
        "DualPrompt":           [96.4, 94.95, 93.76, 92.59, 91.35, 90.32, 89.56, 88.69, 87.92, 87.35],
    },
    "IN-R": {
        "MM - CIL":             [94.63, 92.47, 91.04, 89.76, 88.48, 87.55, 86.82, 86.18, 85.61, 85.06],
        "MM - CA":              [94.63, 92.46, 91.2, 90.22, 88.88, 87.9, 87.16, 86.58, 86.03, 85.47],
        "MM - LCA":             [94.63, 92.53, 91.32, 90.24, 89.0, 88.03, 87.3, 86.72, 86.16, 85.63],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [93.9, 91.99, 90.59, 89.27, 88.17, 87.27, 86.63, 86.04, 85.55, 85.04],
        "SLCA - LCA":           [],
        "EASE":                 [93.32, 90.72, 89.09, 87.7, 86.49, 85.52, 84.7, 83.99, 83.35, 82.75],
        "APER + Adapter":       [92.02, 88.48, 86.46, 84.94, 83.51, 82.32, 81.34, 80.48, 79.7, 79.01],
        "APER + Finetune":      [94.05, 88.09, 84.52, 81.82, 79.62, 77.78, 76.29, 74.98, 73.87, 72.85],
        "APER + SSF":           [93.47, 89.58, 87.34, 85.56, 83.89, 82.56, 81.47, 80.54, 79.72, 78.97],
        "APER + VPT-Deep":      [91.44, 88.6, 86.69, 85.12, 83.72, 82.54, 81.58, 80.77, 80.02, 79.3],
        "APER + VPT-Shallow":   [],
        "L2P":                  [88.39, 85.6, 83.83, 82.56, 81.48, 80.56, 79.85, 79.18, 78.58, 77.96],
        "CODA-Prompt":          [86.01, 84.95, 83.92, 82.81, 81.83, 80.88, 79.95, 79.11, 78.45, 77.91],
        "DualPrompt":           [85.49, 83.1, 81.5, 80.13, 78.77, 77.77, 76.98, 76.31, 75.7, 75.04],
    },
    "IN-A": {
        "MM - CIL":             [87.43, 84.56, 81.13, 78.0, 75.16, 72.74, 70.74, 69.16, 67.73, 66.52],
        "MM - CA":              [87.43, 86.06, 84.14, 81.79, 80.06, 78.4, 77.07, 75.89, 74.79, 73.72],
        "MM - LCA":             [87.43, 86.06, 84.38, 82.58, 80.78, 79.2, 77.97, 76.89, 75.7, 74.66],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [83.43, 80.74, 79.25, 77.85, 62.3, 51.94, 44.53, 38.97, 34.65, 31.19],
        "SLCA - LCA":           [83.43, 81.02, 78.66, 77.09, 75.44, 73.65, 71.74, 70.21, 68.97, 67.97],
        "EASE":                 [85.71, 82.85, 80.31, 78.35, 76.5, 75.05, 73.5, 72.05, 70.64, 69.44],
        "APER + Adapter":       [76.57, 73.7, 70.84, 68.57, 66.51, 64.93, 63.39, 61.96, 60.61, 59.48],
        "APER + Finetune":      [],
        "APER + SSF":           [83.43, 78.1, 74.06, 71.27, 69.19, 67.47, 65.95, 64.53, 63.2, 62.09],
        "APER + VPT-Deep":      [73.14, 71.02, 68.21, 66.27, 64.27, 62.74, 61.17, 59.61, 58.16, 56.93],
        "APER + VPT-Shallow":   [74.29, 71.45, 68.64, 66.24, 64.17, 62.67, 61.22, 59.85, 58.58, 57.51],
        "L2P":                  [73.14, 66.57, 63.99, 61.32, 59.14, 57.5, 56.11, 54.83, 53.69, 52.67],
        "CODA-Prompt":          [72.31, 66.64, 62.01, 58.68, 56.2, 54.05, 52.18, 50.51, 48.97, 47.57],
        "DualPrompt":           [73.14, 71.16, 68.31, 65.84, 63.64, 61.9, 60.39, 58.97, 57.7, 56.56],
    },
    "CUB": {
        "MM - CIL":             [98.38, 97.29, 95.63, 94.21, 92.71, 90.96, 89.69, 88.76, 87.93, 87.16],
        "MM - CA":              [98.38, 97.72, 96.47, 95.35, 94.39, 93.37, 92.57, 91.72, 90.94, 90.18],
        "MM - LCA":             [98.38, 97.94, 96.5, 95.35, 94.43, 93.51, 92.72, 91.93, 91.17, 90.46],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [99.19, 98.5, 97.16, 95.75, 94.83, 93.88, 93.03, 92.43, 91.78, 91.23],
        "SLCA - LCA":           [],
        "EASE":                 [97.98, 97.12, 95.9, 94.94, 94.2, 93.3, 92.41, 91.67, 91.1, 90.52],
        "APER + Adapter":       [97.98, 97.12, 95.9, 94.94, 94.22, 93.36, 92.55, 91.88, 91.3, 90.74],
        "APER + Finetune":      [98.79, 97.86, 96.44, 95.4, 94.55, 93.57, 92.7, 92.01, 91.38, 90.8],
        "APER + SSF":           [97.57, 96.8, 95.54, 94.53, 93.79, 93.0, 92.23, 91.62, 91.08, 90.54],
        "APER + VPT-Deep":      [96.36, 95.65, 94.17, 93.34, 92.7, 91.87, 91.06, 90.41, 89.86, 89.28],
        "APER + VPT-Shallow":   [97.98, 97.23, 95.92, 94.93, 94.16, 93.27, 92.43, 91.72, 91.12, 90.56],
        "L2P":                  [98.79, 92.46, 89.46, 87.36, 84.96, 82.69, 80.98, 79.57, 78.41, 77.25],
        "CODA-Prompt":          [94.81, 90.05, 85.18, 82.75, 81.59, 80.45, 79.16, 78.18, 77.26, 76.47],
        "DualPrompt":           [98.79, 94.11, 91.56, 88.99, 86.97, 85.16, 83.67, 82.25, 81.03, 79.85],
    },
    "OB": {
        "MM - CIL":             [94.5, 91.92, 89.81, 87.85, 86.21, 84.84, 83.59, 82.39, 81.24, 80.2],
        "MM - CA":              [94.5, 93.25, 91.57, 89.34, 87.5, 85.89, 84.52, 83.13, 81.78, 80.5],
        "MM - LCA":             [94.5, 93.21, 91.62, 89.51, 87.71, 86.1, 84.77, 83.42, 82.15, 80.92],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [93.0, 91.5, 90.18, 88.57, 87.23, 85.98, 84.9, 83.88, 82.97, 82.11],
        "SLCA - LCA":           [],
        "EASE":                 [90.17, 89.46, 88.34, 86.92, 85.6, 84.31, 83.26, 82.24, 81.34, 80.61],
        "APER + Adapter":       [88.83, 88.45, 87.52, 86.12, 84.81, 83.51, 82.42, 81.39, 80.5, 79.78],
        "APER + Finetune":      [87.0, 86.12, 84.8, 83.28, 81.94, 80.53, 79.32, 78.23, 77.29, 76.55],
        "APER + SSF":           [89.83, 89.04, 87.85, 86.39, 85.08, 83.73, 82.63, 81.61, 80.71, 79.98],
        "APER + VPT-Deep":      [88.33, 87.74, 86.77, 85.43, 84.21, 82.94, 81.92, 80.93, 80.05, 79.33],
        "APER + VPT-Shallow":   [87.0, 86.78, 85.96, 84.7, 83.47, 82.24, 81.19, 80.2, 79.35, 78.65],
        "L2P":                  [89.67, 85.87, 83.44, 81.2, 79.41, 77.65, 76.26, 74.9, 73.66, 72.59],
        "CODA-Prompt":          [90.47, 87.42, 78.42, 74.3, 72.92, 72.48, 72.26, 71.78, 71.37, 71.03],
        "DualPrompt":           [88.67, 85.32, 83.36, 81.56, 80.02, 78.49, 77.03, 75.56, 74.27, 73.22],
    },
    "VTAB": {
        "MM - CIL":             [99.52, 98.38, 94.64, 89.9, 86.53],
        "MM - CA":              [99.52, 98.64, 97.06, 95.92, 94.57],
        "MM - LCA":             [99.52, 98.6, 97.04, 95.91, 94.65],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [98.91, 98.24, 96.14, 94.51, 92.47],
        "SLCA - LCA":           [],
        "EASE":                 [95.7, 93.88, 93.66, 93.41, 93.41],
        "APER + Adapter":       [99.39, 97.02, 94.32, 92.75, 90.92],
        "APER + Finetune":      [99.03, 96.83, 94.27, 92.75, 90.96],
        "APER + SSF":           [99.39, 97.22, 94.66, 93.15, 91.32],
        "APER + VPT-Deep":      [99.52, 97.3, 94.77, 93.32, 91.59],
        "APER + VPT-Shallow":   [99.64, 97.34, 94.58, 92.99, 91.28],
        "L2P":                  [98.67, 96.82, 92.39, 88.67, 84.4],
        "CODA-Prompt":          [],
        "DualPrompt":           [98.31, 95.89, 92.66, 89.61, 86.24],
    },
    "CARS": {
        "MM - CIL":             [90.34, 85.74, 82.5, 80.07, 77.69, 75.45, 73.69, 72.09, 70.72, 69.55],
        "MM - CA":              [90.34, 88.36, 86.87, 85.45, 83.88, 82.24, 80.81, 79.25, 77.9, 76.73],
        "MM - LCA":             [90.34, 88.5, 87.18, 85.93, 84.46, 82.93, 81.43, 79.81, 78.39, 77.18],
        "MOS - CIL":            [],
        "MOS - CA":             [],
        "MOS - LCA":            [],
        "SLCA - CIL":           [],
        "SLCA - CA":            [89.0, 85.3, 80.63, 78.55, 77.51, 76.17, 75.06, 74.07, 73.12, 72.23],
        "SLCA - LCA":           [],
        "EASE":                 [79.05, 68.66, 64.61, 61.51, 58.71, 55.94, 53.77, 51.84, 50.13, 48.48],
        "APER + Adapter":       [76.52, 70.46, 66.79, 63.68, 61.09, 58.82, 56.77, 54.92, 53.24, 51.7],
        "APER + Finetune":      [78.9, 73.0, 69.55, 66.47, 63.93, 61.69, 59.7, 57.83, 56.16, 54.64],
        "APER + SSF":           [76.23, 70.4, 67.06, 64.0, 61.32, 59.03, 57.0, 55.12, 53.45, 51.93],
        "APER + VPT-Deep":      [73.55, 66.61, 62.87, 59.59, 56.82, 54.36, 52.23, 50.32, 48.67, 47.17],
        "APER + VPT-Shallow":   [76.37, 70.31, 66.66, 63.61, 61.02, 58.72, 56.66, 54.82, 53.15, 51.63],
        "L2P":                  [79.35, 72.04, 68.49, 65.34, 62.53, 60.48, 58.77, 57.26, 55.96, 54.75],
        "CODA-Prompt":          [57.89, 46.5, 42.28, 36.69, 33.25, 30.86, 29.05, 27.69, 26.65, 25.77],
        "DualPrompt":           [83.21, 76.16, 71.66, 67.48, 63.86, 60.66, 57.92, 55.62, 53.66, 51.82],
    }
}

def create_ina_bar_chart():
    dataset_name = "IN-A"
    methods_to_plot = ["MM - CIL", "MM - CA", "MM - LCA", "MOS - CIL", "MOS - CA", "MOS - LCA", 
                       "SLCA - CIL", "SLCA - CA", "SLCA - LCA"]
    
    # Define color scheme
    color_map = {
        "MM": "#1f77b4",    # Blue
        "MOS": "#2ca02c",   # Green  
        "SLCA": "#d62728"   # Red
    }
    
    # Organize data by groups
    method_groups = {}
    for method in methods_to_plot:
        if method in results[dataset_name] and results[dataset_name][method]:
            group = method.split(' - ')[0]  # Get MM, MOS, or SLCA
            method_type = method.split(' - ')[1]  # Get CIL, CA, or LCA
            
            if group not in method_groups:
                method_groups[group] = {}
            method_groups[group][method_type] = results[dataset_name][method]["mean"]
    
    # Prepare data for grouped bar chart
    groups = list(method_groups.keys())
    approach_types = ['CIL', 'CA', 'LCA']
    
    # Set up positions for bars
    x = np.arange(len(groups))
    width = 0.25  # Width of bars
    
    plt.figure(figsize=(12, 8))
    
    # Create bars for each approach type
    bars = {}
    for i, approach in enumerate(approach_types):
        values = []
        colors = []
        for group in groups:
            if approach in method_groups[group]:
                values.append(method_groups[group][approach])
                # Adjust color intensity based on approach (CIL lighter, LCA darker)
                base_color = color_map[group]
                if approach == 'CIL':
                    colors.append(base_color + '80')  # More transparent
                elif approach == 'CA':
                    colors.append(base_color + 'B0')  # Medium transparency
                else:  # LCA
                    colors.append(base_color + 'FF')  # Full opacity
            else:
                values.append(0)
                colors.append('#00000000')  # Transparent for missing data
        
        bars[approach] = plt.bar(x + i*width - width, values, width, 
                               label=approach, color=colors, 
                               edgecolor='black', linewidth=0.5)
    
    # Customize the chart
    plt.xlabel('Method Groups', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'Performance Comparison on {dataset_name} Dataset (Grouped)', fontsize=14, fontweight='bold')
    plt.xticks(x, groups, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for approach in approach_types:
        for i, (bar, group) in enumerate(zip(bars[approach], groups)):
            if approach in method_groups[group]:
                value = method_groups[group][approach]
                if value > 0:  # Only label non-zero values
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement arrows within each group
    for i, group in enumerate(groups):
        group_methods = method_groups[group]
        base_x = x[i]
        
        # CIL -> CA arrow
        if 'CIL' in group_methods and 'CA' in group_methods:
            cil_val = group_methods['CIL']
            ca_val = group_methods['CA']
            improvement = ca_val - cil_val
            
            # Position arrow between CIL and CA bars
            start_x = base_x - width
            end_x = base_x
            arrow_y = max(cil_val, ca_val) + 3
            
            plt.annotate('', xy=(end_x, ca_val), xytext=(start_x, cil_val),
                        arrowprops=dict(arrowstyle='->', color=color_map[group], lw=1.5, alpha=0.8))
            plt.text((start_x + end_x)/2, arrow_y, f'+{improvement:.1f}%', 
                    ha='center', va='bottom', fontsize=8, color=color_map[group], fontweight='bold')
        
        # CA -> LCA arrow
        if 'CA' in group_methods and 'LCA' in group_methods:
            ca_val = group_methods['CA']
            lca_val = group_methods['LCA']
            improvement = lca_val - ca_val
            
            # Position arrow between CA and LCA bars
            start_x = base_x
            end_x = base_x + width
            arrow_y = max(ca_val, lca_val) + 3
            
            plt.annotate('', xy=(end_x, lca_val), xytext=(start_x, ca_val),
                        arrowprops=dict(arrowstyle='->', color=color_map[group], lw=1.5, alpha=0.8))
            plt.text((start_x + end_x)/2, arrow_y, f'+{improvement:.1f}%', 
                    ha='center', va='bottom', fontsize=8, color=color_map[group], fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Add approach type legend
    legend_elements.append(Patch(facecolor='gray', alpha=0.5, label='CIL'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.7, label='CA'))
    legend_elements.append(Patch(facecolor='gray', alpha=1.0, label='LCA'))
    
    # Add separator
    legend_elements.append(Patch(facecolor='white', edgecolor='white', label=''))
    
    # Add method group legend
    legend_elements.append(Patch(facecolor=color_map["MM"], label='MM (Our Method)'))
    legend_elements.append(Patch(facecolor=color_map["MOS"], label='MOS'))
    legend_elements.append(Patch(facecolor=color_map["SLCA"], label='SLCA'))
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.ylim(0, max([max(group.values()) for group in method_groups.values()]) * 1.25)
    
    # Show the plot
    plt.show()
    
    # Print the data for reference with improvements
    print(f"\n{dataset_name} Dataset Results (Grouped Comparison):")
    print("-" * 60)
    for group, methods in method_groups.items():
        print(f"\n{group} Group:")
        for method_type in ['CIL', 'CA', 'LCA']:
            if method_type in methods:
                value = methods[method_type]
                print(f"  {method_type:<4}: {value:>6.2f}%")
        
        # Calculate improvements
        if 'CIL' in methods and 'CA' in methods:
            improvement = methods['CA'] - methods['CIL']
            print(f"  CA improvement over CIL: +{improvement:.2f}%")
        if 'CA' in methods and 'LCA' in methods:
            improvement = methods['LCA'] - methods['CA']
            print(f"  LCA improvement over CA: +{improvement:.2f}%")
        if 'CIL' in methods and 'LCA' in methods:
            improvement = methods['LCA'] - methods['CIL']
            print(f"  LCA improvement over CIL: +{improvement:.2f}%")

# Call the function to create and display the chart
if __name__ == "__main__":
    create_ina_bar_chart()



