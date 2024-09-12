import pandas as pd
import numpy as np
import re

data_path = "C:\\Users\\azeem\\OneDrive\\Azeem Documents\\Desktop\\MyProjects\\Amazon ML Challenge\\student_resource 3\\dataset\\"
train_df = pd.read_csv(data_path + "train.csv")
test_df = pd.read_csv(data_path + "test.csv")

train_df.dropna(subset=['image_link', 'entity_name', 'entity_value'])

train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()

def convert_item_weight(value):
    if '[' in value and ']' in value:
        values = re.findall(r"(\d+(\.\d+)?)", value)
        if len(values) > 1:
            value = values[1][0]
        else:
            return "Invalid data"
    match = re.match(r"(\d+(\.\d+)?)\s?(gram|kilogram|milligram|microgram|ounce|pound|ton)", value.lower())
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if unit == "kilogram":
            value *= 1000
        elif unit == "milligram":
            value /= 1000  
        elif unit == "microgram":
            value /= 1000000  
        elif unit == "ounce":
            value *= 28.3495  
        elif unit == "pound":
            value *= 453.592  
        elif unit == "ton":
            value *= 1000000
        return f"{value:.3f} gram"
    return value

def convert_dimensions(value):
    if '[' in value and ']' in value:
        values = re.findall(r"(\d+(\.\d+)?)", value)
        if len(values) > 1:
            value = values[1][0]
        else:
            return "Invalid data"
    match = re.match(r"(\d+(\.\d+)?)\s?(metre|centimetre|foot|inch|millimetre|yard)", value.lower())
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if unit == "centimetre":
            value /= 100  
        elif unit == "millimetre":
            value /= 1000  
        elif unit == "inch":
            value *= 0.0254  
        elif unit == "foot":
            value *= 0.3048  
        elif unit == "yard":
            value *= 0.9144  
        return f"{value:.3f} metre"
    return value 

def convert_voltage(value):
    if '[' in value and ']' in value:
        values = re.findall(r"(\d+(\.\d+)?)", value)
        if len(values) > 1:
            value = values[1][0]
        else:
            return "Invalid data"
    match = re.match(r"(\d+(\.\d+)?)\s?(kilovolt|millivolt|volt)", value.lower())
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if unit == "millivolt":
            value /= 1000  
        elif unit == "kilovolt":
            value *= 1000 
        return f"{value:.3f} volt"
    return value

def convert_wattage(value):
    if '[' in value and ']' in value:
        values = re.findall(r"(\d+(\.\d+)?)", value)
        if len(values) > 1:
            value = values[1][0]
        else:
            return "Invalid data"
    match = re.match(r"(\d+(\.\d+)?)\s?(watt|kilowatt)", value.lower())
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        if unit == "kilowatt":
            value *= 1000  
        return f"{value:.3f} watt"
    return value 

def convert_item_volume(value):
    if '[' in value and ']' in value:
        values = re.findall(r"(\d+(\.\d+)?)", value)
        if len(values) > 1:
            value = values[1][0] 
        else:
            return "Invalid data"
    match = re.match(r"(\d+(\.\d+)?)\s?(litre|millilitre|centilitre|microlitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|pint|quart)", value.lower())
    if match:
        value = float(match.group(1))  
        unit = match.group(3)  
        if unit == "millilitre":
            value /= 1000  
        elif unit == "centilitre":
            value /= 100  
        elif unit == "microlitre":
            value /= 1000000  
        elif unit == "cubic foot":
            value *= 28.3168  
        elif unit == "cubic inch":
            value *= 0.0163871  
        elif unit == "cup":
            value *= 0.24  
        elif unit == "decilitre":
            value /= 10  
        elif unit == "fluid ounce":
            value *= 0.0295735  
        elif unit == "gallon":
            value *= 3.78541  
        elif unit == "imperial gallon":
            value *= 4.54609  
        elif unit == "pint":
            value *= 0.473176  
        elif unit == "quart":
            value *= 0.946353  
        return f"{value:.3f} litre"
    return value

def convert_entity(row):
    entity_name = row['entity_name']
    entity_value = row['entity_value']
    if entity_name == 'item_weight':
        return convert_item_weight(entity_value)
    elif entity_name in ['width', 'depth', 'height']:
        return convert_dimensions(entity_value)
    elif entity_name == 'voltage':
        return convert_voltage(entity_value)
    elif entity_name == 'wattage':
        return convert_wattage(entity_value)
    elif entity_name == 'item_volume':
        return convert_item_volume(entity_value)
    return entity_value

train_df['entity_value'] = train_df.apply(convert_entity, axis = 1)


train_df.to_csv(data_path + "processed_train.csv", index=False)