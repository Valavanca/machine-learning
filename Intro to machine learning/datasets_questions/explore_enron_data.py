#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "--",len(enron_data["SKILLING JEFFREY K"])

print "--[1]", enron_data["PRENTICE JAMES"]["total_stock_value"]

print "--[2]", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "--[3]", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]


poi_name_record = open("../final_project/poi_names.txt").read().split("\n")
poi_name_total = [record for record in poi_name_record if "(y)" in record or "(n)" in record]
print("Total number of POIs: ", len(poi_name_total))


