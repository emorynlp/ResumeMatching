import pandas as pd
import os
import shutil
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime

train_keys = []
with open('../Resume_Classification/finetuning_data/resume_train.json') as json_file:
    train = json.load(json_file)
    for item in train:
        key = list(item.keys())[0]
        train_keys.append(key)

print(train_keys)
print(len(train_keys))

dev_keys = []
with open('../Resume_Classification/finetuning_data/resume_dev.json') as json_file:
    dev = json.load(json_file)
    for item in dev:
        key = list(item.keys())[0]
        dev_keys.append(key)

print(dev_keys)
print(len(dev_keys))

test_keys = []
with open('../Resume_Classification/finetuning_data/resume_test.json') as json_file:
    test = json.load(json_file)
    for item in test:
        key = list(item.keys())[0]
        test_keys.append(key)

print(test_keys)
print(len(test_keys))

label_dic = {}
data_path = ["../Resume_Classification/finetuning_data/resume_train.json","../Resume_Classification/finetuning_data/resume_dev.json", "../Resume_Classification/finetuning_data/resume_test.json"]
for path in data_path:
    with open(path) as f:
        data = json.load(f)
        for item in data:
            name = list(item.keys())[0]
            label = item["label"]
            label_dic[name] = label


path ="E:/Emory/Research/ResumeClassification/rchilli-parsed"
filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root,file))

print("total_number_of_files", len(filelist))

# filelist =["E:/Emory/Research/ResumeClassification/rchilli-parsed/48051.json", "E:/Emory/Research/ResumeClassification/rchilli-parsed/49504.json"]

train_results = []
dev_results = []
test_results = []

train_edges = []
dev_edges = []
test_edges = []

for file in tqdm(filelist):
    with open(file, 'rb') as f:
        total_dic = {}

        total_edges = {}
        edges = []
        school_count = 0
        job_count = 0

        content_dic = {}
        contents = []
        items = []
        sections = []
        name = str(os.path.splitext(os.path.basename(file))[0])
        data = json.load(f)

        if "ResumeParserData" in list(data.keys()):
            data = data["ResumeParserData"]
        else:
            print("error!", file)
            exit(0)
        for key, value in data.items():

            if key=="SegregatedQualification":
                if isinstance(data[key], list):

                    # edges.append(["education", "education"])

                    for item in data[key]:
                        school_count += 1
                        edges.append(["education", "institute"+str(school_count)])

                        # edges.append(["institute" + str(school_count), "institute" + str(school_count)])

                        # print(edges)
                        if item["Institution"]["Name"] != "":
                            contents.append(item["Institution"]["Name"])
                            items.append("institution name")
                            sections.append("education")
                            edges.append(["institute"+str(school_count), "name"+str(school_count)])
                            edges.append(["name"+str(school_count), item["Institution"]["Name"]])

                            # edges.append(["name"+str(school_count), "name"+str(school_count)])
                            # edges.append([item["Institution"]["Name"], item["Institution"]["Name"]])

                        # if item["Degree"]["DegreeName"] != "":
                        #     contents.append(item["Degree"]["DegreeName"])
                        #     items.append("degree name")
                        #     sections.append("education")
                        if item["Degree"]["NormalizeDegree"] != "":
                            contents.append(item["Degree"]["NormalizeDegree"])
                            items.append("degree")
                            sections.append("education")
                            edges.append(["institute"+str(school_count), "degree"+str(school_count)])
                            edges.append(["degree"+str(school_count), item["Degree"]["NormalizeDegree"]])

                            # edges.append(["degree"+str(school_count), "degree"+str(school_count)])
                            # edges.append([item["Degree"]["NormalizeDegree"], item["Degree"]["NormalizeDegree"]])

                        if item["Degree"]["Specialization"]:
                            specialization = item["Degree"]["Specialization"][0]
                            # for spec in specialization:
                            #     if spec != "":
                            contents.append(specialization)
                            items.append("specialization")
                            sections.append("education")
                            edges.append(["institute"+str(school_count), "specialization"+str(school_count)])
                            edges.append(["specialization"+str(school_count), specialization])

                            # edges.append(["specialization"+str(school_count), "specialization"+str(school_count)])
                            # edges.append([specialization, specialization])

                        if item["StartDate"] != "":
                            start = datetime.strptime(item["StartDate"], "%d/%m/%Y")
                            if item["EndDate"] != "":
                                end = datetime.strptime(item["EndDate"], "%d/%m/%Y")
                            else:
                                end = datetime.strptime("01/06/2020", "%d/%m/%Y")
                            num_year = str(end.year - start.year)
                            contents.append(num_year)
                            items.append("education duration")
                            sections.append("education")
                            edges.append(["institute"+str(school_count), "duration"+str(school_count)])
                            edges.append(["duration"+str(school_count), num_year])

                            # edges.append(["duration"+str(school_count), "duration"+str(school_count)])
                            # edges.append([num_year, num_year])

            elif key=="SegregatedExperience":
                if isinstance(data[key], list):

                    # edges.append(["work", "work"])

                    for item in data[key]:
                        job_count += 1
                        edges.append(["work", "job"+str(job_count)])

                        # edges.append(["job"+str(job_count), "job"+str(job_count)])

                        # print(edges)
                        if item["Employer"]["EmployerName"] != "":
                            contents.append(item["Employer"]["EmployerName"])
                            items.append("employer name")
                            sections.append("work experience")
                            edges.append(["job"+str(job_count), "employer"+str(job_count)])
                            edges.append(["employer"+str(job_count), item["Employer"]["EmployerName"]])

                            # edges.append(["employer"+str(job_count), "employer"+str(job_count)])
                            # edges.append([item["Employer"]["EmployerName"], item["Employer"]["EmployerName"]])

                        if item["JobProfile"]["Title"] != "":
                            contents.append(item["JobProfile"]["Title"])
                            items.append("job title")
                            sections.append("work experience")
                            edges.append(["job"+str(job_count), "title"+str(job_count)])
                            edges.append(["title"+str(job_count), item["JobProfile"]["Title"]])

                            # edges.append(["title"+str(job_count), "title"+str(job_count)])
                            # edges.append([item["JobProfile"]["Title"], item["JobProfile"]["Title"]])

                        if item["StartDate"] != "":
                            start = datetime.strptime(item["StartDate"], "%d/%m/%Y")
                            if item["EndDate"] != "":
                                end = datetime.strptime(item["EndDate"], "%d/%m/%Y")
                            else:
                                end = datetime.strptime("01/06/2020", "%d/%m/%Y")
                            num_year = str(end.year - start.year)
                            contents.append(num_year)
                            items.append("job duration")
                            sections.append("work experience")
                            edges.append(["job"+str(job_count), "period"+str(job_count)])
                            edges.append(["period"+str(job_count), num_year])

                            # edges.append(["period"+str(job_count), "period"+str(job_count)])
                            # edges.append([num_year, num_year])

                        if item["JobDescription"] != "":
                            text = item["JobDescription"]
                            text = text.replace("\r", " ")
                            text = text.replace("\t", " ")
                            text = text.replace("\n", " ")
                            contents.append(text)
                            items.append("job description")
                            sections.append("work experience")
                            edges.append(["job"+str(job_count), "description"+str(job_count)])
                            edges.append(["description"+str(job_count), text])

                            # edges.append(["description"+str(job_count), "description"+str(job_count)])
                            # edges.append([text, text])

            elif key=="Publication":
                if data[key] != "":
                    text = data[key]
                    text = text.replace("\r", " ")
                    text = text.replace("\t", " ")
                    text = text.replace("\n", " ")
                    contents.append(text)
                    items.append("publication")
                    sections.append("other")

                    # edges.append(["other", "publication"])
                    edges.append(["publication", text])

                    # if ["other", "other"] not in edges:
                    #     edges.append(["other", "other"])
                    # edges.append(["publication", "publication"])
                    # edges.append([text, text])

                else:
                    continue

            elif key=="Hobbies":
                if data[key] != "":
                    text = data[key]
                    text = text.replace("\r", " ")
                    text = text.replace("\t", " ")
                    text = text.replace("\n", " ")
                    contents.append(text)
                    items.append("hobbies")
                    sections.append("other")

                    # edges.append(["other", "hobbies"])

                    edges.append(["hobbies", text])

                    # if ["other", "other"] not in edges:
                    #     edges.append(["other", "other"])
                    # edges.append(["hobbies", "hobbies"])
                    # edges.append([text, text])

                else:
                    continue

            elif key=="Objectives":
                if data[key] != "":
                    text = data[key]
                    text = text.replace("\r", " ")
                    text = text.replace("\t", " ")
                    text = text.replace("\n", " ")
                    contents.append(text)
                    items.append("objectives")
                    sections.append("other")

                    # edges.append(["other", "objectives"])

                    edges.append(["objectives", text])

                    # if ["other", "other"] not in edges:
                    #     edges.append(["other", "other"])
                    # edges.append(["objectives", "objectives"])
                    # edges.append([text, text])

                else:
                    continue

            elif key=="Achievements":
                if data[key] != "":
                    text = data[key]
                    text = text.replace("\r", " ")
                    text = text.replace("\t", " ")
                    text = text.replace("\n", " ")
                    contents.append(text)
                    items.append("achievements")
                    sections.append("other")

                    # edges.append(["other", "achievements"])

                    edges.append(["achievements", text])

                    # if ["other", "other"] not in edges:
                    #     edges.append(["other", "other"])
                    # edges.append(["achievements", "achievements"])
                    # edges.append([text, text])

                else:
                    continue

            else:
                continue

        content_dic["content"] = contents
        content_dic["item"] = items
        content_dic["section"] = sections
        total_dic[name] = content_dic
        total_edges[name] = edges
        if name in list(label_dic.keys()):
            total_dic["label"] = label_dic[name]
            total_edges["label"] = label_dic[name]

        else:
            print("error: no label!!!")
            exit(0)

        if name in train_keys:
            train_results.append(total_dic)
            train_edges.append(total_edges)
        elif name in dev_keys:
            dev_results.append(total_dic)
            dev_edges.append(total_edges)

        elif name in test_keys:
            test_results.append(total_dic)
            test_edges.append(total_edges)

        else:
            print("wtf, what happens?")


    # results.append(total_dic)
# print(results)
print("length of train set", len(train_results))
print("length of dev set", len(dev_results))
print("length of test set", len(test_results))
print("length of train set", len(train_edges))
print("length of dev set", len(dev_edges))
print("length of test set", len(test_edges))

# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_train.json', 'w', encoding='utf-8') as f:
#     # res = results[:60]
#     json.dump(train_results, f, ensure_ascii=False)
#
# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_dev.json', 'w', encoding='utf-8') as f:
#     # res = results[60:70]
#     json.dump(dev_results, f, ensure_ascii=False)
#
# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_test.json', 'w', encoding='utf-8') as f:
#     # res = results[70:80]
#     json.dump(test_results, f, ensure_ascii=False)

# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_train_edges_self.json', 'w', encoding='utf-8') as f:
#     json.dump(train_edges, f, ensure_ascii=False)
#
# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_dev_edges_self.json', 'w', encoding='utf-8') as f:
#     json.dump(dev_edges, f, ensure_ascii=False)
#
# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_test_edges_self.json', 'w', encoding='utf-8') as f:
#     json.dump(test_edges, f, ensure_ascii=False)

with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_train_edges_multi.json', 'w', encoding='utf-8') as f:
    json.dump(train_edges, f, ensure_ascii=False)

with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_dev_edges_multi.json', 'w', encoding='utf-8') as f:
    json.dump(dev_edges, f, ensure_ascii=False)

with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_test_edges_multi.json', 'w', encoding='utf-8') as f:
    json.dump(test_edges, f, ensure_ascii=False)

# with open('E:/Emory/Research/ResumeClassification/rchilli_parsed_data_graph/resume_graph_labels.json', 'w', encoding='utf-8') as f:
#     json.dump(label_dic, f, ensure_ascii=False)