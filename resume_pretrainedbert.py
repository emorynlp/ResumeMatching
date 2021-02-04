import json
import numpy as np
import networkx as nx
import re
import torch
from torch_geometric.data import Data


from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("/home/xdong57/Resume_Classification_new/bert_pretrain_output/checkpoint-95520", do_lower_case=True)
model = BertModel.from_pretrained("/home/xdong57/Resume_Classification_new/bert_pretrain_output/checkpoint-95520", output_hidden_states=True)
print("get model")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval() # Put the model in "evaluation" mode, meaning feed-forward operation
print("------------------load model successfully---------------")


embeddings_dict = {}
random_word_embeddings = {}
print("get embedding")

def get_doc_embedding(phrase):
    #input_ids = tokenizer(phrase, max_length=128, truncation=True)
    #input_ids = torch.tensor(input_ids, dtype=torch.long)
    #input_ids = input_ids.to(device)
    #input_ids = input_ids.unsqueeze(0)

    with torch.no_grad(): #tells PyTorch not to construct the compute graph during this forward pass this just reduces memory consumption and speeds things up a little
      #input_ids = tokenizer(phrase, return_tensors="pt",max_length=128, truncation=True)
      #input_ids=input_ids.unsqueeze(0)
      #input_ids=input_ids.to(device)
      input_ids = torch.tensor(tokenizer.encode(phrase, max_length=128, truncation=True)).unsqueeze(0)  # Batch size 1
      input_ids=input_ids.to(device)
      # print(outputs.shape)
      #hidden_states = model(input_ids)[2]
      #sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze().tolist()
      #print(sentence_embedding.shape)
      #sentence_embedding = np.asarray(hidden_states[-1].cpu().numpy()[0]).mean(axis=0)

      #print("sentence_embedding", sentence_embedding.shape)
    #return sentence_embedding  
      outputs = model(input_ids)[1].cpu().numpy()[0]

    return outputs  # The last hidden-state is the first element of the output tuple

#np.asarray(vectors).mean(axis=0)

total_labels = ["NQ", "CRCI", "CRCII", "CRCIII", "CRCIV"]
# total_sections = ["profile", "education", "skills", "work experience", "other", "activities"]
total_sections = ["education", "work", "other"]

section_mappings = {}
for i, item in enumerate(total_sections):
    section_mappings[item] = i
print(section_mappings)

label_mappings = {}
for i, item in enumerate(total_labels):
    label_mappings[item] = i
print(label_mappings)


def get_resume_graph(data_path):
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    final = []

    for item in data:
        nodes = []
        edges = []
        labels = []
        node_idx = 0

        keys = list(item.keys())
        resume = item[keys[0]]

        label = item[keys[1]]
        node_dict = {}

        for item in resume:

            if item[0] not in node_dict.values():
                node_dict[node_idx] = item[0]
                left = node_idx
                node_idx += 1
            else:
                for idx, value in node_dict.items():
                    if value == item[0]:
                        left = idx

            if item[1] not in node_dict.values():
                node_dict[node_idx] = item[1]
                right = node_idx
                node_idx += 1
            else:
                for idx, value in node_dict.items():
                    if value == item[1]:
                        right = idx
                # if re.match("^institute[0-9]$", txt)
            edges.append([left, right])
            edges.append([right, left])
            edges.append([left, left])
            edges.append([right, right])


        keys_sorted = sorted(list(node_dict.keys()))

        for key in keys_sorted:
            if re.match("^institute[0-9]$", node_dict[key]):
                context = "institute"
            elif re.match("^name[0-9]$", node_dict[key]):
                context = "name"
            elif re.match("^degree[0-9]$", node_dict[key]):
                context = "degree"
            elif re.match("^specialization[0-9]$", node_dict[key]):
                context = "specialization"
            elif re.match("^duration[0-9]$", node_dict[key]):
                context = "duration"
            elif re.match("^job[0-9]$", node_dict[key]):
                context = "job"
            elif re.match("^employer[0-9]$", node_dict[key]):
                context = "employer"
            elif re.match("^title[0-9]$", node_dict[key]):
                context = "title"
            elif re.match("^period[0-9]$", node_dict[key]):
                context = "period"
            elif re.match("^description[0-9]$", node_dict[key]):
                context = "description"
            else:
                context = node_dict[key]

            context_embedding = get_doc_embedding(context)
            #print("context embedding",context_embedding.shape)

            nodes.append(context_embedding)
            #print("len(nodes)", len(nodes))

        #print(len(nodes) == len(keys_sorted))

        if len(nodes) == 0:
            continue
        labels.append(label_mappings[label])

        # print("nodes", nodes)
        # print("edges", edges)
        # print("labels", labels)

        G = nx.Graph()
        G.add_edges_from(edges)
        print("length", len(nodes))
        print("nodes", G.number_of_nodes())

        edge_index = torch.tensor(edges, dtype=torch.long)
        x = torch.tensor(nodes, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        res = Data(x=x, edge_index=edge_index, y=y)
        # print("x.size(0)", x.size(0))
        # print("edge_index", edge_index)
        # print("torch.max(edge_index)", torch.max(edge_index))
        # print("torch.min(edge_index)", torch.min(edge_index))
        if torch.max(edge_index) > x.size(0):
            print("!!!!!!!!!!!!!!!!!!@#$%!!!!!!!!!!", keys[0])

        final.append(res)
    return final


train_dataset = get_resume_graph('./data/resume_train_edges_multi.json')
dev_dataset = get_resume_graph('./data/resume_dev_edges_multi.json')
test_dataset = get_resume_graph('./data/resume_test_edges_multi.json')

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of dev graphs: {len(dev_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

torch.save(train_dataset, './data/pretrainedbert/train_multi_self_cls.pt')
torch.save(dev_dataset, './data/pretrainedbert/dev_multi_self_cls.pt')
torch.save(test_dataset, './data/pretrainedbert/test_multi_self_cls.pt')

from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("------------------data load successfully--------------------------")

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(train_dataset[0].num_features, hidden_channels)
        self.lin = Linear(hidden_channels, 5)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


for lr in [0.001]:
    print("#########################",lr,"#############################################")
    # model = BERTGCN.from_pretrained("bert-uncased-base")
    model = GCN(hidden_channels=300).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        loss_all = 0

        for data in train_loader:  # Iterate in batches over the training dataset.
            # print(data.x)
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            loss_all += loss.item() * data.num_graphs
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return loss_all / len(train_dataset)

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            #loss = criterion(out, data.y)  # Compute the loss.
            #print('Test Loss', loss)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    best_dev_acc = 0
    best_test_acc = 0

    for epoch in range(1, 2001):
        train_loss = train()
        train_acc = test(train_loader)
        dev_acc = test(dev_loader)
        test_acc = test(test_loader)
        best_dev_acc = max(best_dev_acc, dev_acc)
        best_test_acc = max(best_test_acc, test_acc)
        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Dev Acc: {:.7f}, Test Acc: {:.7f}, Best Dev Acc: {:.7f}, Best Test Acc: {:.7f}'.format(epoch, train_loss, train_acc, dev_acc, test_acc, best_dev_acc, best_test_acc))
