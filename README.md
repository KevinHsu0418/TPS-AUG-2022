# TPS-AUG-2022

This repository is the machine learning final project on real-world competition(https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview). 

## Specification of dependencies

1. Using colab to run 109550147_Final_train.ipynb to generate model, and put model on right place.
2. Run 109550147_Final_inference.ipynb on colab to test your own model. 
3. If you want to reproduce my result, just download the pre-trained model below and put it on right place. So that you can get the result with my accuracy. 

## Training code

To train the model , run this code on colab:

#### Data Pre-process
```datapre
#read csv file
TRAIN_PATH = "/content/drive/MyDrive/Messi/Final/train.csv"
TEST_PATH = "/content/drive/MyDrive/Messi/Final/test.csv"
MODEL_PATH = "/content/drive/MyDrive/Messi/Final/model.pt"
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
cat_all = train_data.columns[2:25].tolist() #get out the feature name
val_ind = train_data.columns[25:26].tolist() #get out the target name

rep = {'material_5':5, 'material_6':6, 'material_7':7, 'material_8':8} #how to replace the material with number
#replace two columns in train_data and test_data
train_data = train_data.replace({'attribute_0': rep})
train_data = train_data.replace({'attribute_1': rep})
test_data = test_data.replace({'attribute_0': rep})
test_data = test_data.replace({'attribute_1': rep})

#fill in the nan value
#inspire by https://www.kaggle.com/code/tomjosephmo/logistic-regression-with-missing-values-accounted
for col in train_data.columns: #modify train data
    if not type(train_data[col][0]) == str: #if it is nan
        train_data[col].fillna(train_data[col].mean(), inplace=True)
        
for col in test_data.columns: #modify test data
    if not type(test_data[col][0]) == str: #if it is nan
        test_data[col].fillna(test_data[col].mean(), inplace=True)

device = "cuda" if torch.cuda.is_available() else "cpu" #if cuda is available use cuda
print(f"Using {device} device")

x_train = train_data[cat_all].values #get out the feature training data
y_train = train_data[val_ind].values #get out the target training data
y_train = y_train.flatten()
x_test = test_data[cat_all].values #get out the feature test data
```
#### Training Dataset
```traindataset
class TPS_train_dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)
        self.length = self.x_data.shape[0]

    def __getitem__(self, ind):
        return self.x_data[ind], self.y_data[ind] #return x data and y data

    def __len__(self):
        return self.length
#load training data
train_ds = TPS_train_dataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size = 16, shuffle = True) #need to shuffle
```
#### Model
```md
#three fully connected layer
class Model(nn.Module):
    def __init__(self, num_feature):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(num_feature, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1) #make output of the network to right dimension

    def forward(self, x_data):
        x_data = torch.relu(self.fc1(x_data)) #put in relu activation function
        x_data = torch.relu(self.fc2(x_data))
        x_data = torch.sigmoid(self.fc3(x_data)) #use sigmoid to make value between 0-1
        return x_data #return model
```
#### Training
```train
model = Model(num_feature = x_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.BCELoss() #Use binary cross entropy
losses = []
for i in range(250): #run 250 epochs
    for x, y in train_dl:
        output = model(x)
        loss = loss_fn(output, y.reshape(-1, 1)) #calculate the loss
        #updata
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
## Evaluation Code

To evaluation the model , run this code on colab:

#### Testing Dataset
```testdata
class TPS_test_dataset(Dataset):
    def __init__(self, x_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.length = self.x_data.shape[0]

    def __getitem__(self, ind):
        return self.x_data[ind] #test data dont have target feature so only return x

    def __len__(self):
        return self.length
test_ds = TPS_test_dataset(x_test)
test_dl = DataLoader(test_ds, batch_size = 1, shuffle = False)
```
#### Testing
```test
ind = np.array([i for i in range(26570, 47345)])
result = []
model.eval() #turn model to evalution mode
torch.save(model.state_dict(), MODEL_PATH)
for x_data in test_dl:
    pred = model(x_data) #predict the result
    pred = pred.detach().numpy()
    pred = pred.flatten()
    result = np.concatenate((result, pred)) #concate all result together

df = pd.DataFrame({"id":ind, "failure":result}, columns = ["id", "failure"]) #set dataframe
df.to_csv(f"109550147_submission.csv", index=False) #index set to false so that csv file dont have one extra column
```

## Pre-trained Model

You can download my pretrained model here:

- https://drive.google.com/file/d/1CdFe9PiPcRU_cKRWHs0f9uk1QsJjZTuJ/view?usp=sharing


## Results

My model achieves the result accuracy 0.59109 on private data

