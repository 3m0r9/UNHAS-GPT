import torch
import torch.nn as nn
from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom dataset class that loads and preprocesses the text data
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # Preprocess the text using the BERT tokenizer
        encoded_text = tokenizer.encode(text, return_tensors='pt')

        # Convert the label to a tensor
        label = torch.tensor(label)

        # Return the preprocessed text and label as a dataset tuple
        return encoded_text, label

    def __len__(self):
        return len(self.texts)

# Load the text data and labels
train_texts = ["This is an example sentence.", "This is another example sentence."]
train_labels = [1, 1]

test_texts = ["This is a test sentence.", "This is another test sentence."]
test_labels = [0, 0]

# Create a custom dataset from the text data and labels
train_dataset = BertDataset(train_texts, train_labels)
test_dataset = BertDataset(test_texts, test_labels)

# Define a simple neural network model
class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = tokenizer.bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        hidden_state = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = self.dropout(hidden_state)
        output = self.classifier(hidden_state)
        return output

# Initialize the model, optimizer, and scheduler
model = BertModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.swa_utils.SWALR(optimizer, 0.001, 0.5, 2)

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_dataset:
        input_ids = batch[0].to(device)
        attention_mask = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}')

# Evaluate the model on the test set
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for batch in test_dataset:
        input_ids = batch[0].to(device)
        attention_mask = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(test_dataset)
print(f'Test Loss: {test_loss / len(test_dataset)}')
print(f'Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'bert_model.pth')

# Load the trained model
model.load_state_dict(torch.load('bert_model.pth'))

#Use the trained model to make predictions on new data
new_data = ["This is a new sentence."]
new_data = tokenizer.encode(new_data, return_tensors='pt')
prediction = model(new_data)
predicted_label = torch.argmax(prediction)
print(f'Predicted label: {predicted_label}')





