import torch
import torch.nn as nn
import torch.optim 
from datasets import get_dataloaders
from models.customnet import CustomNet  # Importa il modello definito in model.py

# Carica i dati
trainloader, testloader, classes = get_dataloaders(batch_size=8)

# Inizializza il modello
customnet = CustomNet()

# Definisci la funzione di perdita e l'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(customnet.parameters(), lr=0.001)

# Funzione di training
def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Stampa ogni 100 batch
            print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}")
            running_loss = 0.0

# Ciclo di addestramento
for epoch in range(50):  # 50 epoche
    train(customnet, trainloader, optimizer, epoch)  # Ciclo di addestramento
    print(f"Finished Epoch {epoch + 1}")

# Salva il modello allenato
torch.save(customnet.state_dict(), 'customnet_model.pth')
print("Training Finished and Model Saved!")
