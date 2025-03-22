import torch
from models.customnet import CustomNet
from datasets import get_dataloaders

# Carica i dati
_, testloader, _ = get_dataloaders(batch_size=10000)

# Inizializza il modello e carica i pesi pre-addestrati
customnet = CustomNet()
customnet.load_state_dict(torch.load('customnet_model.pth'))  # Assicurati di aver salvato il modello

# Funzione di test
def test(model, test_loader):
    model.eval()
    success = 0
    counter = 0

    with torch.no_grad():
        for im, ground_truth in test_loader:
            op = model(im)  # Calcola le predizioni
            _, pred = torch.max(op, 1)
            correct = (pred == ground_truth).squeeze()
            counter += ground_truth.size(0)
            success += correct.sum().item()

    accuracy = 100 * success / counter
    print(f'Model accuracy on test dataset: {accuracy:.2f}%')

# Esegui il test
test(customnet, testloader)
