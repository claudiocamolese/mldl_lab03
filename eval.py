import torch
from models.customnet import CustomNet
from datasets import get_dataloaders

# Carica i dati
_, testloader, classes = get_dataloaders(batch_size=10000)

# Inizializza il modello e carica i pesi pre-addestrati
customnet = CustomNet()
customnet.load_state_dict(torch.load('customnet_model.pth'))  # Assicurati di aver salvato il modello

# Funzione di valutazione con accuratezza per classe
def evaluate(model, test_loader):
    model.eval()
    class_success = [0. for _ in range(10)]  # Successi per classe
    class_counter = [0. for _ in range(10)]  # Contatore per classe

    with torch.no_grad():
        for im, ground_truth in test_loader:
            op = model(im)  # Calcola le predizioni
            _, pred = torch.max(op, 1)
            correct = (pred == ground_truth).squeeze()

            for i in range(im.size(0)):  # Per ogni immagine nel batch
                gt = ground_truth[i]
                class_success[gt] += correct[i].item()
                class_counter[gt] += 1

    # Stampa l'accuratezza per classe
    for i in range(10):
        print(f'Model accuracy for class {classes[i]}: {100 * class_success[i] / class_counter[i]:.2f}%')

# Esegui la valutazione
evaluate(customnet, testloader)
