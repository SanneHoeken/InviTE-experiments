from tqdm import tqdm
import torch, random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from scipy.special import softmax
from sklearn.utils.class_weight import compute_class_weight
import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedKFold


class StratifiedBatchSampler:
    
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.indices = np.arange(len(y))
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8) , size=()).item()
        for _, batch_idx in self.skf.split(self.indices, self.y):
            yield batch_idx

    def __len__(self):
        #return len(self.y) 
        return self.skf.get_n_splits()
  
  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_to_dataloader(data_df, tokenizer, label_encoder, batch_size, 
                       sampling='stratified', is_test=False):

    texts = data_df['text'].to_list()
    labels = list(data_df['label'])
    labels = torch.tensor([label_encoder[l] for l in labels])

    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    data = TensorDataset(input_ids, attention_masks, labels)

    if is_test:
        dataloader = DataLoader(data, batch_size=batch_size)
    
    else:
        if sampling == 'stratified':
            sampler = StratifiedBatchSampler(labels, batch_size=batch_size)
            dataloader = DataLoader(dataset=data, batch_sampler=sampler)  
        else:
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        
    return dataloader


def train_model(model, optimizer, scheduler, train_data, test_data, 
                epochs, device, loss_type):
    
    lowest_loss = None
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} out of {epochs}...")
        print('Train...')
        model, avg_train_loss = train(model, optimizer, scheduler, train_data, device, loss_type)
        print(f"Average training loss: {avg_train_loss}")
        print('Evaluate...')
        _, _, accuracy = evaluate_f(model, test_data, device)
        print(f"Accuracy on dev set: {accuracy}")
          
        if lowest_loss == None or avg_train_loss < lowest_loss:
            lowest_loss = avg_train_loss
            best_model = model
        
    return best_model


def train(model, optimizer, scheduler, train_data, device, loss_type):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_data):
        batch = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(batch[0], attention_mask=batch[1], labels=batch[2]) #token_type_ids=None
        logits = outputs['logits']#.detach().cpu().numpy()
        gold = batch[2]#.detach().cpu().numpy()
        
        if loss_type == 'focal': 
            loss_fn = smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE)
            loss = loss_fn(logits, gold)
        
        elif loss_type == 'cross_entropy_custom':
            num_classes = logits.shape[1]
            if len(torch.unique(gold)) < 2: # uniform weights if only one class in batch
                class_weights = torch.ones(num_classes, dtype=torch.float32).to(device)
            else:
                all_classes = np.arange(num_classes)
                class_weights_np = compute_class_weight(class_weight="balanced", classes=all_classes,
                                                        y=gold.detach().cpu().numpy())
                class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            loss = criterion(logits, gold)

        else:
            loss = outputs['loss']
    
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_data)
    return model, avg_train_loss

        
def evaluate_f(model, test_data, device):
    model.eval()
    preds = []
    labels = []
    
    for batch in tqdm(test_data):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch[0], attention_mask=batch[1]) #token_type_ids=None, 
  
        logits = outputs[0].detach().cpu().numpy()        
        #pos_class_probabilities = softmax(logits, axis=1)[:, 1]
        gold = batch[2].to('cpu').numpy()
        preds.extend(np.argmax(logits , axis=1).tolist())
        labels.extend(gold.tolist())
    
    accuracy = accuracy_score(labels, preds)

    return preds, labels, accuracy


def train_for_sequence_classification(model_dir, output_dir, train_df, test_df, label_encoder, 
                                      batch_size, epochs, seed, loss_type, lr):
    
    #device = torch.device("mps")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    test_dataloader = data_to_dataloader(test_df, tokenizer, label_encoder, batch_size, is_test=True)
    num_labels = len(label_encoder)
    train_dataloader = data_to_dataloader(train_df, tokenizer, label_encoder, batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_model = train_model(model, optimizer, scheduler, train_dataloader, test_dataloader, 
                             epochs, device, loss_type)
    best_model.save_pretrained(output_dir)


def predict_sequence_classification(model_dir, tokenizer_dir, test_df, label_encoder, batch_size):   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
    test_dataloader = data_to_dataloader(test_df, tokenizer, label_encoder, batch_size, is_test=True)
    num_labels = len(label_encoder)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)
    print('Evaluate...')
    preds, labels, _ = evaluate_f(model, test_dataloader, device)
    label_decoder = {encoding: label for label, encoding in label_encoder.items()}
    preds = [label_decoder[p] for p in preds]
    #true_labels = [label_decoder[l] for l in labels]
    #print(classification_report(labels, preds))
    
    return preds