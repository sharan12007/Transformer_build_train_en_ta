import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path
import numpy as np
from model import Transformer
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import zipfile
from google.colab import files
import os
def buildTokenizer(dataset,lang:str,save_path:str) -> Tokenizer:
    save_path = Path(save_path)
    if save_path.exists():  
        tokenizer=Tokenizer.from_file(save_path)
        return tokenizer
    else:
        tokenizer=Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],
            min_frequency=2
        )
    def get_texts():
        for item in dataset["train"]:
            yield item[lang]
    tokenizer.train_from_iterator(get_texts(),trainer=trainer)
    tokenizer.save(str(save_path))
    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer,max_length: int = 50):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
    def encode(self,text,tokenizer:Tokenizer):
        tokens=tokenizer.encode(text).ids
        tokens = tokens[:self.max_length-2]
        return [2] + tokens + [3]
    def pad(self,token):
        if(len(token)<self.max_length):
            token=token+[1]*(self.max_length-len(token))
        return token
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item=self.dataset[idx]
        src=self.encode(item["en"],self.src_tokenizer)
        tgt=self.encode(item["ta"],self.tgt_tokenizer)
        return {
            "src": torch.tensor(self.pad(src)),
            "tgt": torch.tensor(self.pad(tgt))
        }

def pad_mask(seq):
    return (seq==1).unsqueeze(1).unsqueeze(2)
def look_ahead_mask(size):
    mask = torch.tril(torch.ones((size, size))).bool()
    return mask
def make_decoding_mask(tgt):
    tgt_pad_mask = pad_mask(tgt)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = look_ahead_mask(tgt_len).to(tgt.device)
    combined_mask = tgt_pad_mask & tgt_look_ahead_mask
    return combined_mask.unsqueeze(0).unsqueeze(1)

def train_model_with_warmup_cosine(model, trainloader, valloader, optimizer, criterion, device, epochs=10):
    
    warmup_epochs = 3
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,  
        eta_min=1e-6  )
    
    scaler = GradScaler() if device.type == 'cuda' else None
    train_losses = []
    val_losses = []

    print(f"Training using Warmup + Cosine Annealing")
    print(f"Warmup: {warmup_epochs} epochs, Cosine: {epochs - warmup_epochs} epochs")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(
            trainloader, 
            desc=f'🏋️ Epoch {epoch+1}/{epochs} [Train]',
            leave=True
        )
        
        for batch_idx, batch in enumerate(train_pbar):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)
            
            src_mask = pad_mask(src).to(device)
            tgt_mask = make_decoding_mask(tgt_input).to(device)

            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with autocast():
                    output = model(src, tgt_input, src_mask, tgt_mask)
                    output = output.reshape(-1, output.shape[-1])
                    loss = criterion(output, tgt_output.reshape(-1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(src, tgt_input, src_mask, tgt_mask)
                output = output.reshape(-1, output.shape[-1])
                loss = criterion(output, tgt_output.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_train_loss += loss.item()
            
            avg_loss_so_far = total_train_loss / (batch_idx + 1)
            current_lr = optimizer.param_groups[0]['lr']
            
            train_pbar.set_postfix({
                'loss': f'{avg_loss_so_far:.4f}',
                'lr': f'{current_lr:.2e}',
                'phase': 'Warmup' if epoch < warmup_epochs else 'Cosine'
            })
        
        avg_train_loss = total_train_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        
        val_pbar = tqdm(
            valloader, 
            desc=f'Epoch {epoch+1}/{epochs} [Val]',
            leave=False
        )
        
        with torch.no_grad():
            for batch in val_pbar:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                tgt_input = tgt[:, :-1].to(device)
                tgt_output = tgt[:, 1:].to(device)
                
                src_mask = pad_mask(src).to(device)
                tgt_mask = make_decoding_mask(tgt_input).to(device)

                output = model(src, tgt_input, src_mask, tgt_mask)
                output = output.reshape(-1, output.shape[-1])
                batch_loss = criterion(output, tgt_output.reshape(-1))
                total_val_loss += batch_loss.item()
                
                current_val_loss = total_val_loss / (val_pbar.n + 1)
                val_pbar.set_postfix({'val_loss': f'{current_val_loss:.4f}'})
        
        avg_val_loss = total_val_loss / len(valloader)
        val_losses.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            phase = "Warmup"
        else:
            cosine_scheduler.step()
            phase = "Cosine"
        
        print(f"Epoch {epoch+1}/{epochs} | {phase} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def translate_sentence(model, tokenizer_src, tokenizer_tgt, sentence, device, max_len=50):
    model.eval()
    src = [2] + tokenizer_src.encode(sentence).ids + [3]
    src = torch.tensor([src]).to(device)
    src_mask = pad_mask(src)

    memory = model.encoder(src, src_mask)
    tgt = torch.tensor([[2]]).to(device)  

    for _ in range(max_len):
        tgt_mask = make_decoding_mask(tgt)
        output = model.decoder(tgt, memory, src_mask, tgt_mask)
        output = model.output_linear(output)
        next_token = output.argmax(-1)[:, -1].item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
        if next_token == 3:  
            break
    decoded_tokens = tgt.squeeze(0).tolist()
    decoded_tokens = [token for token in decoded_tokens if token not in [2, 3]]
    decoded = tokenizer_tgt.decode(decoded_tokens)
    return decoded


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset=load_dataset("Hemanth-thunder/en_ta")
    ds_train=dataset["train"]
    ds_val=dataset["validation"]
    ds_test=dataset["test"]

    src_tokenizer=buildTokenizer(dataset,"en","src_tokenizer.json")
    tgt_tokenizer=buildTokenizer(dataset,"ta","tgt_tokenizer.json")

    train_dataset=TranslationDataset(ds_train,src_tokenizer,tgt_tokenizer)
    val_dataset=TranslationDataset(ds_val,src_tokenizer,tgt_tokenizer)
    test_dataset=TranslationDataset(ds_test,src_tokenizer,tgt_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)     

    model = Transformer(
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_model=512,
    num_heads=8,
    ff_dim=2048,
    src_vocab_size=src_tokenizer.get_vocab_size(),
    tgt_vocab_size=tgt_tokenizer.get_vocab_size()
).to(device)
    
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    criterion=nn.CrossEntropyLoss(ignore_index=1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,          
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    train_model_with_warmup_cosine(model, train_loader, val_loader, optimizer, criterion, device, epochs=15)    
    
    torch.save(model.state_dict(), "transformer_final.pth")
    src_tokenizer.save("src_tokenizer_final.json")
    tgt_tokenizer.save("tgt_tokenizer_final.json")
    print("\nTranslating:")
    test_sentence = "How are you?"
    result = translate_sentence(model, src_tokenizer, tgt_tokenizer, test_sentence, device)
    print(f"EN: {test_sentence}")
    print(f"TA: {result}")
    print("\n💾 Saving model and tokenizers into a zip file ")
    with zipfile.ZipFile("transformer_complete_package.zip", 'w') as zipf:
        zipf.write("transformer_final.pth")
        zipf.write("src_tokenizer_final.json")
        zipf.write("tgt_tokenizer_final.json")
    print("📦 Downloading trained model to your computer...")
    if os.path.exists("transformer_complete_package.zip"):
        files.download("transformer_complete_package.zip")
        print("Download complete! Check your downloads folder.")
    else:
        print("Error: Zip file not created!")

if __name__ == "__main__":
    main()