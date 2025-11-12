import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from model import Transformer

def pad_mask(seq):
    return (seq == 1).unsqueeze(1).unsqueeze(2)

def look_ahead_mask(size):
    mask = torch.tril(torch.ones((size, size))).bool()
    return mask

def make_decoding_mask(tgt):
    tgt_pad_mask = pad_mask(tgt)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = look_ahead_mask(tgt_len).to(tgt.device)
    combined_mask = tgt_pad_mask & tgt_look_ahead_mask
    return combined_mask.unsqueeze(0).unsqueeze(1)

def run_model(max_len=50, temperature=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    src_tokenizer = Tokenizer.from_file("bpe_en_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("bpe_ta_tokenizer.json")
    
    # Print tokenizer info
    print(f"Source vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"Target vocab size: {tgt_tokenizer.get_vocab_size()}")
    
    model = Transformer(
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_model=512,
        num_heads=8,
        ff_dim=2048,
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size()
    ).to(device)
    
    print("Entering the Evaluation Stage ... \n")
    
    try:
        checkpoint = torch.load("checkpoint_epoch_7.pth", map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print("Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval() 
    
    while True:
        user_input = input("Enter the input in English: ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        if not user_input:
            continue
            
        try:
            input_tokens = src_tokenizer.encode(user_input).ids
            print(f"Input tokens: {input_tokens}")
            print(f"Input text: '{user_input}'")
            
            tokens = [2] + input_tokens[:max_len-2] + [3]  
            
            if len(tokens) < max_len:
                tokens = tokens + [1] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
                
            src = torch.tensor([tokens], dtype=torch.long).to(device)
            src_mask = pad_mask(src)
            
            tgt = torch.tensor([[2]], dtype=torch.long).to(device)
            
            with torch.no_grad():
                memory = model.encoder(src, src_mask)
            
            print("Generating translation...")
            all_tokens = []
            
            for i in range(max_len):
                with torch.no_grad():
                    tgt_mask = make_decoding_mask(tgt)
                    output = model.decoder(tgt, memory, src_mask, tgt_mask)
                    output = model.output_linear(output)
                    probabilities = torch.softmax(output[:, -1, :] / temperature, dim=-1)
                    print(probabilities[:, :10])  
                    top_probs, top_indices = torch.topk(probabilities, 5)
                    
                    for j in range(5):
                        token_id = top_indices[0, j].item()
                        prob = top_probs[0, j].item()
                        token_text = tgt_tokenizer.decode([token_id]) if token_id not in [1,2,3] else f"[special:{token_id}]"
                    
                    next_token = torch.multinomial(probabilities, 1).item()
                    all_tokens.append(next_token)
                    
                    tgt = torch.cat([
                        tgt, 
                        torch.tensor([[next_token]], dtype=torch.long).to(device)
                    ], dim=1)
                    
                    if next_token == 3:
                        print("End token (3) generated")
                        break
            
            decoded_tokens = tgt.squeeze(0).tolist()[1:]  # Remove SOS token
            decoded_tokens = [token for token in decoded_tokens if token not in [2, 3]]
            decoded_text = tgt_tokenizer.decode(decoded_tokens)
            
            print(f"\nTranslated output: {decoded_text}\n")
        except Exception as e:
            print(f"Error during translation: {e}\n")
            
def main():
    run_model(50,0.7)

if __name__ == "__main__":
    main()