import torch
from dataset import causal_mask
from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

def translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, max_len, device):
    """Translate English sentence to Italian."""
    model.eval()
    with torch.no_grad():
        sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
        
        enc_input_tokens = tokenizer_src.encode(sentence).ids
        enc_num_padding_tokens = max_len - len(enc_input_tokens) - 2
        
        if enc_num_padding_tokens < 0:
            print(f"Sentence was too long! Maximum length: {max_len - 2} tokens")
            return None
        
        encoder_input = torch.cat([
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ], dim=0).unsqueeze(0).to(device)
        
        encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)
        
        # Greedy decode
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')
        
        # Precompute encoder output
        encoder_output = model.encode(encoder_input, encoder_mask)
        
        # Initialize decoder input với SOS token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
        
        while True:
            if decoder_input.size(1) == max_len:
                break
            
            # Build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
            
            # Calculate output
            out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            # Get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([
                decoder_input, 
                torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)
            ], dim=1)
            
            if next_word == eos_idx:
                break
        
        model_out_text = tokenizer_tgt.decode(decoder_input.squeeze(0).detach().cpu().numpy())
        return model_out_text

print("\n=== Demo with validation set ===")
run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=1)

print("\n" + "="*50)
print("TRANSLATOR - place your English sentence below to translate to Italian")
print("Type 'quit' or 'exit' to end the program.")
print("="*50)

while True:
    try:
        user_input = input("\nInput your English Sentence: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please input your sentence!")
            continue
        
        # Dịch câu
        translation = translate_sentence(model, user_input, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        
        if translation:
            print(f"Translated: {translation}")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")