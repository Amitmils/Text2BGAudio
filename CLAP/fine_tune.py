import os
import sys
working_dir = os.getcwd().split('TEXT2AUDIO')[0]
sys.path.append(working_dir)
from transformers import ClapModel,ClapProcessor
import torch 
import torch.nn.functional as F
from Dataset_Creation import audio_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchaudio.transforms as T
from multiprocessing import freeze_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def info_nce_loss(text_embeddings,audio_embeddings,labels,temperature=0.5):
    """
    Compute the InfoNCE loss for a batch of embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape (2N, embedding_dim), where N is the batch size,
                                   and the embeddings are organized in positive pairs.
        temperature (float): Temperature scaling factor for the softmax. Default is 0.5.

    Returns:
        torch.Tensor: InfoNCE loss value.
    """
    # Normalize the embeddings to unit vectors
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute pairwise similarity
    similarity_matrix = torch.mm(embeddings, embeddings.t())  # Shape: (2N, 2N)
    
    # Scale by temperature
    similarity_matrix /= temperature

    # Mask to exclude self-similarities
    N = embeddings.size(0) // 2
    labels = torch.cat([torch.arange(N), torch.arange(N)], dim=0).to(embeddings.device)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=embeddings.device)

    # Compute the numerator for positive pairs
    positive_sim = similarity_matrix[torch.arange(2 * N), labels]

    # Compute the denominator for all pairs except itself
    exp_sim = torch.exp(similarity_matrix) * mask
    denominator = exp_sim.sum(dim=1)

    # Loss computation
    loss = -torch.log(torch.exp(positive_sim) / denominator).mean()

    return loss

if __name__ == '__main__':
    freeze_support()
    model_name = "laion/clap-htsat-fused"
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    # Freeze most layers except projection layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze text and audio projection heads
    for param in model.text_projection.parameters():
        param.requires_grad = True
    for param in model.audio_projection.parameters():
        param.requires_grad = True


    music_dataset = audio_dataset.AudioDataset(r"_Data\Music\music_dataset_size7635.pt")
    batch_size = 32
    data_loader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(
    list(model.text_projection.parameters()) + list(model.audio_projection.parameters()),
    lr=1e-4
    )
    new_sr = 48000
    resampler = T.Resample(orig_freq=8000, new_freq=new_sr)

    for epoch in range(10):
        model.eval()
        total_loss = 0
        
        for batch in data_loader: # Replace with your DataLoader
            audio = batch[0]
            text = batch[1]
            inputs = processor(text=list(text), audios=audio.numpy(), return_tensors="pt", sampling_rate = new_sr,padding=True)
            # Print the shapes of inputs for debugging
            print(f"Text shape: {inputs['input_ids'].shape}")
            print(f"Audio shape: {inputs['input_features'].shape}")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds
            audio_embeds = outputs.audio_embeds
            
            loss = 0 #info_nce_loss(text_embeds, audio_embeds) #TODO : Make the loss function work with our setup (multiple true classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
    
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(data_loader)}")
