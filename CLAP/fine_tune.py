import os
import sys

working_dir = os.getcwd().split("TEXT2AUDIO")[0]
sys.path.append(working_dir)
from transformers import ClapModel, ClapProcessor
import torch
import torch.nn.functional as F
from Dataset_Creation import audio_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchaudio.transforms as T
from multiprocessing import freeze_support
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Device Name:", torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device("cpu")
    print("Device Name: CPU")


def NT_Xent_loss(text_embeddings, audio_embeddings, labels, temperature=0.5):
    """
    Compute the InfoNCE loss for a batch of embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape (2N, embedding_dim), where N is the batch size,
                                   and the embeddings are organized in positive pairs.
        temperature (float): Temperature scaling factor for the softmax. Default is 0.5.

    Returns:
        torch.Tensor: InfoNCE loss value.
    """
    # Normalize embeddings
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Compute similarity matrix between audio and class embeddings
    similarity_matrix = torch.mm(audio_embeddings, text_embeddings.t())

    # Scale by temperature
    similarity_matrix /= temperature

    label_to_index = {label: idx for idx, label in enumerate(set(labels))}
    encoded_labels = torch.tensor([label_to_index[label] for label in labels])
    loss = 0
    for i in range(audio_embeddings.size(0)):
        # Find the text embeddings for the same class as the current audio
        positive_indices = (encoded_labels == encoded_labels[i]).nonzero(as_tuple=True)[
            0
        ]  # All text indices with same label as current audio
        negative_indices = (encoded_labels != encoded_labels[i]).nonzero(as_tuple=True)[
            0
        ]  # All text indices with same label as current audio

        # Create logits for positive pairs (text embeddings of the same class as current audio)
        positive_logits = similarity_matrix[
            i, positive_indices
        ]  # Select positive similarities for current audio

        # Denominator: sum of all exponentials (for contrastive part)
        denom = torch.sum(
            torch.exp(similarity_matrix[i])
        )  # Sum over all text embeddings

        # Compute the NT-Xent loss
        loss += -torch.log(
            torch.sum(torch.exp(positive_logits)) / denom
        )  # Sum over positive pairs

    return loss


if __name__ == "__main__":
    freeze_support()
    model_name = "laion/larger_clap_music"
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(DEVICE)

    # # Switch BatchNorm layers to evaluation mode during training
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.BatchNorm2d):
    #         layer.eval()
    # Freeze most layers except projection layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze text and audio projection heads
    for param in model.text_projection.parameters():
        param.requires_grad = True
    for param in model.audio_projection.parameters():
        param.requires_grad = True

    new_sr = 48000
    epochs = 400
    resampler = T.Resample(orig_freq=8000, new_freq=new_sr)

    music_dataset = audio_dataset.AudioDataset(r"_Data\Music\music_dataset_train_size7507.pt")
    batch_sizes = [32]  # 32
    lrs = [1e-5]  # 1e-7
    runs_summary = dict()
    for batch_size in batch_sizes:
        for lr in lrs:
            data_loader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)
            optimizer = optim.AdamW(
                list(model.text_projection.parameters())
                + list(model.audio_projection.parameters()),
                lr=lr,
            )
            loss_per_epoch = []
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                print(f"Epoch {epoch+1}")
                for batch in tqdm(data_loader, desc="Batches"):
                    audio = batch[0]
                    labels = list(batch[1])
                    inputs = processor(
                        text=labels,
                        audios=audio.numpy(),
                        return_tensors="pt",
                        sampling_rate=new_sr,
                        padding=True,
                    )
                    # Print the shapes of inputs for debugging
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    text_embeds = outputs.text_embeds
                    audio_embeds = outputs.audio_embeds

                    loss = NT_Xent_loss(text_embeds, audio_embeds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % 50 == 0:
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "Loss_Per_Epoch": loss_per_epoch,
                            },
                            rf"CLAP\models\clap_fine_tunned_BatchSize_{batch_size}_LR_{lr}_Epochs_{epoch+1}_LOSS_{loss_per_epoch[-1]:.2f}.pt",
                        )
                    total_loss += loss.item()
                print(
                    f"Batch {batch_size} LR {lr} Epoch {epoch+1}: Loss = {total_loss / len(data_loader)}"
                )
                loss_per_epoch.append(total_loss / len(data_loader))
            runs_summary[(lr, batch_size)] = loss_per_epoch.copy()
        for key in runs_summary:
            print(f"Batch Size: {key[1]} LR: {key[0]} Loss: {runs_summary[key]}")
