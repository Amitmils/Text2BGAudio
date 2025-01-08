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

    print("Loading Data...")
    train_data, val_data, test_data = list(torch.load(r"_Data\Music\Music Data New\music_dataset_test_Music Data New_tr3204_val398_te405.pt", weights_only=False).values())
    print("Data Loaded!")
    train_dataset = audio_dataset.AudioDataset(train_data)
    val_dataset = audio_dataset.AudioDataset(val_data)
    test_dataset = audio_dataset.AudioDataset(test_data)

    batch_sizes = [32]  # 32
    lrs = [1e-5]  # 1e-7
    runs_summary = {"train": dict(), "val": dict()}
    for batch_size in batch_sizes:
        for lr in lrs:
            train_data_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_data_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True
            )

            optimizer = optim.AdamW(
                list(model.text_projection.parameters())
                + list(model.audio_projection.parameters()),
                lr=lr,
            )
            train_loss_per_epoch = []
            val_loss_per_epoch = []

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                print(f"Epoch {epoch+1}")
                for batch in tqdm(train_data_loader, desc="Train Batches"):
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

                    total_loss += loss.item()
                train_loss_per_epoch.append(total_loss / len(train_data_loader))

                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    for batch in tqdm(val_data_loader, desc="Validation Batches"):
                        audio = batch[0]
                        labels = list(batch[1])
                        inputs = processor(
                            text=labels,
                            audios=audio.numpy(),
                            return_tensors="pt",
                            sampling_rate=new_sr,
                            padding=True,
                        )
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                        outputs = model(**inputs)
                        text_embeds = outputs.text_embeds
                        audio_embeds = outputs.audio_embeds

                        loss = NT_Xent_loss(text_embeds, audio_embeds, labels)
                        val_loss += loss.item()
                    val_loss_per_epoch.append(val_loss / len(val_data_loader))
                print(
                    f"Train Loss = {total_loss / len(train_data_loader)} | Validation Loss = {val_loss / len(val_data_loader)}\n"
                )
                if (epoch + 1) % 50 == 0:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "train_Loss_Per_Epoch": train_loss_per_epoch,
                            "val_Loss_Per_Epoch": val_loss_per_epoch,
                        },
                        rf"CLAP\models\clap_fine_tunned_BatchSize_{batch_size}_LR_{lr}_Epochs_{epoch+1}_VAL_LOSS_{val_loss_per_epoch[-1]:.2f}.pt",
                    )
