'''
This skript builds the C4T architecture consisting of two encoders (inspired by CLIP), trains it with the generated image-text pairs (A2D2) and enables image-retrieval
It can be used with a pretrained model (C4T or CLIP) to encode/embedd images and texts
'''

# Import Packages
import os
import cv2
import gc                            
import numpy as np
import pandas as pd
import itertools
import re
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
import clip
from torch import nn
from PIL import Image
from tqdm import tqdm
import statistics
from datetime import datetime
import torch.nn.functional as F
import timm
import textwrap
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from sklearn.metrics import auc

from torch.utils.tensorboard import SummaryWriter


# 1. PARAMETERS

# Set date
current_datetime = datetime.now().strftime('%Y%m%d')
current_datetime = str(current_datetime)

# Paths (ADD PATHS!)
captions_path = [xxx]   #generated image_captions as csv
images_path = [xxx]     #folder with camera images
model_path = [xxx]      #to save trained model or load existing one

image_embeddings_C4T = [xxx.npy]    #save/load image_embeddings
image_embeddings_CLIP = [xxx.npy]

# Model/architecture params
class CFG:
  debug = False
  image_path = images_path
  captions_path = captions_path
  batch_size = 16 
  num_workers = 4                  #describes how many threads will be used to load the data 
  head_lr = 1e-3
  image_encoder_lr = 1e-4 
  text_encoder_lr = 1e-5 
  weight_decacy = 1e-3             #regularization technique by adding a small penalty (L2 norm of the weights) to the loss function
  patience = 1
  factor = 0.5
  epochs = 6
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model_name = 'resnet50'                           #image-encoder
  image_embedding = 2048
  text_encoder_model = 'distilbert-base-uncased'    #text-encoder
  text_embedding = 768
  text_tokenizer = 'distilbert-base-uncased'
  max_length = 200

  pretrained = True #for both encoders
  trainable = True  #for both encoders
  temperature = 1.0 #can be used instead of k parameter in the exponent of the cross entropy/loss --> with large T at the beginning: no values in the probability distribution are close to zero, so the gradients get propageted more easily
                    #--> Increasing T: flatter distribution ('annealing': start with high T and reduce during training)
                    #comparable to learning rate: no direct correlation to accuracy


  #image size
  size = 224        #images have to be downsized (same size as CLIP images)

  #for the projection head: used for image & text encoder
  num_projection_layers = 1
  projection_dim = 512
  dropout = 0.1

# Parameter management
class AvgMeter:
    def __init__(self, name='Metric'):
      self.name = name
      self.reset()

    def reset(self):
      self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
      self.count += count
      self.sum += val * count
      self.avg = self.sum / self.count

    def __repr__(self):
      text = f"{self.name}: {self.avg:.4f}"
      return text
  
def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']



#2. Dataset

# Read prepared captions-dataframe
captions_df = pd.read_csv(captions_path)
# Change/Standardize column_names if necessary (example for given A2D2) --> 'id', 'image', 'caption'
captions_df = captions_df.rename(columns={'image_id': 'id', 'camera image': 'image'})

# Pytorch Dataset class
class C4TDataset(torch.utils.data.Dataset):
    '''
    get image_filenames, images and corresponding captions
    preprocess them for pipeline
    '''
    def __init__(self, image_filenames, captions, tokenizer, transforms):
      '''
      image filenames and captions must have the same length, so if there are multiple captions 
      for each image, the image_filenames must have repetitive file names
      '''
      self.image_filenames = image_filenames
      self.captions = list(captions)
      self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=CFG.max_length) #docu: https://huggingface.co/transformers/preprocessing.html

      self.transforms = transforms
    
    def __getitem__(self, idx):
      '''
      load encoded image (dictionary with keys: 'input_ids', 'attention_mask') and convert to tensor
      load corresponding image, transform and augment it and safe in dictionary with key 'image'
      '''
      item = {
          key: torch.tensor(values[idx])
          for key, values in self.encoded_captions.items()
      }

      image = cv2.imread(f'{CFG.image_path}/{self.image_filenames[idx]}')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = self.transforms(image=image)["image"]
      
      item["image"] = torch.tensor(image).permute(2, 0, 1).float() #permute(2, 0, 1) switches the axes of the tensor - dimensions permuted
      item["caption"] = self.captions[idx]

      return item
    
    def __len__(self):
      return len(self.captions)
  

def get_transforms(mode='train'):
    if mode == 'train':
      return A.Compose(
          [
              A.Resize(CFG.size, CFG.size, always_apply=True),
              A.Normalize(max_pixel_value=255.0, always_apply=True),
          ]
      )
    else:
      return A.Compose(
          [
              A.Resize(CFG.size, CFG.size, always_apply=True),
              A.Normalize(max_pixel_value=255.0, always_apply=True),
          ]
      )


# 3. ENCODERS

# A. Image Encoder
class ImageEncoder(nn.Module):
    '''
    Encode image to a fixed size vector (ResNet-50: 2048)
    '''
    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable): #we initiliaze with pretrained ImageNet weights
      super().__init__()  #delegates the function to the parent class, initialize nn.Module 
      self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")

      for p in self.model.parameters():
        p.requires_grad = trainable #stores gradients for training

    def forward(self, x):
      return self.model(x)

# B. Text Encoder
'''
We will use DistilBERT as text encoder. Very similar to BERT, but lighter & faster.

Two special tokens will be added to the actual input tokens:
CLS ('Classification'): Represents sentence-level classification, that represents the meaning of the sentence ▶ make pooling scheme of BERT work
SEP ('Seperation'): Indicated where next sentence begins

To get the whole representation of sentence, we use the final representation of the CLS token and claim, that it captures the overall meaning of the caption. In this way it's a similar procedure as with the images, converting them to a fixed size vector.
▶ The output hidden representation for each token is a vector with the size 768.
▶ The whole caption will be encoded in the CLS token representation (size = 768).
'''

class TextEncoder(nn.Module):
    '''
    Encode the text to a fixed size vector (DistilBERT: 768)
    '''
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable): #initialize with pretrained weights to gain generalization
      super().__init__()
      
      if pretrained:
        self.model = DistilBertModel.from_pretrained(model_name)
      else:
        self.model = DistilBertModel(config=DistilBertConfig())

      for p in self.model.parameters():
        p.requires_grad = trainable #stores gradients for training
      
      # we are using the CLS token hidden representation as the sentence's embedding
      self.target_token_idx = 0
      
    def forward(self, input_ids, attention_mask):
      output = self.model(input_ids=input_ids, attention_mask=attention_mask)
      last_hidden_state = output.last_hidden_state #hidden state = output of each layer (Transformer: Several similar layers stacked on top of each other [output layer n-1 = input layer n])
      return last_hidden_state[:, self.target_token_idx, :]

# 4. Projection Head
'''
Now that the image (vector size: 2048) and text (vector size: 768) are encoded, we need to project them into the same dimension, in order to be able to compare them.
The following 5-layer fully-connected network (ProjectionHead) brings both vectors into a 512 dimensional world.
'''
class ProjectionHead(nn.Module):
    '''
    Project both vectors into the same 512 dimension/length
    '''
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
      '''embedding_dim: size of the input vector | projection_dim: size of the output vector'''
      super().__init__()
      self.projection = nn.Linear(embedding_dim, projection_dim) #nn.Linear: applies a linear transformation to the incoming data (y = x*W^T + b) [params: in_features, out_features, bias]
      self.gelu = nn.GELU() #activation function: applies the gaussian error linear units function
      self.fc = nn.Linear(projection_dim, projection_dim)
      self.dropout = nn.Dropout(dropout) #randomly zeroes some of the elements of the input tensor with probability p. Each channel will be zeroed out independently on every forward call. Technique for regularization.
      self.layer_norm = nn.LayerNorm(projection_dim) #applies Layer Normalization over a mini-batch inputs, to reduce training and computing time

    def forward(self, x):
      projected = self.projection(x)
      x = self.gelu(projected)
      x = self.fc(x)
      x = self.dropout(x)
      x = x + projected
      x = self.layer_norm(x)
      return x

# 5. CLIP4Traffic (C4T)
'''
First, we encode the images and texts seperately, using the corresponding encoder. Afterwards we get the in the same dimension, using our Projection Head, resulting in vectors of size 512. 
These embeddings are now matrices with the shape (batch_size, 512) and are therefore compared using the dot product. 
Afterwards, we calculate our target and the loss for both instances, using the cross entropy. 
'''
class C4TModel(nn.Module):
    '''
    Bring the encoders together and embedd them trough the projection-heads
    '''
    def __init__(self, temperature=CFG.temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
      super().__init__()
      self.image_encoder = ImageEncoder()
      #print(f'image encoder params: {sum(p.numel() for p in self.image_encoder.parameters())}') #if interested in the encoder/model parameter count
      self.text_encoder = TextEncoder()
      #print(f'text encoder params: {sum(p.numel() for p in self.text_encoder.parameters())}')
      self.image_projection = ProjectionHead(embedding_dim=image_embedding)
      #print(f'image encoder projection params: {sum(p.numel() for p in self.image_projection.parameters())}')
      self.text_projection = ProjectionHead(embedding_dim=text_embedding)
      #print(f'text encoder projection: {sum(p.numel() for p in self.text_projection.parameters())}')
      self.temperature = temperature

    def forward(self, batch):
      '''
      encode image and text seperately into fixed size vectors and project them into the same dimension
      '''
      # get image and text features
      image_features = self.image_encoder(batch["image"])
      text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

      # get image and text embeddings (same dimension [vector size: 512]) --> types: matrix with shape (batch_size, 512)
      image_embeddings = self.image_projection(image_features)
      text_embeddings = self.text_projection(text_features)

      # calculate the loss
      logits = (text_embeddings @ image_embeddings.T) / self.temperature #dot product to compare matrices [shape(batch_size, batch_size)]
      images_similarity = image_embeddings @ image_embeddings.T #'.T' expects the tensor input to be <=2-D and transposes dimensions to 0 and 1
      texts_similarity = text_embeddings @ text_embeddings.T    #needed for the target
      targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
      texts_loss = cross_entropy(logits, targets, reduction='none') #Loss reduction (.sum() or .mean()): https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641
      images_loss = cross_entropy(logits.T, targets.T, reduction='none')
      loss = (images_loss + texts_loss) / 2.0   #shape(batch_size)
      return loss.mean()

def cross_entropy(preds, targets, reduction='none'): 
  '''calculate the loss to make the models prediction similar to the target'''
  log_softmax = nn.LogSoftmax(dim=-1)
  loss = (-targets * log_softmax(preds)).sum(1)
  if reduction == 'none':
    return loss
  elif reduction == 'mean':
    return loss.mean()

'''
simpler/alternative approach: nn.CrossEntropyLoss()(logits, torch.arrange(batch_size))
reason: there's more then one caption per image - possibility that two identical images with similar captions exists in a batch
'''

# 6. TRAIN
'Only neeeded if a new model should be trained using the parameters and architecture from 1.-5'

# A. Training loop & utility functions
def make_train_valid_dfs():
  '''take id's from the captions dataframe and split into train (80%) and validation (20%) dataframe'''
  dataframe = captions_path
  max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
  image_ids = np.arange(0, max_id)
  np.random.seed(42)

  valid_ids = np.random.choice(image_ids, size=int(0.2 * len(image_ids)), replace=False)
  train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
  train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
  valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
  return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
  '''build pytorch dataloaders'''
  transforms = get_transforms(mode=mode)
  dataset = C4TDataset(dataframe["image"].values, dataframe["caption"].values, tokenizer=tokenizer, transforms=transforms)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True if mode == 'train' else False)
  return dataloader

# B. Training & Validation functions
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
  loss_meter = AvgMeter()
  tqdm_object = tqdm(train_loader, total=len(train_loader))
  for batch in tqdm_object:
    batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
    loss = model(batch)
    optimizer.zero_grad() #set the gradients to zero for every mini-batch during training before starting the backprop (upadating weights & biases) --> clean old gradient before calculating the new one
    loss.backward()       #computes loss/gradient for every parameter
    optimizer.step()      #optimizer iterates over all parameters it's supposed to update and use their internally stored grad to update their values

    if step == "batch":
      lr_scheduler.step()

    count = batch["image"].size(0)
    loss_meter.update(loss.item(), count)
    tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    loss.detach()   #to detach loss from graph, otherwise GPU overloads
  return loss, loss_meter

def valid_epoch(model, valid_loader):
  loss_meter = AvgMeter()

  tqdm_object = tqdm(valid_loader, total=len(valid_loader))
  for batch in tqdm_object:
    batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
    loss = model(batch)

    count = batch["image"].size(0)
    loss_meter.update(loss.item(), count)
    tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    loss.detach()
  return loss, loss_meter

# C. Run training and validation
def main():
  train_df, valid_df = make_train_valid_dfs()
  tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
  train_loader = build_loaders(train_df, tokenizer, mode='train')
  valid_loader = build_loaders(valid_df, tokenizer, mode='valid')

  model = C4TModel().to(CFG.device)
  params = [
      {'params': model.image_encoder.parameters(), 'lr': CFG.image_encoder_lr},
      {'params': model.text_encoder.parameters(), 'lr': CFG.text_encoder_lr},
      {'params': itertools.chain(model.image_projection.parameters(), model.text_projection.parameters()), 'lr': CFG.head_lr, 'weight_decay': CFG.weight_decacy}
  ]
  optimizer = torch.optim.AdamW(params, weight_decay=0.) #AdamW corrects the 'wrong' weight decay application of the Adam (source: https://www.fast.ai/2018/07/02/adam-weight-decay/)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=CFG.patience, factor=CFG.factor) #Reduces learning rate when a metric has stopped improving ('min': lr will be reduced when the quantity monitored has stopped decreasing, 'patience': number of epochs with no improvement, 'factor':  Factor by which the learning rate will be reduced --> new_lr = lr * factor)
  step = "epoch"

  tb = SummaryWriter(comment='A2D2_double_4E') #source: https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3

  best_loss = float('inf')
  for epoch in range(CFG.epochs):
    print(f'Epoch: {epoch + 1}')
    model.train()
    train_loss, train_loss_meter = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
    model.eval()
    with torch.no_grad():
      valid_loss, valid_loss_meter = valid_epoch(model, valid_loader)
      print(f'Validation loss: {valid_loss_meter}')
    
    # add metrics to tensorboard
    tb.add_scalar("Train Loss", train_loss, epoch)
    tb.add_scalar("Valid Loss", valid_loss, epoch)
    tb.close()
    
    if valid_loss_meter.avg < best_loss:
      best_loss = valid_loss_meter.avg
      torch.save(model.state_dict(), model_path_A2D2_double)
      print('Saved best model!')
    
    lr_scheduler.step(valid_loss_meter.avg)
    
    # check memory consumption 
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))        #current GPU memory occupied by device
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))          #current GPU memory managed by the caching allocator 
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    # empty cache
    torch.cuda.empty_cache()
    torch._C._cuda_emptyCache() #[source: https://github.com/pytorch/pytorch/issues/1529]
    print("torch.cuda.memory_allocated (cleaned cache): %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))        
    print("torch.cuda.memory_reserved (cleaned cache): %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))          
    print("torch.cuda.max_memory_reserved (cleaned cache): %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    
    train_loss.detach(), valid_loss.detach()
    del train_loss_meter, valid_loss_meter, train_loss, valid_loss
    print("torch.cuda.memory_allocated (deleted): %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))        
    print("torch.cuda.memory_reserved (deleted): %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))          
    print("torch.cuda.max_memory_reserved (deleted): %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


# Run training
train_C4T = False

if train_C4T:
    main()


# 7. Inference (If you want to encode data with a pretrained C4T or CLIP model you can start here)

# A. Load models

#C4T
C4T = C4TModel().to(CFG.device)
C4T.load_state_dict(torch.load(model_path))
C4T.eval()

#CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

#compare model parameters (trainable)
def get_params(model):
  print(sum(p.numel() for p in model.parameters()))

print(f'parameters C4T: {get_params(C4T)}')             #90,724,928
print(f'parameters CLIP: {get_params(clip_model)}')     #151,277,313

# B. Image Embeddings
'As we have a "database" of images we want the model to retrieve from, we can encode and embed them beforehand to gain efficieny'

def get_image_embeddings(valid_df, model_path):
  '''
  load the saved model, feed it images (validation set) and return the image_embeddings (shape[valid_set_size, 512]) and the model itself
  '''
  tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
  valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
  
  model = C4TModel().to(CFG.device)
  model.load_state_dict(torch.load(model_path, map_location=CFG.device))
  model.eval()
  
  valid_image_embeddings = []
  #valid_images = []
  with torch.no_grad():
      for batch in tqdm(valid_loader):
        image_features = model.image_encoder(batch["image"].to(CFG.device))
        image_embeddings = model.image_projection(image_features)
        valid_image_embeddings.append(image_embeddings)
        x = batch["image"]
        valid_images.append(x)
  
  #torch.save(valid_images, '/content/drive/MyDrive/Master/06_Data/A2D2/valid_images_A2D2_6E_512.npy')

  return model, torch.cat(valid_image_embeddings) #torch.cat: concatenates the given sequence in the given dimension

  
# now we build a validation-set and generate the corresponding image-embeddings
_, valid_df = make_train_valid_dfs()
captions = valid_df['caption'].values

model, image_embeddings = get_image_embeddings(valid_df, model_path)
# save image_embeddings
torch.save(image_embeddings, image_embeddings_C4T)


# calculate the same embeddings with CLIP
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_embeddings_clip = []

for image in tqdm(valid_df['image']):
  image_path = images_path + '/' + image
  img = Image.open(image_path)
  image_input = preprocess(img).unsqueeze(0).to(CFG.device)

  with torch.no_grad():
    image_embeddings = clip_model.encode_image(image_input.to(CFG.device))
  
  image_embeddings_clip.append(image_embeddings.detach().cpu().numpy())

image_embeddings_clip_np = np.array(image_embeddings_clip)

# save CLIP image embeddings
image_embeddings_clip = torch.tensor(image_embeddings_clip).squeeze()
torch.save(image_embeddings_clip, image_embeddings_CLIP)


# Load processed image-embeddings of both models for image-retrieval etc. 

#C4T
image_embeddings_C4T = torch.load(image_embeddings_C4T)
print(f'C4T embeddings: {image_embeddings.shape}')

#CLIP
image_embeddings_clip = torch.load(image_embeddings_CLIP)
image_embeddings_clip = image_embeddings_clip.to(CFG.device)
print(f'clip embeddings: {image_embeddings_clip.shape}')



# 8. Image-Retrieval: find matches
'After having a trained model (C4T or CLIP) we can now use text-queries and ask for matching images out of the "database" which in this case is the validation-set previously processed'

def find_matches(model, image_embeddings, query, captions_dataframe, plot, extended, dot_similarity_extended=False, perfect_matches_counter, n=9):
  '''
  use the given model to encode the given text-query and search for highest matches in the database (here: image-embeddings) based on the cosine similarity, return n matches
  compare ground-truth (caption in the validation-set) with the actual output and measure performance based on Precision & Recall
  '''
  #initialize lists for dataframe
  filenames = []
  cos_sims = []

  global text_embeddings
  #check wich model was parsed and encode the text with the corresponding tokenizer & encoder

  #C4T
  if model == C4T:
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
      text_features = model.text_encoder(
          input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
      )
      text_embeddings = model.text_projection(text_features)

  #CLIP
  else:
    encoded_query = clip.tokenize(query).to(CFG.device)
    with torch.no_grad():
      text_embeddings = model.encode_text(encoded_query).to(CFG.device)
  
  #process prepared image_embeddings (=database) and calculate the dot-product between text & image
  if model == C4T:
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.to(CFG.device).T
  else:
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1).half()
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.to(CFG.device).T

  values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
  # save top embeddings/images
  top_embeddings_list = []
  for i in indices: 
    top_embedding = image_embeddings_n[i]
    top_embeddings_list.append(top_embedding.detach().cpu().numpy())
  if model == C4T_model:
    top_embeddings = torch.tensor(top_embeddings_list).squeeze() 
    #torch.save(top_embeddings, 'PATH_TO_SAVE.npy')
    #print('Embeddings saved!')
  else:
    top_embeddings = torch.tensor(top_embeddings_list).squeeze() 
    #torch.save(top_embeddings, 'PATH_TO_SAVE.npy')
    #print('Embeddings saved!')

  # get corresponding image-filenames
  image_filenames = captions_dataframe['image'].values
  matches = [image_filenames[idx] for idx in indices[::5]]
  for match in matches:
    filenames.append(match)
  # get the cosine similarity values
  similarities, ids = torch.topk(dot_similarity.squeeze(0), n)
  sims = [x.numpy() for x in similarities.cpu()]
  for s in sims:
    cos_sims.append(s)

  # get 'ground-truth' captions of the images
  true_captions_list = list()
  def true_captions(index):
    'get image ground-truth (caption) for comparison'
    x = captions_dataframe.loc[captions_dataframe['image'] == str(matches[index]), 'caption'].iloc[0]
    x = '\n'.join(textwrap.wrap(x, (len(x)/2)))
    return x

  for i in range(n):
    true_captions_list.append(true_captions(i))

  # compare the caption with the query & give points for "correct" matches
  captions_score = []
  query_ = query.split(",")
  for true_caption in true_captions_list:
    for t in query_:
      y = re.findall(str(t.replace(",", "")), true_caption, flags=re.IGNORECASE) #source: https://medium.com/quantrium-tech/extracting-words-from-a-string-in-python-using-regex-dac4b385c1b8
      if len(y) >= 1:
        captions_score.append(1)
      else:
        captions_score.append(0)
  
  '''
    captions_score = []
    query_ = query.split(",")
    for true_caption in true_captions_list:
      y = all(elem in true_caption for elem in query) #source: https://medium.com/quantrium-tech/extracting-words-from-a-string-in-python-using-regex-dac4b385c1b8
      if y:
        captions_score.append(1)
      else:
        captions_score.append(0)
  '''

  '''
  # search for 'perfect' matches
  perfect_matches = []
  for true_caption in true_captions_list:
    if len(query_) > 1: 
      if query in true_caption:
        perfect_matches.append(1)

  if len(perfect_matches) >= 1: 
    print(f'Perfect Matches: {len(perfect_matches)}')
    if perfect_matches_counter == True:
      return len(perfect_matches)
  '''

  # store TPs & FPs 
  TPs = []
  FPs = []
  for x in captions_score:
    if x == int(1):
      TPs.append(int(1))
      FPs.append(int(0))
    elif x == int(0):
      TPs.append(int(0))
      FPs.append(int(1))

  # metrics

  # get TP for query on whole dataset
  true_positives = 0
  for i in captions_dataframe['caption']:
    z = re.findall(str(query), i, flags=re.IGNORECASE)
    if len(z) >= 1:
      true_positives += 1
  
  '''
  else:
    freq = 0
    x = query.split(',')
    for i in captions_dataframe['caption']:
      z = all(elem in i for elem in x)
      if z:
        freq +=1
  '''

  # get true positives, false positives and false negative
  tp = [x for x in captions_score if x is 1]
  tp = len(tp)

  fp = [x for x in captions_score if x is 0]
  fp = len(fp)

  fn = true_positives - tp

  '''
  if len(query.split(',')) == 1:
    fn = true_positives - tp  #true_positives checks the number of matches in the valid_df and tp the correct outputs (caution before interpreting)
  else:
    fn = abs(freq - tp)
    r = tp / (tp+fn)
    return r
  '''

  # calculate precision 
  precision = tp / (tp + fp)
  if plot == True:
    print(f'Precision = {precision: .4f}')
  
  # calculate recall (caution for multi-class queries)
  if (tp + fn) > 0:
    recall = tp / (tp + fn)
    return recall
  else:
    recall = 0
    return recall
    

  if plot == True: 
    # plot top 9 images
    _, axes = plt.subplots(3, 3, figsize=(20, 15))
    #plot cosine similarity values as titles and 'truth-caption'
    axes[0,0].set_title(f'similarity: {sims[0]:.8f}')
    axes[0,0].text(x=50, y=1100, s=true_captions(0), fontsize=11, color=(1,1,1)) #source: https://www.delftstack.com/howto/matplotlib/how-to-add-title-to-subplots-in-matplotlib/
    axes[0,1].set_title(f'similarity: {sims[1]:.8f}')
    axes[0,1].text(x=50, y=1100, s=true_captions(1), fontsize=11, color=(1,1,1))
    axes[0,2].set_title(f'similarity: {sims[2]:.8f}')
    axes[0,2].text(x=50, y=1100, s=true_captions(2), fontsize=11, color=(1,1,1))
    axes[1,0].set_title(f'similarity: {sims[3]:.8f}')
    axes[1,0].text(x=50, y=1100, s=true_captions(3), fontsize=11, color=(1,1,1))
    axes[1,1].set_title(f'similarity: {sims[4]:.8f}')
    axes[1,1].text(x=50, y=1100, s=true_captions(4), fontsize=11, color=(1,1,1))
    axes[1,2].set_title(f'similarity: {sims[5]:.8f}')
    axes[1,2].text(x=50, y=1100, s=true_captions(5), fontsize=11, color=(1,1,1))
    axes[2,0].set_title(f'similarity: {sims[6]:.8f}')
    axes[2,0].text(x=50, y=1100, s=true_captions(6), fontsize=11, color=(1,1,1))
    axes[2,1].set_title(f'similarity: {sims[7]:.8f}')
    axes[2,1].text(x=50, y=1100, s=true_captions(7), fontsize=11, color=(1,1,1))
    axes[2,2].set_title(f'similarity: {sims[8]:.8f}')
    axes[2,2].text(x=50, y=1100, s=true_captions(8), fontsize=11, color=(1,1,1))

    # #plot images
    for match, ax in zip(matches, axes.flatten()):
      image = cv2.imread(f"{CFG.image_path}/{match}")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      ax.imshow(image)
      ax.axis("off")

    if model == C4T:
      plt.suptitle(f'Query: {query} \n Correct matches: {captions_score.count(1)} \n Actual matches: {true_positives} \n (C4T model)', fontsize=16)
      plt.show()
    else:
      plt.suptitle(f'Query: {query} \n Correct matches: {captions_score.count(1)} \n Actual matches: {true_positives} \n (CLIP model)', fontsize=16)
      plt.show()

    # compare the similarity of the whole dataset with the query
    if dot_similarity_extended == True:
      plt.figure(figsize=(15,14))
      plt.grid()
      extended_values, extended_indices = torch.topk(dot_similarity.squeeze(0), len(captions_dataframe))

      #scatterplot
      plt.subplot(121)
      plt.grid()
      plt.scatter(x=range(len(captions_dataframe)), y=extended_values.cpu())
      plt.title(f'query: {query}')
      plt.ylabel('cosine similarity')
      plt.axhline(torch.mean(extended_values.cpu()), label=str(f'mean: {torch.mean(extended_values.cpu()):.6f}'), color='red')
      plt.legend(loc='upper right')

      #histogram
      plt.subplot(122)
      plt.grid()
      n, bins, pacthes = plt.hist(extended_values.cpu(), 20, density=True, facecolor='b', edgecolor='black', alpha=0.75)
      plt.title(f'query: {query}')
      plt.xlabel('cosine similarity')
      plt.ylabel('frequency')
      plt.axvline(torch.mean(extended_values.cpu()), label=str(f'mean: {torch.mean(extended_values.cpu()):.6f}'), color='red')
      plt.legend(loc='upper right')

      plt.show()
  

  if extended == True:
    return filenames, cos_sims, TPs, FPs

  
  if plot == False:
    return precision, recall


# Now we can test and plot some queries based on the A2D2-dataset (validation-set)
image_retrieval = False

if image_retrieval:
  # A: 1-class queries
  #C4T: use loaded C4T model and corresponding embeddings (see section 7)
  find_matches(model=C4T, image_embeddings=image_embeddings_C4T, query='Buildings', captions_dataframe=valid_df, plot=True, dot_similarity_extended=False, extended=False, perfect_matches_counter=False, n=9)
  #CLIP
  find_matches(model=clip_model, image_embeddings=image_embeddings_clip, query='Buildings', captions_dataframe=valid_df, plot=True, dot_similarity_extended=False, extended=False, perfect_matches_counter=False, n=9)

  # B: Multi-Class queries (caution with the Recall value)
  #C4T
  find_matches(model=C4T, image_embeddings=image_embeddings_C4T, query='Sidewalk, Pedestrian', captions_dataframe=valid_df, plot=True, dot_similarity_extended=False, extended=False, perfect_matches_counter=False, n=9)
  #CLIP
  find_matches(model=clip_model, image_embeddings=image_embeddings_clip, query='Sidewalk, Pedestrian', captions_dataframe=valid_df, plot=True, dot_similarity_extended=False, extended=False, perfect_matches_counter=False, n=9)

  # C: Other queries (no performance measuring due to missing ground-truth)
  #C4T
  find_matches(model=C4T, image_embeddings=image_embeddings_C4T, query='Intersection', captions_dataframe=valid_df, plot=True, dot_similarity_extended=False, extended=False, perfect_matches_counter=False, n=9)
  #CLIP
  find_matches(model=clip_model, image_embeddings=image_embeddings_clip, query='Intersection', captions_dataframe=valid_df, plot=True, dot_similarity_extended=False, extended=False, perfect_matches_counter=False, n=9)


# 9. Evaluation
'''
Most image-retrieval (CBIR) systems dont enable a standardized method to evaluate outputs on a given ground-truth or compare them with other models
As we aim to "beat" CLIP for traffic-data, we will compare our outputs against the ones from CLIP
Note: This is only possible if the validation-set has a ground-truth (caption) for each image to measure against and also the queries has to be according to this ground-truth
'''

# First we define some queries to test based on the validation-set
automotive_query_single = ['Buildings', 'Sidewalk', 'Nature object', 'Car', 'Parking area', 'Drivable cobblestone', 'Slow drive area', 'Poles', 'Non-drivable street', 'Curbstone', 'Solid line', 
                           'Dashed line', 'Truck', 'Traffic sign', 'Road blocks', 'Bicycle', 'Utility vehicle', 'Pedestrian', 'Signal corpus', 'Zebra crossing', 'Sidebars', 'Small vehicles']

automotive_query_single_small = ['Traffic sign', 'Road blocks', 'Traffic guide obj.', 'Bicycle', 'Painted driv. instr.', 'Utility vehicle', 'Pedestrian', 'Signal corpus', 'Zebra crossing', 'Sidebars', 
                                 'Small vehicles']

automotive_query_multi = ['Car, Truck', 'Car, Bicycle', 'Car, Pedestrian', 'Pedestrian, Bicycle', 'Buildings, Traffic sign', 'Car, Parking area', 'Pedestrian, Solid line', 'Buildings, Pedestrian', 
                          'Bicycle, Traffic sign', 'Truck, Dashed line', 'Slow drive area, Pedestrian', 'Sidewalk, Car', 'Zebra crossing, Car, Pedestrian', 'Nature object, Drivable cobblestone',
                          'Sidewalk, Drivable cobblestone', 'Car, Poles, Bicycle', 'Truck, Traffic sign', 'Car, Road blocks, Pedestrian', 'Utility vehicle, Pedestrian']

automotive_query_random = ['Car', 'Truck', 'Bicycle', 'Traffic sign', 'Drivable cobblestone', 'Poles', 'Curbstone', 'Solid line', 'Dashed line', 'Pedestrian', 'Zebra crossing', 'Parking area', 
                           'Traffic signal', 'Car, Truck', 'Car, Traffic sign', 'Pedestrian, Car', 'Traffic sign, Bicycle', 'Pedestrian, Solid line', 'Car, Parking area', 
                           'Buildings, Pedestrian, Traffic signal']


# Check the frequency of occurence per query to make sure they are represented 
def frequency(query_list, dataframe): #single classes
    'given a query list and the validation-set to search for, outputs the frequency of occurence'
  freq_list = []
  query = []
  for a in query_list:
    frequency = 0
    query.append(a)
    for c in dataframe['caption']:
      x = re.findall(str(a), c, flags=re.IGNORECASE)
      if len(x) >= 1:
        frequency += 1
    freq_list.append(frequency)
          
  df = pd.DataFrame(np.column_stack([query, freq_list]), columns=['query', 'freq(s)'])
  return df

check_frequency = False
if check_frequency:
  frequency(automotive_query_single, valid_df)

# For multi-class queries a different approach is faster
def frequency_multi(query_list, dataframe): #multiple classes
    'given a query list and the validation-set to search for, outputs the frequency of occurence'
  freq_l = []
  query_l = []
  for a in query_list:
    freq = 0
    x = a.split(', ')
    query_l.append(x)
    for c in dataframe['caption']:
      result = all(elem in c for elem in x)
      if result:
        freq += 1
    freq_l.append(freq)

    print(x, freq)

check_frequency_multi = False
if check_frequency_multi:
  frequency_multi(automotive_query_multi, valid_df)


# Calculate Precision (& Recall) for given query_list for C4T and CLIP
def average_precision_recall(query_list, n):
  C4T_precision = []
  C4T_recall = []

  clip_precision = []
  clip_recall = []

  # calculate precision
  for q in tqdm(query_list):
    'C4T'
    precision_C4T, recall_C4T = find_matches(model=C4T_model, image_embeddings=image_embeddings_C4T, query=str(q), captions_dataframe=valid_df, plot=False, extended=False, perfect_matches_counter=False, n=n)
    C4T_precision.append(precision_C4T)
    C4T_recall.append(recall_C4T)

    'CLIP'
    precision_clip, recall_clip = find_matches(model=clip_model, image_embeddings=image_embeddings_clip, query=str(q), captions_dataframe=valid_df, plot=False, extended=False, perfect_matches_counter=False, n=n)
    clip_precision.append(precision_clip)
    clip_recall.append(recall_clip)
  
  # calculate mean precision
  C4T_precision_mean = np.mean(C4T_precision)
  print(f'Precision C4T Model at n={n}: {C4T_precision_mean: .5f}')

  clip_precision_mean = np.mean(clip_precision)
  print(f'Precision CLIP Model at n={n}: {clip_precision_mean: .5f}')

  # calculate precision difference (performance)
  difference = ((C4T_precision_mean - clip_precision_mean)/C4T_precision_mean)*100
  print(f'Precision difference (%): {difference: .2f}')

  # calculate mean recall
  C4T_recall_mean = np.mean(C4T_recall)
  print(f'Recall C4T Model at n={n}: {C4T_recall_mean: .5f}')

  clip_recall_mean = np.mean(clip_recall)
  print(f'Recall CLIP Model at n={n}: {clip_recall_mean: .5f}')

# conduct performance measurement
measure_performance = False

if measure_performance: # calculate for each query-category seperately for n outputs
  n = [10, 25, 50, 100, 250]

for n_ in n:
  print('###SINGLE###')
  average_precision_recall(automotive_query_single, n_)
for n_ in n:
  print('###SMALL###')
  average_precision_recall(automotive_query_single_small, n_)
for n_ in n:
  print('###MULTI###')
  average_precision_recall(automotive_query_multi, n_)


# 10. Zero-Shot Experiment
'''
Lastly, we want to measure C4Ts zero-shot abilities on a different/unknown traffic-dataset as the prepared TJU-DHD and compare its performance to CLIP.
The images and captions were generated in the folder "Data" --> "TJU-DHD.py" with examples under "TJU-DHD_preprocessed". 
'''

# First we create a new validation dataframe
valid_df_TJU = make_train_valid_dfs() #important: change captions path inside the function!

# Now we get the queries based on the top n ocurrences
n = 15

occurances = pd.DataFrame(valid_df_TJU['caption'].value_counts()[:n].tolist()).rename({0: 'Frequency'}, axis=1)
top_captions = pd.DataFrame(valid_df_TJU['caption'].value_counts()[:n].index.tolist()).rename({0: 'Caption'}, axis=1)
# some preprocessing
top_captions['Caption'] = top_captions['Caption'].apply(lambda x: x.replace('There is', ''))
top_captions['Caption'] = top_captions['Caption'].apply(lambda x: x.strip())

frequent_captions_TJU = pd.concat([occurances, top_captions], axis=1)
#print(frequent_captions_TJU.head(n))

# Now we calculate the Precision & Recall based on 9. 
TJU_queries = frequent_captions_TJU['Caption'].tolist()

for n_ in n:
  average_precision_recall(TJU_queries, n_)