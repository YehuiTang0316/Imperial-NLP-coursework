# Imperial-NLP-coursework
Task: to predict how funny the edited headline is
## Preparation
### Load data
To obtain data frame for training, validation and test, please first use pandas to load dataset ``` df = pd.read_csv('./dataset/xxx.csv') ``` .

### Replace original headline with editted headline
Use ``` get_edited_df(df)``` to replace the headline.

## Part 1
### Experiment 1
In the first experiment we introduced attention mechanism into the baseline BiLSTM model to do the task.
#### Train and Test
To train the model, use
```
train(train_loader, dev_loader, model, epochs, 'attn')
```
To see how the model performs at test set, run
```
test_loss, test_mse, __, __ = eval(test_loader, model, 'attn')
print(test_loss, test_mse)
```

### Experiment 2
In the second experiment we use pretrained bert to do the regression.
#### Generate Dataloader
use the function ``` create_dataloader(df) ``` to generate dataset for train, val and test.

#### Train and Test
To train the model, use
```
train(train_loader, dev_loader, model, epochs, 'bert')
```
To see how the model performs at test set, run
```
test_loss, test_mse, __, __ = eval(test_loader, model, 'bert')
print(test_loss, test_mse)
```


## Part 2
To train your own word embedding, follow the data pre-processing steps under section 2-1, or load the embedding from our trained model
```
# Load the embedding
PATH = './w2v_origin.pt'
model = SkipGram(100, len(vocab), device, negative_sampling=True, noise_dist=noise_dist, negative_samples=15).to(device)
model.load_state_dict(torch.load(PATH))
embedding = model.embed_input.weight.data

# Load the bi-GRU model
PATH = './results/bigru-1.pt'
gru_model = GRU(EMBEDDING_DIM, HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, device)
gru_model.load_state_dict(torch.load(PATH))
```
To train the model, use
```
gru_model, train_loader, dev_loader = create_gru_dataloader(training_x, dev_x, training_y, dev_y, is_train, w2i, EMBEDDINGS_1, hidden_dim, batch_size, device)
train(train_loader, dev_loader, gru_model, epochs)
```
To test the model, use
```
# Get required dictionary
w2i = load_dict('./results/w2i-1.pkl')

# Create dataloader
test_loader = create_gru_dataloader(None, test_df['edited'], None, test_df['meanGrade'], train=False, word2idx=w2i, batch_size=128)

# Testing
test_loss, test_mse, __, __ = eval(test_loader, gru_model)
```
