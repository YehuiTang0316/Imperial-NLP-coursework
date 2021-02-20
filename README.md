# Imperial-NLP-coursework
Task: to predict how funny the edited headline is
## Preparation
### Load data
To obtain data frame for training, validation and test, please first use pandas to load dataset ``` df = pd.read_csv('./dataset/xxx.csv') ``` .

### Replace original headline with editted headline
Use ``` get_edited_df(df)``` to replace the headline.

## Part 1
### Experiment 1

### Experiment 2
In the second experiment we use pretrained bert to do the regression.
#### Generate Dataloader
use the function
```
create_dataloader(df)
```
to generate dataset for train, val and test.

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
