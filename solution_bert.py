
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric

def contextualized_word_embeddings(model_name = "nlptown/bert-base-multilingual-uncased-sentiment"):
  raw_datasets = load_dataset("amazon_reviews_multi", "de")
  checkpoint = model_name
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)

  def tokenize_function(example):
      return tokenizer(example["review_body"], truncation=True)

  tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  tokenized_datasets = tokenized_datasets.remove_columns(['review_id', 'review_body', 'product_id','reviewer_id','review_title', 'language', 'product_category'])
  tokenized_datasets = tokenized_datasets.rename_column("stars", "labels")
  tokenized_datasets.set_format("torch")
  #tokenized_datasets["train"].column_names

  def sub_one_label(example):
      return {"labels": example['labels'].item() - 1} # 5 classes (0 to 4)

  tokenized_datasets['validation'] = tokenized_datasets['validation'].map(sub_one_label)
  tokenized_datasets['train'] = tokenized_datasets['train'].map(sub_one_label)
  tokenized_datasets['test'] = tokenized_datasets['test'].map(sub_one_label)

  # data loader #
  train_dataloader = DataLoader(tokenized_datasets["train"].shuffle().select(range(5000)) , shuffle=True, batch_size=8, collate_fn=data_collator)
  eval_dataloader = DataLoader(tokenized_datasets["validation"].shuffle().select(range(500)), batch_size=8, collate_fn=data_collator)
  test_dataloader = DataLoader(tokenized_datasets["test"].shuffle().select(range(500)), batch_size=8, collate_fn=data_collator)


  # model #
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)
  optimizer = AdamW(model.parameters(), lr=5e-5)

  num_epochs = 5
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
  print("Number of training steps : ",num_training_steps)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)
  print("Device : ",device)


  metric_acc = load_metric("accuracy")
  metric_precision = load_metric("precision")

  progress_bar = tqdm(range(num_training_steps))
  for epoch in range(num_epochs):
    model.train()
    print('Epoch : ',epoch+1)
    
    train_accuracy = 0
    train_precision = 0
    totloss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        # Accuracy and f1 compytation
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        performance_acc = metric_acc.compute(predictions=predictions, references=batch["labels"])
        train_accuracy += performance_acc['accuracy']
        performance_precision = metric_precision.compute(predictions=predictions, references=batch["labels"], average= 'micro')
        train_precision += performance_precision['precision']

        loss = outputs.loss
        loss.backward()
        totloss += loss.item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    print("Train: Total loss: ",totloss/len(train_dataloader))
    print("Train: Total accuracy : ", train_accuracy/len(train_dataloader))
    print("Train: Total precision : ", train_precision/len(train_dataloader))

    # validation #
    model.eval()
    val_accuracy = 0
    val_precision = 0
    val_totloss = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        performance_acc = metric_acc.compute(predictions=predictions, references=batch["labels"])
        val_accuracy += performance_acc['accuracy']
        performance_precision = metric_precision.compute(predictions=predictions, references=batch["labels"], average= 'micro')
        val_precision += performance_precision['precision']

        loss = outputs.loss
        val_totloss += loss.item()

    print("Validation: Total loss: ",val_totloss/len(eval_dataloader))
    print("Validation: Total accuracy : ", val_accuracy/len(eval_dataloader))
    print("Validation: Total precision : ", val_precision/len(eval_dataloader))


  ## Test ##
  print("_____________________________________________________________________________")
  model.eval()
  test_accuracy = 0
  test_precision = 0
  test_totloss = 0
  for batch in test_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      performance_acc = metric_acc.compute(predictions=predictions, references=batch["labels"])
      test_accuracy += performance_acc['accuracy']
      performance_precision = metric_precision.compute(predictions=predictions, references=batch["labels"], average= 'micro')
      test_precision += performance_precision['precision']

      loss = outputs.loss
      test_totloss += loss.item()

  print("Test: Total loss: ",test_totloss/len(test_dataloader))
  print("Test: Total accuracy : ", test_accuracy/len(test_dataloader))
  print("Test: Total precision : ", test_precision/len(test_dataloader))