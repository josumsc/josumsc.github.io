---
layout: post
title: "Fine-tuning HuggingFace Transformers for Text Classification"
date: 2023-03-05 00:00:00-0100
description: "Creating your own text classification API using HuggingFace Transformers and Flask"
tags: 
  - nlp
  - python
  - mlops
  - pytorch
  - deep learning
giscus_comments: true
---

{% include figure.html path="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" class="img-fluid rounded z-depth-1" %}

> The full implementation of the project described in this post can be found on [GitHub](https://github.com/josumsc/fake-news-detector)

As [Bing and Google fight for delivering the best AI based search solution](https://www.technologyreview.com/2023/02/16/1068695/chatgpt-chatbot-battle-search-microsoft-bing-google/), Large Language Models popularity keeps increasing. We, mere mortals far from the computing and data repositories available to the big tech companies, can also take advantage of these models to solve our own problems. In this post, we will see how to fine-tune a HuggingFace Transformer model to leverage the work of those giants and create our own text classification model, with SOTA results.

Furthermore, we will use Flask to create an API that serves our model predictions, and implement MLOps best practices to deploy our model in production and ensure a CD flow.

## Fine-Tuning the Model

### The Transformers ecosystem

HuggingFace is the go-to company for Natural Language Processing (NLP) tasks. They have a wide range of models, datasets, and tools to help you solve your NLP problems. They also have a great community that is always willing to help. They started their journey with the [Transformers library](https://huggingface.co/docs/transformers/index) but nowadays they also provide a [Hub to download pretrained models and tokenizers](https://huggingface.co/models), a [Datasets library](https://huggingface.co/datasets) to download and process datasets, and [access to spaces to train your model on them and show demos of their capabilities](https://huggingface.co/spaces).

In this post, we will use the `datasets` library to download a dataset that fits our needs, the `transformers` library to download a pretrained LLM, and the Hub to upload our model and tokenizer to be able to use them in our API.

To work with the HuggingFace ecosystem and combine it with PyTorch, we have defined a `DetectorPipeline` class that will serve as an interface to interact with our model pipeline:

```python
class DetectorPipeline:
    def __init__(
        self,
        dataset_name: str = "GonzaloA/fake_news",
        checkpoint: str = "distilbert-base-uncased-finetuned-sst-2-english",
        model_name: str = "fake_news_detector",
    ):
        """Detector pipeline class.

        :param dataset_name: Name of the dataset to download, defaults to "GonzaloA/fake_news"
        :type dataset_name: str, optional
        :param checkpoint: Name of the model to fine-tune, defaults to "distilbert-base-uncased-finetuned-sst-2-english"
        :type checkpoint: str, optional
        :param model_name: Name of the model to save, defaults to "fake_news_detector"
        :type model_name: str, optional
        """
        self.dataset_name = dataset_name
        self.checkpoint = checkpoint
        self.model_name = model_name
```

The functions shown later on this part of the post will be defined as methods of this class.

### The Dataset

We will use the [Fake News Dataset](https://huggingface.co/datasets/GonzaloA/fake_news) that includes approximately 40k news articles with their header and corpus labeled as either fake or real. The dataset is already split into train, validation, and test sets, so we can download it directly from the Hub with those splits already specified.

```python
def download_dataset(self) -> datasets.DatasetDict:
    """Download dataset from HuggingFace datasets library.

    :return: DatasetDict object with the train, validation and test splits.
    :rtype: datasets.DatasetDict
    """
    dataset = datasets.load_dataset(self.dataset_name)
    return dataset
```

### The Model

With the dataset on memory, the next step would be to download the model. In this case, we will be using the `distilbert-base-uncased-finetuned-sst-2-english` model, which is a fine-tuned version of the `distilbert-base-uncased` model for the SST-2 dataset. The SST-2 dataset is a binary classification dataset with the goal of predicting whether a sentence is positive or negative. The model was trained on the SST-2 dataset and then fine-tuned on the `fake_news` dataset.

```python
def get_tokenizer_and_model(
    self, checkpoint: str = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Get tokenizer and model from model name.

    :param checkpoint: Name of the model to fine-tune, defaults to None
    :type checkpoint: str, optional
    :return: Tokenizer and Model objects.
    :rtype: (AutoTokenizer, AutoModelForSequenceClassification)
    """
    model_name = checkpoint if checkpoint else self.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    return tokenizer, model
```

### The DataLoaders

Text datasets can be huge, and even though we were able to load them in memory it doesn't mean that we will be able to load them into our GPU all at once, given that our model and logits also have to coexist on that instance. To solve this problem, we will use PyTorch's `DataLoader` class to load the data in batches, passing the text by the tokenizer first to receive them as integers.

But first, we need to define a [DataCollator](https://huggingface.co/docs/transformers/main_classes/data_collator) that will be in charge of padding the sequences to the same length, which in this case will be the longer sequence. We will use the `DataCollatorWithPadding` class for this task.

Also, please note that some preprocessing is done on the dataset to remove the string columns after the tokenization process and to rename the target column to `labels` to match the model's expected input.

```python
def get_data_collator(self, tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    """Get data collator from tokenizer.

    :param tokenizer: Tokenizer object.
    :type tokenizer: AutoTokenizer
    :return: Data collator object.
    :rtype: DataCollatorWithPadding
    """

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator

def get_dataloaders(
    self,
    dataset: datasets.DatasetDict,
    batch_size: int,
    tokenizer: AutoTokenizer,
    data_collator: DataCollatorWithPadding,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get the dataloaders for train, validation and test splits.

    :param dataset: Train dataset.
    :type dataset: datasets.DatasetDict
    :param batch_size: Batch size.
    :type batch_size: int
    :param tokenizer: Tokenizer object.
    :type DataLoader: AutoTokenizer
    :param data_collator: Data collator object.
    :type data_collator: DataCollatorWithPadding
    :return: Dataloaders for train, validation and test splits.
    :rtype: (DataLoader, DataLoader, DataLoader)
    """

    # Tokenize dataset
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Put in format that the model expects
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["Unnamed: 0", "title", "text"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader, test_dataloader
```

### The Trainer

Once we have the dataset, the model, and the dataloaders, we can start training the model. To do this, we will use PyTorch interfaces to define the training loop and the evaluation loop:

```python
def train_model(
    self,
    model: AutoModelForSequenceClassification,
    train_dataloader: DataLoader,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
) -> AutoModelForSequenceClassification:
    """Train model.

    :param model: Model to train.
    :type model: AutoModelForSequenceClassification
    :param train_dataloader: Dataloader with train data.
    :type train_dataloader: DataLoader
    :param epochs: Number of epochs to train, defaults to 3
    :type epochs: int, optional
    :param lr: Learning rate, defaults to 2e-5
    :type lr: float, optional
    :param weight_decay: Weight decay, defaults to 0.0
    :type weight_decay: float, optional
    :param warmup_steps: Number of warmup steps, defaults to 0
    :type warmup_steps: int, optional
    :param max_grad_norm: Maximum gradient norm, defaults to 1.0
    :type max_grad_norm: float, optional
    :return: Trained model.
    :rtype: AutoModelForSequenceClassification
    """

    num_training_steps = len(train_dataloader) * epochs

    # Set device
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Train
    pbar = tqdm(range(num_training_steps))
    with tqdm(total=num_training_steps) as pbar:
        for _ in range(epochs):
            model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)

    return model

def evaluate_model(
    self,
    dataset: datasets.Dataset,
    dataloader: DataLoader,
    model: AutoModelForSequenceClassification,
) -> float:
    """Evaluate model.

    :param eval_dataloader: dataset to evaluate.
    :type eval_dataloader: DatasetDict
    :param dataset: Dataloader with eval data.
    :type dataset: DataLoader
    :param model: Model to evaluate.
    :type model: AutoModelForSequenceClassification
    :return: Accuracy.
    :rtype: float
    """

    # Set device
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    # Evaluate
    model.eval()
    predictions = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        running_predictions = (
            torch.argmax(logits, dim=-1).to("cpu").numpy().tolist()
        )
        predictions.extend(running_predictions)

    # Print results
    print("Results of the model:\n")
    f1score = f1_score(dataset["validation"]["label"], predictions, average="macro")
    print(f"F1 score: {f1score}")
    print(classification_report(dataset["validation"]["label"], predictions))
    print(confusion_matrix(dataset["validation"]["label"], predictions))

    return f1score
```

### Putting everything together

So far we have the different methods that should be called from the pipeline, although we still lack the pipeline itself. The pipeline will be the one that will call the different methods:

```python
def train_pipeline(
    self, epochs: int = 3, lr: float = 2e-5, batch_size: int = 16
) -> AutoModelForSequenceClassification:
    """Runs the train pipeline and returns the trained model.
    :param epochs: Number of epochs to train, defaults to 3
    :type epochs: int, optional
    :param lr: Learning rate, defaults to 2e-5
    :type lr: float, optional
    :param batch_size: Batch size, defaults to 16
    :type batch_size: int, optional
    :return: Trained model.
    :rtype: AutoModelForSequenceClassification
    """
    dataset = self.download_dataset()
    tokenizer, model = self.get_tokenizer_and_model()
    data_collator = self.get_data_collator(tokenizer)
    train_dataloader, eval_dataloader, _ = self.get_dataloaders(
        dataset,
        batch_size=batch_size,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model = self.train_model(model, train_dataloader, epochs=epochs, lr=lr)
    self.evaluate_model(dataset, eval_dataloader, model)
    model.save_pretrained(os.path.join("models", self.model_name))
    tokenizer.save_pretrained(os.path.join("models", self.model_name))

    return
```

Notice how the pipeline ends by calling the `save_pretrained` method of the model and tokenizer. This will save the model and tokenizer in the `models` directory with the name defined in the class instantiation, so that we can use them later.

After training, we will need to be able to predict the results of a given text. For this procedure, we will need to load the model and tokenizer that we have previously saved and create a `predict` function:

```python
def load_model_from_directory(
    self, model_name: str = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load model from directory.

    :param model_name: Name of the model to load, if None, self.model_name is used, defaults to None.
    :type model_name: str
    :return: Loaded tokenizer and model.
    :rtype: (AutoTokenizer, AutoModelForSequenceClassification
    """
    load_path = (
        os.path.join("models", model_name)
        if model_name
        else os.path.join("models", self.model_name)
    )
    tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path, local_files_only=True
    )
    return tokenizer, model

def predict(
    self,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    text: str,
) -> int:
    """Predict class of text.

    :param tokenizer: Tokenizer to use for prediction.
    :type tokenizer: AutoTokenizer
    :param model: Model to use for prediction.
    :type model: AutoModelForSequenceClassification
    :param text: Text to predict.
    :type text: str
    :return: Predicted class.
    :rtype: int
    """
    # Set device
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    # Predict
    model.eval()
    encoded_text = tokenizer(
        text, truncation=True, padding=True, return_tensors="pt"
    )
    encoded_text = {k: v.to(device) for k, v in encoded_text.items()}
    with torch.no_grad():
        outputs = model(**encoded_text)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).to("cpu").numpy().tolist()[0]
    return prediction, logits
```

Finally, once we have the model and tokenizer loaded, we can publish them into HuggingFace Hub. For this, we will need to create another function in our pipeline:

```python
def publish_model_from_directory(self, model_name: str = None) -> None:
    """Publish model to Hugging Face Hub from the specified directory.
    Both the model in the directory and the model on the Hub must have the same name.
    :param model_name: Name of the model to publish, if None, self.model_name is used, defaults to None.
    :type model_name: str
    :return: None
    :rtype: None
    """
    model_name = model_name if model_name else self.model_name
    tokenizer, model = self.load_model_from_directory(model_name=model_name)
    tokenizer.push_to_hub(model_name)
    model.push_to_hub(model_name)
    return None
```

We have to consider that in order to publish models to the Hub we need to be logged in. For this, we can use the `huggingface-cli login` command (more info to be found in [their documentation](https://huggingface.co/docs/huggingface_hub/quick-start#login)).

## Creating a CLI to interact with the model

Even though we have simplified the process of loading, training and predicting using our pipeline, we would still need to import the class and call the different methods every time we would want to use it. For this reason, we will create a CLI that will allow us to interact with the model without having to write any unnecessary code, increasing our efficiency and avoiding errors.

We will be using the `click` library to create the CLI. The first step is to create a `cli.py` file that will contain the CLI:

```python
# Description: Command line interface for the fake news detector.
import click


@click.group()
def cli():
    pass

if __name__ == "__main__":
    cli() 
```

This will create the command line, but for now we have nothing to do with it. We need to add the different methods to the CLI by using the `@click.command()` decorator and the `cli.add_command()` method. We will start by adding the `train` command:

```python
from detector import DetectorPipeline

@click.command(name="train")
@click.option(
    "--dataset",
    default="GonzaloA/fake_news",
    help="Dataset to download from HuggingFace datasets library.",
)
@click.option(
    "--checkpoint",
    default="distilbert-base-uncased-finetuned-sst-2-english",
    help="Model to fine-tune.",
)
@click.option(
    "--output", default="fake-news-detector", help="Name of the model to save."
)
@click.option("--lr", default=2e-5, help="Learning rate.")
@click.option("--batch_size", default=16, help="Batch size.")
@click.option("--epochs", default=3, help="Number of epochs.")
def train(
    dataset: str, checkpoint: str, output: str, lr: float, batch_size: int, epochs: int
):
    pipeline = DetectorPipeline(
        dataset_name=dataset, checkpoint=checkpoint, model_name=output
    )
    pipeline.train_pipeline(epochs=epochs, lr=lr, batch_size=batch_size)
    click.echo(f"Model saved into directory ./models/{output}.")
```

By looking careful we can see how `click` deals with CLI arguments, which are passed to the function as parameters. This syntax based on decorators can be seen weird in the beginning but help us abstract those functionalities from the function to be defined. We can also appreciate the `click.echo` method, that will inform us of the finishing of the training process and the saving directory.

Finally, we can declare the `predict` and `publish` commands and calling the `cli.add_command()` method to add these new functions to the CLI:

```python
@click.command(name="predict")
@click.option(
    "--model", default="fake-news-detector", help="Model to use for prediction."
)
@click.option(
    "--checkpoint",
    default="distilbert-base-uncased-finetuned-sst-2-english",
    help="Tokenizer used to tokenize the text.",
)
@click.option("--text", default="This is a fake news", help="Text to predict.")
def predict(model: str, checkpoint: str, text: str):
    pipeline = DetectorPipeline(model_name=model, checkpoint=checkpoint)
    tokenizer, model = pipeline.load_model_from_directory()
    prediction, logits = pipeline.predict(tokenizer, model, text)
    click.echo(f"Prediction: {prediction}")
    click.echo(f"Logits: {logits}")

@click.command(name="publish")
@click.option("--model", default="fake-news-detector", help="Model to publish.")
def publish(model: str):
    pipeline = DetectorPipeline(model_name=model)
    pipeline.publish_model_from_directory()
    click.echo("Model published into HuggingFace Hub.")

cli.add_command(train)
cli.add_command(predict)
cli.add_command(publish)
```

Now whenever we want to interact with the model, we can simply run the `cli.py` file and use the different commands:

```bash
python cli.py train --epochs 1
python cli.py predict --text "This is a fake news"
python cli.py publish
```

## Creating a REST API

Having a CLI tool is great to work on your server or local machine, but it is not very convenient to use it in a production environment. Web or app teams should need to SSH to the server to run the commands, which will add a lot of complexity and delays in the communication, so it's not a proper solution to serve predictions to the final user.

For this reason, we will create a REST API that will allow us to interact with the model using HTTP requests. We will be using the `flask` library to create the API. The first step is to create a `app.py` file that will contain the code needed to run the server:

```python
from flask import Flask, render_template, request, jsonify
from detector import DetectorPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

This will create a simple Flask server on the port 5000 open for traffic access from every IP (*care with this in a real production environment!*) that will serve the `index.html` file when we access the root of the server. We will create this file in the `templates` folder:

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Fake news detector</title>
    </head>
    <body>
        <h1>Powered by HuggingFace and PyTorch</h1>
        <p>This app is conceived to be used as an API REST, although this particular endpoint serves as an entrypoint where we can test the functionalities using a form.</p>
        <form name="input" action="/detect_html" method="post">
            <label for="text">Text to detect:</label>
            <input type="text" id="text" name="text" value="The president of the United States is Donald Trump.">
            <input type="submit" value="Submit">
        </form>
        {% if result %}
            <p>Result: {{result}}</p>
            <p>Logits: {{logits}}</p>
        {% endif %}
    </body>
</html>
```

The main functionality of our app will be to be used in a simulated production environment by serving HTTP requests, but it's a good practice to serve also an index at route `/` to verify the correct functioning of the server. Furthermore, in this example we can use Jinja templates to pass parameters to the HTML file and render them in the browser. These parameters will be the result of the prediction and the logits, which will be passed to the template using the following function on the `app.py` file:

```python
@app.route("/detect_html", methods=["POST"])
def detect_html():
    if request.method == "POST":
        text = request.form["text"]
        result, logits = inference(text)
        return render_template("index.html", result=result, logits=logits)
```

{% include figure.html path="https://github.com/josumsc/fake-news-detector/blob/master/docs/img/html-interface.png?raw=true" class="img-fluid rounded z-depth-1" %}

Now that we have dealt with the HTML part of the server, we can create the endpoint that will be used to serve the predictions. We will create a `detect_json` function that will be called when we send a POST request to the `/detect_json` endpoint:

```python
@app.route("/detect_json", methods=["POST"])
def detect_json():
    if request.method == "POST":
        text = request.json["text"]
        result, logits = inference(text)
        return jsonify(result=result, logits=logits)
```

{% include figure.html path="https://github.com/josumsc/fake-news-detector/blob/master/docs/img/api-rest.png?raw=true" description="Detect JSON endpoint" class="img-fluid rounded z-depth-1" %}

To run our server and test the different endpoints, we can simply run the `app.py` file:

```bash
python app.py
```

## Deploying the model to production

We now have a proper API to serve predictions and a CLI command to train and publish the model, great! But, let's imagine for a moment that our production team decides to change the server where the model is running. They would need to configure everything to run the flask API again, which would imply some downtime and a lot of work. Even if we prepare for this, sometimes a small change in the libraries installed in the server could mean a total failure of the API. For this reason, it is very important to have a proper deployment strategy to avoid these problems.

The deployment strategy we will follow is based on using **Docker** as a containerization tool to create a container with all the dependencies and the code needed to run the API.

> In a real production environment, we could also add this container to a tool like **Kubernetes** to ensure high availability and scalability of the API, but as our API is very simple, we will not add this complexity to the example.

To create this container, we will create a `Dockerfile` file that will contain the instructions to build the image:

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY src/ .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]
```

We can see how the `python:3.9` image is downloaded from Docker Hub and used as a base image to build our own image. Then, we copy the `requirements.txt` file and the `src` folder to the container and install the dependencies. Finally, we expose the port 5000 and set the `app.py` file as the entrypoint of the container. To build the image we can run the following command:

```bash
docker build -t fake-news-detector .
```

Once our container is built we can share it with our production team and they will be able to run it without any problem. In the example of Docker Hub, publishing the container is as simple as running the following command:

```bash
docker push fake-news-detector
```

Now we can run the container in our production server using the following command:

```bash
docker run -p 5001:5000 fake-news-detector
```

Or even better, use `docker-compose.yml` file to run the container with the rest of the services in case extra dependencies are added:

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "5001:5000"
    restart:
      always
```

This way we can run and stop the container with the following commands:

```bash
docker-compose -f docker-compose.yml up -d --remove-orphans --build --force-recreate
docker-compose -f docker-compose.yml down
```

## Extra: Makefile

We have everything we need now to train our model, create our API and share it with the world without thinking about compatibilities or requirements, and that's a really good job done! But, we have skipped some important nuances of the MLOps cycle, such as linting the code or formatting, and we have left some manual steps that could be automated. For this reason, we will create a `Makefile` file to automate some of these tasks:

```makefile
# Makefile
DOCKER_USERNAME = josumsc

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black src/*.py

lint:
	pylint --disable=R,C,W1203,E1101 src/.*
	docker run --rm -i hadolint/hadolint < Dockerfile

publish:
	python src/cli.py publish
	docker build -t $(DOCKER_USERNAME)/flask-fake-news:latest .
	docker push $(DOCKER_USERNAME)/flask-fake-news:latest

run:
	docker-compose -f docker-compose.yml up -d --remove-orphans --build --force-recreate
	@echo "App deployed at http://localhost:5001"

stop:
	docker-compose -f docker-compose.yml down
```

We can see how we have added some commands to install the dependencies, format the code, lint the code and Dockerfile, publish the model and run the container. We can run these commands with the following command:

```bash
make <command>
```

## Conclusion

By following the steps in this article, we have created a simple API to serve predictions and a CLI command to train and publish the model. We have also created a Dockerfile to build a container that automatically serves our API. Finally, we have created a Makefile to automate some of the tasks and make the development process easier.

NLP and MLOps are 2 concepts that the modern Machine Learning Engineer should master to make the most of machine learning applications. Fortunately, the most groundbreaking tools are also open source, so we can read their documentation and use them to create powerful applications as the one shown in this article.

I hope that this was useful to some of you, have fun and happy coding!

> As next step, we could create a GitHub Actions pipeline to automatically publish the model in Docker whenever we push to master, but as we have seen, the process is very simple and we can do it manually without any problem. In case your project expects to have a long lifetime, please do consider adding a CI/CD pipeline to automate the process by [following the steps here](https://docs.github.com/en/actions/publishing-packages/publishing-docker-images#publishing-images-to-docker-hub).

## References

- [HuggingFace Transformers](https://huggingface.co/transformers/index.html)
- [Natural Language Processing with Transformers Revised Edition](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
- [Deploy Machine Learning Models to Production: With Flask, Streamlit, Docker, and Kubernetes on Google Cloud Platform](https://learning.oreilly.com/library/view/deploy-machine-learning/9781484265468/)
- [Open Source Platforms for MLOps](https://www.coursera.org/learn/open-source-mlops-platforms-duke/home/week/1)
