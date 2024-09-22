import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import default_data_collator
from itertools import chain
from datasets import load_from_disk, load_dataset, IterableDataset

class CustomDataset(Dataset):
    def __init__(self, tokens, block_size=2048):
        self.full_tokens = tokens
        self.block_size = block_size
        self._len = len(tokens["input_ids"][0]) // block_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx+1) * self.block_size

        input_ids = self.full_tokens["input_ids"][0, start_idx:end_idx]
        labels = input_ids.clone()
        attn_mask = self.full_tokens["attention_mask"][0, start_idx:end_idx]
        output = {"input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels}
        return output

def get_block_size(args, tokenizer):
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                f"[WARNING] The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            print(
                f"[WARNING] The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    return block_size

def preprocess_split(dataset, tokenizer, block_size, column_name="text"):

    column_name = "text"
    def tokenize_fn(examples):
        return tokenizer(examples[column_name], return_token_type_ids=False)
    
    map_kwargs = { 
        "num_proc": None,
        "load_from_cache_file": True,
        "desc": "Running tokenizer on dataset",
    }

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    print(f"[Load Dataset]grouped train dataset")
    map_kwargs["desc"] = f"Grouping texts in chunks of {block_size}"
    dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
    )
    return dataset

def _get_column_names(dataset):
    if hasattr(dataset, "column_names"):
        return dataset.column_names
    else:
        return next(iter(dataset.take(1))).keys()

def get_column_name(dataset):
    column_names = _get_column_names(dataset)
    if "text" in column_names:
        return "text"
    else:
        return column_names[0]

def preprocess_gptq_split(dataset, tokenizer, block_size):
    column_name = get_column_name(dataset)
    tokens = tokenizer("\n\n".join(dataset[column_name]), return_tensors="pt")
    dataset = CustomDataset(tokens, block_size)
    return dataset

class DatasetBuilder:
    def __init__(self, args):
        self.dataset = {}
        self.data_cache_dir = args.dataset_cache_dir
        self.seed = args.seed

        self.dataset_name = args.calib_dataset_name
        self.dataset_config_name = args.calib_dataset_config_name
        self.dataset_preprocessor = args.calib_dataset_preprocessor
        self.dataset_split = args.calib_dataset_split
        
        self.test_dataset_name = args.eval_dataset_name
        self.test_dataset_config_name = args.eval_dataset_config_name
        self.test_dataset_split = args.eval_dataset_split
        self.test_dataset_preprocessor = args.eval_dataset_preprocessor
        
        #self.test_dataset_name = None
        #self.test_dataset_config_name = None
        #self.test_dataset_split = None
        #self.test_dataset_preprocessor = None
        #assert self.dataset_name is not None

        if (self.test_dataset_name is None) ==\
                (self.test_dataset_config_name is None) ==\
                (self.test_dataset_split is None):
            pass
        else:
            raise ValueError(
                'If one of `test_dataest_name`, `test_dataset_config_name`, '
                '`test_dataset_split` is specified, '
                'all the others must be also specified.'
            )

        if self.test_dataset_name is None and self.test_dataset_preprocessor is not None:
            raise ValueError(
                '`test_dataset_preprocessor` can be specified only when '
                '`test_dataset_name` is also specified'
            )

        if self.test_dataset_name is None:
            if self.dataset_split is not None:
                raise ValueError(
                    'Test dataset is not specified. '
                    'This requires `dataset_split` to be None so that '
                    'both train and test split can be retrieved from the same default dataset.'
                )

            self.test_dataset_preprocessor = self.dataset_preprocessor

        self.train_batch_size = args.per_device_calib_batch_size
        self.train_dataloader = None

        self.eval_batch_size = args.per_device_eval_batch_size
        self.test_dataloader = None
            


    def make_dataset(self, tokenizer=None, args=None, column_name="text", shuffle=True, **kwargs):
        print(f'{self.dataset_name} cache {self.data_cache_dir} config {self.dataset_config_name}')
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.dataset_split,           
            cache_dir=self.data_cache_dir,
            **kwargs,
        )
        if "train" not in dataset:
            self.dataset["train"] = dataset
        else:
            self.dataset = dataset
        collate_fn=None
        if self.dataset_preprocessor == 'gpt2':
            self.dataset["train"] = self.preprocess_data("train", tokenizer, args)
            collate_fn = default_data_collator
        if self.dataset_preprocessor == "gptq":
            self.dataset["train"] = self.gptq_preprocess_data("train", tokenizer, args)
            collate_fn = default_data_collator
        if "train" in self.dataset.keys():
            generator = None
            is_iterable_dataset = isinstance(self.dataset['train'], IterableDataset)
            if shuffle:
                generator = torch.Generator()
                generator.manual_seed(self.seed)
                if is_iterable_dataset:  # Need to manually shuffle it when the dataset is an iterable dataset.
                    self.dataset["train"] = self.dataset["train"].shuffle(seed=self.seed)

            self.train_dataloader = DataLoader(
                self.dataset["train"], shuffle=shuffle and not is_iterable_dataset, 
                batch_size=self.train_batch_size, collate_fn=collate_fn, generator=generator,
            )

        if self.test_dataset_name is not None:
            self.dataset["test"] = load_dataset(
                self.test_dataset_name,
                self.test_dataset_config_name,
                split=self.test_dataset_split,           
                cache_dir=self.data_cache_dir,
                **kwargs,
            )
        collate_fn=None
        if self.test_dataset_preprocessor == 'gpt2':
            self.dataset["test"] = self.preprocess_data("test", tokenizer, args)
            collate_fn = default_data_collator
        if self.test_dataset_preprocessor == 'gptq':
            self.dataset["test"] = self.gptq_preprocess_data("test", tokenizer, args)
            collate_fn = default_data_collator
        if "test" in self.dataset.keys():
            self.test_dataloader = DataLoader(
                self.dataset["test"], shuffle=False, batch_size=self.eval_batch_size, collate_fn=collate_fn,
            )
        else:
            self.test_dataloader = DataLoader(self.dataset['train'], shuffle=False, batch_size=self.train_batch_size, collate_fn=collate_fn)


    def preprocess_data(self, split, tokenizer, args):
        assert args is not None
        assert tokenizer is not None
        print(f"[Load Dataset]tokenized {split} dataset")
        block_size = get_block_size(args, tokenizer)
        
        return preprocess_split(self.dataset[split], tokenizer, block_size)


    def gptq_preprocess_data(self, split, tokenizer, args):
        assert args is not None
        assert tokenizer is not None
        print(f"[Load Dataset]tokenized {split} dataset in the GPTQ style.")
        block_size = get_block_size(args, tokenizer)
        self.dataset[split] = preprocess_gptq_split(self.dataset[split], tokenizer, block_size)

        return self.dataset[split]


    def get_dataset(self):
        return self.dataset

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader
