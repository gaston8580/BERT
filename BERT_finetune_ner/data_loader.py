import os
import copy
import json
import logging
import torch
from torch.utils.data import TensorDataset
from utils import get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    一个单独的序列分类样本实例
    一个样本完全可以用一个dict来表示，但是使用 InputExample 类，作为一个python类，具有一些方便之处
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, slot_labels=None):
        self.guid = guid  # 每个样本的独特的序号
        self.words = words  # 样本的输入序列
        self.slot_labels = slot_labels  # 样本的NER标签

    def __repr__(self):
        # 默认为： “类名+object at+内存地址”这样的信息表示这个实例；
        # 这里重写成了想要输出的信息；
        # print(input_example) 时候显示；
        return str(self.to_json_string())

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        实例序列化为dict
        """
        # __dict__：
        # 类 的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的
        # 对象实例的__dict__中存储了一些self.xxx的一些东西
        # 参见 https://www.cnblogs.com/starrysky77/p/9102344.html

        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        实例序列化为JSON字符串
        """
        # 类的性质等信息dump进入json string
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    输入数据特征→JSON字符串
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        self.input_ids = input_ids  # 输入样本序列在bert词表里的索引，可以直接喂给nn.embedding
        self.attention_mask = attention_mask  # 注意力mask，padding的部分为0，其他为1
        self.token_type_ids = token_type_ids  # 表示每个token属于句子1还是句子2（值为0或1）,单句分类任务值都是0
        self.slot_labels_ids = slot_labels_ids  # NER标签（复数）序号。

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NerProcessor(object):
    """
    Processor for the JointBERT data set
    BERT的NER任务数据处理器
    """

    def __init__(self, args):
        self.args = args
        self.slot_labels = get_slot_labels(args)  # 读出整理好的ner标签。list

        self.input_text_file = 'seq.in'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, slot) in enumerate(zip(texts, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


processors = {
    "atis": NerProcessor,
    "snips": NerProcessor
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 slot_label_lst=None,
                                 ):
    """
    将之前读取的数据进行添加[CLS],[SEP]标记，padding等操作
    args:
        examples: 样本实例列表
        pad_token_label_id: Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        cls_token_segment_id： 取0
        sequence_a_segment_id： 取0
        pad_token_segment_id： 取0
        mask_padding_with_zero： attention mask
        slot_label_lst: ner标签
    """
    # Setting based on the current model type
    # 以BERT tokenizer为例
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 分词
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

            # A——word : B-PER;
            # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, PAD, PAD;
            # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, I-PER, I-PER;

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            # pad_token_label_id: -100, loss function  忽略的label编号

        # Account for [CLS] and [SEP]
        # 记录[CLS]和[SEP]
        special_tokens_count = 2
        # 如果句子长了就截断
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]  # [SEP] label: pad_token_label_id
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids  # 将[CLS]的pad_token_label_id加到slot_labels_ids最前面
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # 把token转化为id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        if 1198 < ex_index < 1200:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("original words: %s" % " ".join([str(x) for x in example.words]))
            logger.info("original slot_labels: %s" % " ".join([slot_label_lst[int(x)] for x in example.slot_labels]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)  # 即：NerProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。pop()返回的是移除的值。
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        slot_label_lst = get_slot_labels(args)

        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        # 添加[CLS],[SEP]标记、input_id、slot_labels_ids、padding等操作
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id,
                                                slot_label_lst=slot_label_lst)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    # 将特征转化为tensor
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    # 将各种tensor打包，类似zip，要求各 tensor 第一维相等(样本数量)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_slot_labels_ids
                            )
    return dataset
