import os
import copy
import json
import logging
import torch
from torch.utils.data import TensorDataset
# 导入自己写的
from utils import get_intent_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    一个单独的序列分类样本实例
    一个样本完全可以用一个dict来表示，但是使用 InputExample 类，作为一个python类，具有一些方便之处
    Args:
        guid: 样本实例的唯一id
        words: list. 序列的字（中文）或者词（英文）
        intent_label: (Optional) string. 样本实例的意图标签
    """

    def __init__(self, guid, words, intent_label=None, ):
        self.guid = guid  # 每个样本的唯一序号
        self.words = words  # 样本的输入序列
        self.intent_label = intent_label  # 样本的CLS标签

    def __repr__(self):
        # 默认为： "<类名.object at 内存地址>"这样的信息表示这个实例，如：<__main__.InputExample at 0x7f50514d7dd0>；
        # 这里重写成了想要输出的信息；
        # print(InputExample)时显示；
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


class ClsProcessor(object):
    """
    Processor for the BERT classfication data set
    BERT分类任务的数据集处理器
    """

    def __init__(self, args):
        self.args = args
        # 读出已经整理好的意图标签；
        self.intent_labels = get_intent_labels(args)  # list
        # 每个数据集的文件夹里面，数据格式是一致的，文件名也一致
        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'

    # 按行读取文件
    @classmethod  # classmethod修饰符对应的函数不需要实例化，不需要self参数，但第一个参数需要是表示自身类的cls参数，可以来调用类的属性，类的方法，实例化对象等。
    def _read_file(cls, input_file, quotechar=None):
        """
        Reads a tab separated value file.
        读一个文件，以行为单位，先把每行读出来，读字段是后续的事情
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines  # list

    def _create_examples(self, texts, intents, set_type):
        """
        Creates examples for the training and dev sets.
		创建训练集和验证集
        Args:
            texts: list. 需要处理的文本组成的列表
            intents: list. 意图label组成的列表
            set_type: str. 数据集类型：训练train/验证dev/测试集test
        """
        examples = []
        for i, (text, intent) in enumerate(zip(texts, intents)):
            guid = "{}-{}".format(set_type, i)   # 给每个样本一个编号
            # 1. input_text
            words = text.split()  # 以空格分词(中文不适用)，后面还会再用tokenizer分一次，我感觉多此一举
            # 2. intent
            # 如果不在已知的意图类别中，则归为"UNK"，intent_label为数字
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")

            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label))
        # print(examples)
        return examples  # list. 格式：[{'guid':guid, 'words':[word1, word2, ...], 'intent_label':intent_label},{...}]

    def get_examples(self, mode):
        """
        Args:
            mode: 区分训练train/验证dev/测试集test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     set_type=mode)


# 如果有多个数据集，则数据集的processor可以通过映射得到
processors = {
    "atis": ClsProcessor,
    "sentiment": ClsProcessor,  # 情绪分类
    "snips": ClsProcessor
}


class InputFeatures(object):
    """
    A single set of features of data.
    输入数据特征 --→ JSON字符串
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id):
        self.input_ids = input_ids  # 输入样本序列在bert词表里的索引，可以直接喂给nn.embedding
        self.attention_mask = attention_mask  # 注意力mask，padding的部分为0，其他为1
        self.token_type_ids = token_type_ids  # 表示每个token属于句子1还是句子2（值为0或1）,单句分类任务值都是0
        self.intent_label_id = intent_label_id  # 意图标签序号

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    将之前读取的数据进行添加[CLS],[SEP]标记，padding等操作
    args:
        examples: 样本实例列表
        pad_token_label_id: Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        cls_token_segment_id： 取0
        sequence_a_segment_id： 取0
        pad_token_segment_id： 取0
        mask_padding_with_zero： attention mask;
    """
    # Setting based on the current model type
    # 这里以BERT tokenizer为例
    cls_token = tokenizer.cls_token  # [CLS]
    sep_token = tokenizer.sep_token  # [SEP]
    unk_token = tokenizer.unk_token  # [UNK]
    pad_token_id = tokenizer.pad_token_id  # [PAD]编号为0

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize words 分词
        tokens = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)  # 分词，中文会按字分
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # 如果句子长了就截断。我认为过于粗暴，会损失长句的很多信息（是否有更好的办法？）
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # 把token转化为id（通过字典）
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # check长度是否符合
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          ))

    return features  # list. 里面元素为dict


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)  # 即：ClsProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),  # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。pop()返回的是移除的值。
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file) and False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
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

        # 添加[CLS],[SEP]标记、input_id、padding等操作
        features = convert_examples_to_features(examples,
                                                args.max_seq_len,
                                                tokenizer,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)  # 保存特征数据

    # Convert to Tensors and build dataset
    # 将features转化为tensor
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    if mode == 'train':
        print(all_input_ids)
        print(all_input_ids.size())
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)

    # 将各种tensor打包，类似zip，要求各 tensor 第一维相等(样本数量)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids)
    return dataset
