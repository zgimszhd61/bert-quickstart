!pip install transformers

from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

def generate_couplet(top_sentence):
    # 将上联中的每个字替换为[MASK]，生成下联
    masked_sentence = " ".join(["[MASK]"] * len(top_sentence))
    # 将上联和下联拼接起来
    input_sentence = top_sentence + " [SEP] " + masked_sentence
    # 对拼接后的句子进行编码
    input_ids = tokenizer.encode(input_sentence, return_tensors='pt')

    # 为下联中的每个[MASK]生成预测
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]

    # 解码生成的下联
    predicted_index = torch.argmax(predictions[0, len(top_sentence)+1:], dim=1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_index)
    return "".join(predicted_tokens)

# 示例上联
top_sentence = "晚风摇树树还挺"
# 生成下联
bottom_sentence = generate_couplet(top_sentence)
print("上联：", top_sentence)
print("下联：", bottom_sentence)
