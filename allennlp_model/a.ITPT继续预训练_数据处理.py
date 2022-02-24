import warnings
import pandas as pd

warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/chinese_question_sim/train.csv', sep='\t')
test_data = pd.read_csv('data/chinese_question_sim/test.csv', sep='\t')
train_data['text'] = train_data['text_a'].apply(lambda x: x if x[-1] in ["？", "。", "！", "，"] else x+"。") + train_data['text_b']
test_data['text'] = test_data['text_a'].apply(lambda x: x if x[-1] in ["？", "。", "！", "，"] else x+"。") + test_data['text_b']
data = pd.concat([train_data, test_data])
data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))

text = '\n'.join(data.text.tolist())

with open('data/chinese_question_sim/chinese_question_for_IPTP_train_text.txt', 'w') as f:
    f.write(text)

