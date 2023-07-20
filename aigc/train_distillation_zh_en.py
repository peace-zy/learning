# Necessary imports
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm
import os
from transformers import BertTokenizer, BertModel
from transformers import CLIPTokenizer, CLIPTextModel
import traceback
import re
import logging

logging.basicConfig(filename='train.log',level=logging.DEBUG)
# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#class ZhEnDataset(Dataset):
class ZhEnDataset(IterableDataset):
    '''
    def __init__(self, data_file, teacher_tokenizer, student_tokenizer):
        self.data_file = data_file
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.num = 100
    '''
    def __init__(self, data_file):
        self.data_file = data_file
        self.num = 396053

    # need to overload
    def __len__(self):
        return self.num

    def str_replace(self, data):
        """ 把写错的中文符号都替换成英文 """
        chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”']
        englishTab = [':', ';', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"']
        for index in range(len(chinaTab)):
            if chinaTab[index] in data:
                data = data.replace(chinaTab[index], englishTab[index])
        data = re.sub(r"([,])\1+", ",", data)
        return data

    # need to overload
    def __iter__(self):
        with open(self.data_file, 'r', errors='ignore', encoding="utf8") as f:
            for line in f:
                try:
                    fields = line.strip().split('\t')
                    zh, en = fields[0], fields[1]
                    en = self.str_replace(en)
                    zh = re.sub(r'[^\w\s]', ',', zh)
                    zh = re.sub(r"([,])\1+", ",", zh)

                    yield {'en': en, 'zh': zh}
                except Exception as e:
                    print(traceback.print_exc())
                    continue

train_data_file = 'en_zh_train_data.txt'
train_data_file = '/ssd2/zhangyan75/project/trans_zh_en/part1.txt'
train_dataset = ZhEnDataset(train_data_file)
test_data_file = 'en_zh_eval_data.txt'
test_data_file = 'en_zh_train_data.txt'
test_dataset = ZhEnDataset(test_data_file)
# Create train and test dataloaders
#train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
#train_loader = DataLoader(dataset=train_dataset, batch_size=256)
train_loader = DataLoader(dataset=train_dataset, batch_size=128)
test_loader = DataLoader(dataset=test_dataset, batch_size=16)
class StudentModel(nn.Module):
    def __init__(self, model_path='student'):
        super(StudentModel, self).__init__()
        self.model_path = model_path
        print(self.model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder")


class TeacherModel(nn.Module):
    def __init__(self, model_path='teacher'):
        super(TeacherModel, self).__init__()
        self.model_path = model_path
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder")

        for param in self.text_encoder.parameters():
            param.requires_grad = False

def check_diff(loader, teacher, student, distillation_loss_fn, device):
    losses = []
    teacher.eval()
    student.eval()
    max_length = 77
    #norm = nn.LayerNorm(768, elementwise_affine=False).to(device)

    with torch.no_grad():
        for data in loader:
            zh, en = data['zh'], data['en']
            en_tokens = teacher.tokenizer(
                            en,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt',
                        )['input_ids'].to(device)

            zh_tokens = student.tokenizer(
                            zh,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt',
                        )['input_ids'].to(device)

            teacher_preds = teacher.text_encoder(en_tokens)[0]
            student_preds = student.text_encoder(zh_tokens)[0]
            #teacher_preds = norm(teacher_preds)
            #student_preds = norm(student_preds)
            #loss = distillation_loss_fn(student_preds, teacher_preds)
            #teacher_preds = teacher_preds / teacher_preds.norm(dim=-1, keepdim=True)
            #student_preds = student_preds / student_preds.norm(dim=-1, keepdim=True)
            #print(student_preds.max(), student_preds.min())
            #print(teacher_preds.max(), teacher_preds.min())
            diff = torch.abs(student_preds - teacher_preds)
            #print(diff.min(), diff.max(), diff.mean(), diff.std())
            loss = distillation_loss_fn(student_preds, teacher_preds)
            #losses.append(loss.sum().item())
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    student.train()
    return avg_loss

def train_step(
    teacher,
    student,
    optimizer,
    distillation_loss_fn,
    epoch,
    device
):
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), position=0, leave=True, desc=f"Epoch {epoch}")
    step = 0
    max_length = 77
    for data in pbar:
        # Get data to cuda if possible
        zh, en = data['zh'], data['en']
        en_tokens = teacher.tokenizer(
                        en,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt',
                    )['input_ids'].to(device)

        zh_tokens = student.tokenizer(
                        zh,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt',
                    )['input_ids'].to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher.text_encoder(en_tokens)[0]

        student_zh_preds = student.text_encoder(zh_tokens)[0]
        student_en_preds = student.text_encoder(en_tokens)[0]

        #teacher_preds = teacher_preds / teacher_preds.norm(dim=-1, keepdim=True)
        #student_preds = student_preds / student_preds.norm(dim=-1, keepdim=True)
        #diff = torch.abs(student_preds - teacher_preds)
        #print(diff.max(), diff.min())


        loss = distillation_loss_fn(torch.cat([student_en_preds, student_zh_preds]),  torch.cat([teacher_preds] * 2))
        losses.append(loss.item())
        #losses.append(loss.sum().item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        #loss.backward(torch.ones_like(loss))

        optimizer.step()
        if step % 10 == 0:
            print('{}\t{}'.format(step, loss.sum().item()))
            logging.info('{}\t{}'.format(step, loss.sum().item()))
        step += 1

    avg_loss = sum(losses) / len(losses)
    return avg_loss

def train(epoches, teacher, student):
    output_dir = 'out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    teacher = teacher.to(device)
    student = student.to(device)
    distillation_loss_fn = nn.MSELoss(reduction='mean')
    #distillation_loss_fn = nn.MSELoss(reduction='none')
    #distillation_loss_fn = nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    teacher.eval()
    student.train()
    for epoch in range(epoches):
        loss = train_step(
            teacher,
            student,
            optimizer,
            distillation_loss_fn,
            epoch,
            device
        )
        #acc = check_accuracy(test_loader, student, device)
        eval_loss = check_diff(test_loader, teacher, student, distillation_loss_fn, device)
        print(f"Epoch:{epoch}\tTrain Loss:{loss:.6f}\tEval Loss:{eval_loss:.6f}")
        outfile = os.path.join(output_dir,
                "epoch_{}_train_{}_eval_{}_student".format(epoch, loss, eval_loss))
        #torch.save(student.state_dict(), outfile)
        student.text_encoder.save_pretrained(os.path.join(outfile, 'text_encoder'))
        student.tokenizer.save_pretrained(os.path.join(outfile, 'tokenizer'))


def main():
    teacher_model = '/xx/bra_v5_en'
    teacher = TeacherModel(teacher_model)
    student_model = '/xx/bra_v5_en'
    #student_model = 'out/epoch_33_train_0.5321654837865096_eval_0.714129701256752_student.pt'
    student = StudentModel(student_model)
    kwargs = {
        'epoches': 20,
        'teacher': teacher,
        'student': student,
    }
    train(**kwargs)
    '''
    distillation_loss_fn = nn.MSELoss(reduction='mean')
    eval_loss = check_diff(test_loader, teacher.to(device), student.to(device), distillation_loss_fn, device)
    print(eval_loss)
    '''
    '''
    student = StudentModel()
    teacher = TeacherModel()
    st_weigth = os.path.join(output_dir, "student.pkl")
    print(st_weigth)
    state_dict = torch.load(st_weigth)
    student.load_state_dict(state_dict)
    te_weigth = os.path.join(output_dir, "teacher.pkl")
    print(te_weigth)
    state_dict = torch.load(te_weigth)
    teacher.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = teacher.to(device)
    student = student.to(device)
    acc = check_accuracy(test_loader, teacher, device)
    print('teacher acc={}'.format(acc))
    acc = check_accuracy(test_loader, student, device)
    print('student acc={}'.format(acc))
    '''

if __name__ == '__main__':
    main()
