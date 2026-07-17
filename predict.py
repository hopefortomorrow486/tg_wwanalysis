import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from scipy.special import softmax

os.chdir("C:\\Users\\UCCU\\Desktop\\project\\telegram")
label=["vg","ww","cult","ota","crazy"]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

model_dir="./predictroles_final"
tokenizer=BertTokenizer.from_pretrained(model_dir)
model=BertForSequenceClassification.from_pretrained(model_dir)
# Move model to appropriate device (e.g., GPU if available)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

conv=""
print("Type \"clear\" to reset the list, \"breaking\" to end the program.")
while (conv != "breaking"):
    conv=""
    listofconversation=[]
    while (conv!="clear"):
        print("Please type your input here: ")
        conv=input()
        conv=conv.lower()
        if (conv=="clear"):
            continue
        elif (conv=="breaking"):
            break

        listofconversation.append(conv)

        inputs=tokenizer(
            listofconversation,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs={key: val.to(device) for key, val in inputs.items()}
    
        # Predict with no gradients
        with torch.no_grad():
            output=model(**inputs)
            logits=output.logits
            batch_preds=torch.argmax(logits, dim=-1).cpu().numpy()  # Move to CPU immediately

        print(listofconversation)
        scores=output[0][0].detach().cpu().numpy()
        scores=softmax(scores)

        for num,finalscore in enumerate(scores):
            print(f"{label[num]}: {finalscore}")