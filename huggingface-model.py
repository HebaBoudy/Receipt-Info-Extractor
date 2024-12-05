from transformers import AutoModel

model = AutoModel.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True)  

model = model.eval()
# input your test image
image_file = 'imgs/Picture1.png'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')
