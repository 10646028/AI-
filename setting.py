from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-xsmall")

# 始まりの文章
first_sentence = "遠い昔、はるか彼方の銀河系で"
# 入力
x = tokenizer.encode(first_sentence, return_tensors="pt", add_special_tokens=False)

y = model.generate(x, max_length=100)
generated_sentence = tokenizer.batch_decode(y, skip_special_tokens=True)
print(generated_sentence)