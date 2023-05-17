from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
generator = pipeline('text-generation',
                     tokenizer=tokenizer,
                     model="/Users/zhaosanqiang/PycharmProjects/dummy/ft/code/logs/ft/")

prompt = """export my_func
x = 2323

collection =
  height: 32434"""
result = generator(prompt, max_length=32, num_return_sequences=1, num_beams=4)
generated_text = result[0]["generated_text"]
print(generated_text)