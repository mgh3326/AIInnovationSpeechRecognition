text = """A barber is a person. 
a barber is good person. 
a barber is huge person. 
he Knew A Secret! 
The Secret He Kept is huge secret. 
Huge secret. 
His barber kept his word. 
a barber kept his word. 
His barber kept his secret. 
But keeping and keeping such a huge secret to himself was driving the barber crazy. 
the barber went up a huge mountain."""
text = text.replace(".", " .").replace("!", " !")
text_sent = text.lower().split("\n")
my_dict = {}
sent = []
for _text in text_sent:
  temp = []
  for splited_text in _text.split():
    splited_text = splited_text.strip()
    print(splited_text)
    if my_dict.get(splited_text) is not None:
      my_dict[splited_text] = my_dict[splited_text] + 1
    else:
      my_dict[splited_text] = 1
    temp.append(splited_text)
  sent.append(temp)

print(my_dict)
my_dict_sorted = sorted(my_dict.items(), reverse=True, key=lambda t: t[1])
print(my_dict_sorted)
index_dict = {}

for idx, (word, word_count) in enumerate(my_dict_sorted):
  index_dict[word] = idx
print(index_dict)
a = [0] * len(index_dict)
a[index_dict["keeping"]] = 1
print(a)
for i in range(len(index_dict)):
  print(i * "0" + "1" + (len(index_dict) - (i + 1)) * "0")
send_index = sent
for i in send_index:
  for j in i:
    i[i.index(j)] = index_dict[j]
print(send_index)
