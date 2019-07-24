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
for _text in text_sent:
  for splited_text in _text.split():
    splited_text = splited_text.strip()
    print(splited_text)
    if my_dict.get(splited_text) is not None:
      my_dict[splited_text] = my_dict[splited_text] + 1
    else:
      my_dict[splited_text] = 1

print(my_dict)
