
""" generate_data.py: recreate noisy-STT dataset (option C).
Usage:
    python src/generate_data.py --train_out ../data/train.jsonl --dev_out ../data/dev.jsonl
"""
import argparse, json, random
from pathlib import Path
random.seed(42)
first_names = ["arjun","sarah","robert","li","maria","omar","anjali","kumar","linda","mike","raj","nina","sunita","lee","alex"]
last_names = ["patel","smith","wang","garcia","singh","khan","doe","brown","chen","jain","owens","lee"]
cities = ["mumbai","bangalore","new york","los angeles","san francisco","delhi","chennai","kolkata","tokyo","london"]
locations = ["central park","gateway of india","marina bay sands","red fort","eiffel tower","hyde park"]
email_domains = ["gmail dot com", "yahoo dot com", "hotmail dot com", "outlook dot com", "proton dot me"]
hesitations = ["uh","umm","hmm","you know","like","so"]
digits_words = { "0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine" }

def spell_out_number(num_str):
    return " ".join(digits_words.get(ch, ch) for ch in num_str)

def noisy_phone():
    num = "".join(str(random.randint(0,9)) for _ in range(10))
    return " ".join(spell_out_number(ch) for ch in num)

def noisy_credit_card():
    num = "".join(str(random.randint(0,9)) for _ in range(16))
    groups = [num[i:i+4] for i in range(0,16,4)]
    spoken = " ".join(" ".join(digits_words[d] for d in g) for g in groups)
    return "card number " + spoken

def noisy_email():
    name = random.choice(first_names)+"."+random.choice(last_names)
    name = name.replace(".", " dot ")
    domain = random.choice(email_domains)
    return f"{name} at {domain}"

def noisy_date():
    day = random.randint(1,28)
    month = random.choice(["january","february","march","april","may","june","july","august","september","october","november","december"])
    year = random.randint(1970,2022)
    spoken = f"{day} {month} {year}"
    return spoken

def pick_name():
    return random.choice(first_names) + " " + random.choice(last_names)

def make_example(template_parts, entities):
    text = ""
    offsets = []
    for part in template_parts:
        if part.startswith("{ENT"):
            idx = int(part[4:-1])
            ent_text = entities[idx]["text"]
            if len(text)>0 and not text.endswith(" "):
                text += " "
            start = len(text)
            text += ent_text
            end = len(text)
            offsets.append({"start": start, "end": end, "label": entities[idx]["label"]})
        else:
            if len(text)>0 and not text.endswith(" "):
                text += " "
            text += part
    return text.strip(), offsets

def main(train_out, dev_out):
    entity_types = ["CREDIT_CARD","PHONE","EMAIL","PERSON_NAME","DATE","CITY","LOCATION"]
    train_examples = []
    dev_examples = []
    for i in range(1000):
        lab = random.choice(entity_types)
        if lab=="PHONE":
            ent = noisy_phone()
        elif lab=="CREDIT_CARD":
            ent = noisy_credit_card()
        elif lab=="EMAIL":
            ent = noisy_email()
        elif lab=="PERSON_NAME":
            ent = pick_name()
        elif lab=="DATE":
            ent = noisy_date()
        elif lab=="CITY":
            ent = random.choice(cities)
        else:
            ent = random.choice(locations)
        text = f"{random.choice(hesitations)} my {lab.lower()} is {ent}"
        train_examples.append({"id": f"utt_gen_{i}", "text": text, "entities": [{"start": text.find(ent), "end": text.find(ent)+len(ent), "label": lab}]})
    for i in range(200):
        lab = random.choice(entity_types)
        if lab=="PHONE":
            ent = noisy_phone()
        elif lab=="CREDIT_CARD":
            ent = noisy_credit_card()
        elif lab=="EMAIL":
            ent = noisy_email()
        elif lab=="PERSON_NAME":
            ent = pick_name()
        elif lab=="DATE":
            ent = noisy_date()
        elif lab=="CITY":
            ent = random.choice(cities)
        else:
            ent = random.choice(locations)
        text = f"{random.choice(hesitations)} my {lab.lower()} is {ent}"
        dev_examples.append({"id": f"utt_dev_{i}", "text": text, "entities": [{"start": text.find(ent), "end": text.find(ent)+len(ent), "label": lab}]})
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(dev_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, 'w', encoding='utf-8') as f:
        for obj in train_examples:
            f.write(json.dumps(obj, ensure_ascii=False) + '\\n')
    with open(dev_out, 'w', encoding='utf-8') as f:
        for obj in dev_examples:
            f.write(json.dumps(obj, ensure_ascii=False) + '\\n')

if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else '../data/train.jsonl', sys.argv[2] if len(sys.argv)>2 else '../data/dev.jsonl')
