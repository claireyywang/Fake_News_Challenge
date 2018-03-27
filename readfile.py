from csv import DictReader

def parse_file(filename):
  rows = []
  with open(filename, encoding='utf-8') as table:
    r = DictReader(table)
    for line in r:
      rows.append(line)
  return rows


def read_file(stances_file, articles_file):
  stances = parse_file(stances_file)
  body_texts = parse_file(articles_file)
  articles = dict()

  for s in stances:
    s['Body ID'] = int(s['Body ID'])

  for t in body_texts:
    articles[int(t['Body ID'])] = t['articleBody']
    
  return stances, articles 
