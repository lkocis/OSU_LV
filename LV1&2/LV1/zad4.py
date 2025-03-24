'''Napišite Python skriptu koja ce u citati tekstualnu datoteku naziva ˇ song.txt.
Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.'''

import string

with open('song.txt', 'r') as f:
    text = f.read()

words = text.split()
dictionary = {}
counter = 0
counterWords = 0

for i in range (len(words)):
    counter = 0
    for j in range (len(words)):
        if words[i] == words[j]:
            counter += 1
    dictionary[words[i]] = counter
    if counter == 1:
        print(words[i])
        counterWords += 1
print(f"Unique words num: {counterWords}")
