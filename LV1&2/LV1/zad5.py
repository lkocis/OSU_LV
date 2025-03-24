'''Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?'''

ham_words = 0
spam_words = 0
ham_messages = 0
spam_messages = 0
spam_with_exclamation = 0

with open('SMSSpamCollection.txt', 'r') as f:
    for line in f:
        line = line.strip()

        tag, message = line.split('\t', 1)

        words = message.split()
        word_count = len(words)

        if tag == 'ham':
            ham_messages += 1
            ham_words += word_count
        elif tag == 'spam':
            spam_messages += 1
            spam_words += word_count

            if message.endswith('!'):
                spam_with_exclamation += 1


average_ham = ham_words / ham_messages
average_spam = spam_words / spam_messages

print(f'Average ham: {average_ham}')
print(f'Average spam: {average_spam}')
print(f'Spam with exclamation: {spam_with_exclamation}')
